"""Monkey patch Diffusers to pass time embeddings into BiMamba self-attn."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

_PATCH_APPLIED = False


def apply_mamba_time_patch() -> bool:
    """Apply runtime patches; return True if applied now."""
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return False

    try:
        from diffusers.models.attention import BasicTransformerBlock
        from diffusers.models.transformers.transformer_temporal import TransformerSpatioTemporalModel
        from diffusers.models.unets.unet_3d_blocks import (
            CrossAttnDownBlockSpatioTemporal,
            CrossAttnUpBlockSpatioTemporal,
            UNetMidBlockSpatioTemporal,
        )
        from diffusers.models.attention import _chunked_feed_forward
    except Exception as exc:  # pragma: no cover - optional runtime patch
        raise RuntimeError(f"Failed to import diffusers for patching: {exc}") from exc

    use_attn1_time_emb = os.getenv("MAMBA_SELF_ATTN_TIME_EMB", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    # --- BasicTransformerBlock.forward (pass time_emb only to attn1) ---
    def _basic_transformer_forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        timestep=None,
        cross_attention_kwargs=None,
        class_labels=None,
        added_cond_kwargs=None,
        time_emb: Optional["torch.Tensor"] = None,
    ):
        import torch  # local import to avoid hard dependency at module import time
        from diffusers.utils import logging

        logger = logging.get_logger(__name__)

        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored."
                )

        batch_size = hidden_states.shape[0]

        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.norm_type == "ada_norm_zero":
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
            norm_hidden_states = self.norm1(hidden_states)
        elif self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif self.norm_type == "ada_norm_single":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)
            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            norm_hidden_states = norm_hidden_states.squeeze(1)
        else:
            raise ValueError("Incorrect norm used")

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        attn1_kwargs = dict(cross_attention_kwargs)
        if use_attn1_time_emb and time_emb is not None and getattr(self.attn1, "supports_time_emb", False):
            attn1_kwargs["time_emb"] = time_emb

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **attn1_kwargs,
        )
        if self.norm_type == "ada_norm_zero":
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.norm_type == "ada_norm_single":
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        if self.attn2 is not None:
            if self.norm_type == "ada_norm":
                norm_hidden_states = self.norm2(hidden_states, timestep)
            elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.norm2(hidden_states)
            elif self.norm_type == "ada_norm_single":
                norm_hidden_states = hidden_states
            elif self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
            else:
                raise ValueError("Incorrect norm")

            if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        if self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif not self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm3(hidden_states)

        if self.norm_type == "ada_norm_zero":
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        if self._chunk_size is not None:
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.norm_type == "ada_norm_zero":
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.norm_type == "ada_norm_single":
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states

    BasicTransformerBlock.forward = _basic_transformer_forward

    # --- TransformerSpatioTemporalModel.forward (accept time_emb) ---
    def _transformer_spatiotemporal_forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        image_only_indicator=None,
        time_emb: Optional["torch.Tensor"] = None,
        return_dict: bool = True,
    ):
        from diffusers.models.transformers.transformer_temporal import TransformerTemporalModelOutput
        import torch

        batch_frames, _, height, width = hidden_states.shape
        num_frames = image_only_indicator.shape[-1]
        batch_size = batch_frames // num_frames

        time_context = encoder_hidden_states
        time_context_first_timestep = time_context[None, :].reshape(
            batch_size, num_frames, -1, time_context.shape[-1]
        )[:, 0]
        time_context = time_context_first_timestep[:, None].broadcast_to(
            batch_size, height * width, time_context.shape[-2], time_context.shape[-1]
        )
        time_context = time_context.reshape(batch_size * height * width, -1, time_context.shape[-1])

        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_frames, height * width, inner_dim)
        hidden_states = self.proj_in(hidden_states)

        num_frames_emb = torch.arange(num_frames, device=hidden_states.device)
        num_frames_emb = num_frames_emb.repeat(batch_size, 1)
        num_frames_emb = num_frames_emb.reshape(-1)
        t_emb = self.time_proj(num_frames_emb)

        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_pos_embed(t_emb)
        emb = emb[:, None, :]

        for block, temporal_block in zip(self.transformer_blocks, self.temporal_transformer_blocks):
            if self.training and self.gradient_checkpointing:
                if time_emb is not None:
                    def _block_forward(hs, attention_mask, enc_hs, enc_mask, te):
                        return block(
                            hs,
                            attention_mask=attention_mask,
                            encoder_hidden_states=enc_hs,
                            encoder_attention_mask=enc_mask,
                            time_emb=te,
                        )

                    hidden_states = torch.utils.checkpoint.checkpoint(
                        _block_forward,
                        hidden_states,
                        None,
                        encoder_hidden_states,
                        None,
                        time_emb,
                        use_reentrant=False,
                    )
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        block,
                        hidden_states,
                        None,
                        encoder_hidden_states,
                        None,
                        use_reentrant=False,
                    )
            else:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    time_emb=time_emb,
                )

            hidden_states_mix = hidden_states
            hidden_states_mix = hidden_states_mix + emb

            hidden_states_mix = temporal_block(
                hidden_states_mix,
                num_frames=num_frames,
                encoder_hidden_states=time_context,
            )
            hidden_states = self.time_mixer(
                x_spatial=hidden_states,
                x_temporal=hidden_states_mix,
                image_only_indicator=image_only_indicator,
            )

        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch_frames, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        output = hidden_states + residual

        if not return_dict:
            return (output,)
        return TransformerTemporalModelOutput(sample=output)

    TransformerSpatioTemporalModel.forward = _transformer_spatiotemporal_forward

    # --- UNet block forward patches to pass time_emb=temb to attn ---
    def _mid_forward(self, hidden_states, temb=None, encoder_hidden_states=None, image_only_indicator=None):
        hidden_states = self.resnets[0](
            hidden_states,
            temb,
            image_only_indicator=image_only_indicator,
        )

        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        return module(*inputs)
                    return custom_forward

                import torch
                from diffusers.utils import is_torch_version

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    time_emb=temb,
                    return_dict=False,
                )[0]
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    image_only_indicator,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    time_emb=temb,
                    return_dict=False,
                )[0]
                hidden_states = resnet(
                    hidden_states,
                    temb,
                    image_only_indicator=image_only_indicator,
                )

        return hidden_states

    UNetMidBlockSpatioTemporal.forward = _mid_forward

    def _cross_down_forward(self, hidden_states, temb=None, encoder_hidden_states=None, image_only_indicator=None):
        output_states = ()
        blocks = list(zip(self.resnets, self.attentions))
        for resnet, attn in blocks:
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        return module(*inputs)
                    return custom_forward

                import torch
                from diffusers.utils import is_torch_version

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    image_only_indicator,
                    **ckpt_kwargs,
                )
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    time_emb=temb,
                    return_dict=False,
                )[0]
            else:
                hidden_states = resnet(
                    hidden_states,
                    temb,
                    image_only_indicator=image_only_indicator,
                )
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    time_emb=temb,
                    return_dict=False,
                )[0]

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states = output_states + (hidden_states,)

        return hidden_states, output_states

    CrossAttnDownBlockSpatioTemporal.forward = _cross_down_forward

    def _cross_up_forward(
        self,
        hidden_states,
        res_hidden_states_tuple,
        temb=None,
        encoder_hidden_states=None,
        image_only_indicator=None,
    ):
        import torch

        for resnet, attn in zip(self.resnets, self.attentions):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        return module(*inputs)
                    return custom_forward

                import torch
                from diffusers.utils import is_torch_version

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    image_only_indicator,
                    **ckpt_kwargs,
                )
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    time_emb=temb,
                    return_dict=False,
                )[0]
            else:
                hidden_states = resnet(
                    hidden_states,
                    temb,
                    image_only_indicator=image_only_indicator,
                )
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    time_emb=temb,
                    return_dict=False,
                )[0]

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states

    CrossAttnUpBlockSpatioTemporal.forward = _cross_up_forward

    _PATCH_APPLIED = True
    return True
