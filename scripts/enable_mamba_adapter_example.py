import torch
from diffusers.models import UNetSpatioTemporalConditionModel
from mamba_temporal.patches.patch_unet_temporal import (
    attach_temporal_mamba_adapters, detach_hooks
)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        "./weights/StereoCrafter"
    ).eval().to(device)

    # Mamba2アダプタを差し込み（conv_in後 & mid後）
    handles = attach_temporal_mamba_adapters(
        unet, where=("pre_down", "post_mid"), d_state=128, headdim=64
    )

    # ダミー入力（5D: [B, F, C, H, W]）
    B, F, C, H, W = 1, 8, unet.config.in_channels, 64, 64
    x = torch.randn(B, F, C, H, W, device=device)

    # timestep は同じ device に
    timesteps = torch.tensor([10], device=device, dtype=torch.long)

    # ---- 追加：最低限の条件入力 ----
    # Cross-Attn の埋め込み（ゼロでOK）: [B, 1, cross_attention_dim]
    ctx_dim = getattr(unet.config, "cross_attention_dim", 1024)  # ない場合の保険
    encoder_hidden_states = torch.zeros((B, 1, ctx_dim), device=device, dtype=x.dtype)

    # Added Time IDs（SVDは [fps, motion_bucket_id, noise_aug_strength] が一般的）
    # 多くの実装では fps-1 を渡す流儀なので合わせます
    fps = 6.0
    motion_bucket_id = 127.0
    noise_aug_strength = 0.0
    added_time_ids = torch.tensor(
        [[fps - 1.0, motion_bucket_id, noise_aug_strength]],
        device=device, dtype=x.dtype
    )  # 形は [B, 3]

    with torch.no_grad():
        y = unet(
            x,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            added_time_ids=added_time_ids,
            return_dict=False
        )[0]

    print("✓ forward ok:", tuple(y.shape))
    detach_hooks(handles)

if __name__ == "__main__":
    main()
