# scripts/demo_patch_and_run.py
import torch
import torch.nn as nn
from diffusers import UNetSpatioTemporalConditionModel

from blocks.patch_diffusers import replace_transformers_with_mamba
from blocks.mamba_spatiotemporal import MambaSpatioTemporalModel

# ========== ログ専用設定 ==========
# 形状・次元の観測のみ。テンソルは絶対に変更しない。
CUR_BATCH = {"n": None}  # 参照のみ（ログ用途）


def _set_cur_batch(x: torch.Tensor, tag: str):
    if isinstance(x, torch.Tensor) and x.dim() in (4, 5):
        CUR_BATCH["n"] = int(x.shape[0])
        # 必要なら下行のコメントアウトを外す
        # print(f"[BATCH] {tag}: set CUR_BATCH={CUR_BATCH['n']}")


def hook_out(label):
    def _hook(module, inp, out):
        x = out[0] if isinstance(out, (tuple, list)) else out
        try:
            shape = tuple(x.shape)
            dim = x.dim()
        except Exception:
            shape, dim = ("<unknown>",), -1
        print(f"[SHAPE] {label}.out: dim={dim} shape={shape}")
        if isinstance(x, torch.Tensor):
            _set_cur_batch(x, f"{label}.out")
        # 出力は変更しない（戻り値なし）
    return _hook


def attach_block_hooks(unet: nn.Module):
    paths = [
        "down_blocks.0.attentions.0",
        "down_blocks.0.temp_attentions.0",
        "down_blocks.0.resnets.1.spatial_res_block.norm1",
    ]
    for p in paths:
        parent = unet
        ok = True
        for key in p.split("."):
            if not hasattr(parent, key):
                print(f"[WARN] skip missing path: {p}")
                ok = False
                break
            parent = getattr(parent, key)
        if not ok:
            continue

        if isinstance(parent, nn.GroupNorm):
            def pre(module, inp):
                x = inp[0]
                print("[GN PRE] {}: dim={} shape={}".format(p, x.dim(), tuple(x.shape)))
                _set_cur_batch(x, f"{p}.pre")
            parent.register_forward_pre_hook(pre)
        else:
            parent.register_forward_hook(hook_out(p))


def print_shape_after(name: str):
    def _hook(module, inp, out):
        x = out if isinstance(out, torch.Tensor) else out[0]
        print(f"[SHAPE] {name}.out: {tuple(x.shape)}")
        _set_cur_batch(x, f"{name}.out")
        # 出力は変更しない
    return _hook


def batch_tracker(name: str):
    """4D/5D を見たら CUR_BATCH を更新しておく軽量プリフック（ログのみ）"""
    def _hook(module, inp):
        x = inp[0]
        if isinstance(x, torch.Tensor) and x.dim() in (4, 5):
            _set_cur_batch(x, f"{name}.pre")
            # ログだけ出す（必要なら有効化）
            # print(f"[BT] {name}.pre: dim={x.dim()} shape={tuple(x.shape)}")
        # 変更しない
    return _hook


def gn_prehook_logonly(name: str):
    """
    GroupNorm 直前の観測専用フック。
    どの次元が来ても一切変更せず、dim/shape を出力するだけ。
    """
    def _hook(module, inp):
        x = inp[0]
        try:
            print(f"[GN PRE] {name}: dim={x.dim()} shape={tuple(x.shape)}")
        except Exception:
            print(f"[GN PRE] {name}: <shape unknown>")
        _set_cur_batch(x, f"{name}.pre")
        # 変更しない（戻り値なし）
    return _hook


def dsus_prehook_logonly(name: str):
    """
    Downsample2D / Upsample2D 等の入口で in.shape と期待チャンネル数をログするだけ。
    テンソルは絶対に変更しない。
    """
    def _hook(module, inp):
        x = inp[0]
        chan = getattr(module, "channels", None)
        if isinstance(x, torch.Tensor) and chan is not None:
            try:
                print(f"[DS/US PRE] {name}: in.shape={tuple(x.shape)} expect C={chan}")
            except Exception:
                print(f"[DS/US PRE] {name}: <shape unknown> expect C={chan}")
            _set_cur_batch(x, f"{name}.pre")
        # 変更しない
    return _hook


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    cfg = {
        "sample_size": 64,
        "in_channels": 4,
        "out_channels": 4,
        "down_block_types": (
            "CrossAttnDownBlockSpatioTemporal",
            "CrossAttnDownBlockSpatioTemporal",
            "CrossAttnDownBlockSpatioTemporal",
            "DownBlockSpatioTemporal",
        ),
        "up_block_types": (
            "UpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
        ),
        "block_out_channels": (160, 320, 640, 640),
        "layers_per_block": 2,
        "cross_attention_dim": 1024,
        "num_attention_heads": (4, 8, 8, 8),
        "num_frames": 3,
    }

    unet = UNetSpatioTemporalConditionModel.from_config(cfg)

    replaced = replace_transformers_with_mamba(unet, keep_mid_attn=True)
    print(f"[MambaPatch] replaced: {replaced}")

    # 代表点のログ
    attach_block_hooks(unet)
    unet.conv_in.register_forward_hook(print_shape_after("conv_in"))

    # === バッチ追跡（ログのみ） ===
    for n, m in unet.named_modules():
        if isinstance(m, (nn.Conv2d, nn.GroupNorm)):
            m.register_forward_pre_hook(batch_tracker(n))

    # === GroupNorm 直前ログ ===
    for n, m in unet.named_modules():
        if isinstance(m, nn.GroupNorm):
            m.register_forward_pre_hook(gn_prehook_logonly(n))

    # === Down/Up-sampler 入口ログ ===
    for n, m in unet.named_modules():
        if hasattr(m, "channels") and hasattr(m, "forward"):
            m.register_forward_pre_hook(dsus_prehook_logonly(n))

    # Mamba 出力形状ログ
    for n, m in unet.named_modules():
        if isinstance(m, MambaSpatioTemporalModel):
            m.register_forward_hook(print_shape_after(f"mamba[{n}]"))

    unet.to(device=device, dtype=dtype).eval()

    # ===== 入力（B,T,C,H,W） =====
    B, C, T, H, W = 2, cfg["in_channels"], cfg["num_frames"], 64, 64
    x = torch.randn(B, T, C, H, W, device=device, dtype=dtype)
    print("x shape (B,T,C,H,W):", tuple(x.shape))

    t = torch.tensor([1, 1], dtype=torch.long, device=device)
    cond = torch.randn(B, 1, cfg["cross_attention_dim"], device=device, dtype=dtype)
    added = torch.zeros(B, 3, device=device, dtype=dtype)

    with torch.no_grad():
        y = unet(
            x, t, encoder_hidden_states=cond, added_time_ids=added, return_dict=False
        )[0]

    print("OK. y shape (B, T, out_channels, H, W):", tuple(y.shape))


if __name__ == "__main__":
    main()
