# -*- coding: utf-8 -*-
# =============================================
# File: /workspace/stereocraft/scripts/test_spatiotemporal_mamba.py
# ---------------------------------------------
# 目的: 時空間Mambaの動作テスト
# =============================================

# Usage: PYTHONPATH=. python scripts/test_spatiotemporal_mamba.py
# Checks forward/backward, indicator shapes, AMP, and chunking paths.
import warnings
import sys
import time

import torch

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch.library.impl_abstract.*register_fake.*",
)

from blocks.mamba_spatiotemporal import MambaSpatioTemporalModel


def _device_dtype():
    use_cuda = torch.cuda.is_available()
    dev = torch.device("cuda" if use_cuda else "cpu")
    # fp16 はとりあえず CUDA のみ推奨
    dt = torch.float16 if use_cuda else torch.float32
    return dev, dt

def _make_model(C=160, d_model=None, ctx_dim=1280, chunk=64, keep_spatial=True, mem_eff=None):
    # Triton 依存の fused/mem‑eff カーネルは GPU 前提。
    # CUDA が無い環境だと Triton のコンパイル段階で
    # "Unexpected mma -> mma layout conversion" などで落ちる場合がある。
    # そのため CPU 実行時は安全側で use_mem_eff_path=False に切り替える。
    use_mem_eff = torch.cuda.is_available() if mem_eff is None else bool(mem_eff)
    return MambaSpatioTemporalModel(
        in_channels=C,
        d_model=d_model or C,
        cross_attention_dim=ctx_dim,
        d_state=256,
        headdim=32,
        expand=2,
        temporal_chunk_size=chunk,
        use_mem_eff_path=use_mem_eff,
        num_groups_gn=32,
        keep_spatial_mixer=keep_spatial,
    )

def _randn(shape, device, dtype):
    # 勾配テスト用に requires_grad 指定は呼び出し側で
    return torch.randn(*shape, device=device, dtype=dtype)

def test_basic_forward():
    dev, dt = _device_dtype()
    B, C, T, H, W = 2, 160, 3, 64, 64
    model = _make_model(C=C).to(dev).to(dtype=dt)
    x = _randn((B, C, T, H, W), dev, dt).requires_grad_(True)
    # encoder_hidden_states: (B, N, ctx_dim)
    ctx = _randn((B, 77, 1280), dev, dt)
    t = torch.tensor([10, 10], device=dev)

    out = model(x, encoder_hidden_states=ctx, timestep=t, image_only_indicator=None, return_dict=True)
    y = out["sample"]
    assert y.shape == x.shape, f"shape mismatch: {y.shape} vs {x.shape}"
    # 逆伝播
    loss = y.square().mean()
    loss.backward()
    # 勾配がちゃんと入るか
    assert torch.isfinite(x.grad).all(), "input grad has NaN/Inf"

def test_tuple_return():
    dev, dt = _device_dtype()
    B, C, T, H, W = 1, 160, 2, 32, 32
    model = _make_model(C=C).to(dev).to(dtype=dt)
    x = _randn((B, C, T, H, W), dev, dt)
    (y,) = model(x, return_dict=False)
    assert y.shape == x.shape

def test_indicator_variants():
    dev, dt = _device_dtype()
    B, C, T, H, W = 2, 160, 4, 32, 32
    model = _make_model(C=C).to(dev).to(dtype=dt)
    x = _randn((B, C, T, H, W), dev, dt)

    # (a) 指定なし（学習可能α）
    y0 = model(x)["sample"]

    # (b) バッチごとのスカラー (B,)
    ind_b = torch.tensor([1.0, 0.0], device=dev, dtype=dt)
    y1 = model(x, image_only_indicator=ind_b)["sample"]

    # (c) フレームごと (B,T)
    ind_bt = torch.zeros((B, T), device=dev, dtype=dt)
    ind_bt[:, : T // 2] = 1.0  # 前半 temporal 寄せ
    y2 = model(x, image_only_indicator=ind_bt)["sample"]

    for y in (y0, y1, y2):
        assert y.shape == x.shape
        assert torch.isfinite(y).all()

def test_indicator_extremes_and_paths():
    # 極端な indicator で空間/時間パスがきちんと通るか（数値の有限性のみ確認）
    dev, dt = _device_dtype()
    B, C, T, H, W = 2, 160, 3, 32, 32
    model = _make_model(C=C).to(dev).to(dtype=dt)
    x = _randn((B, C, T, H, W), dev, dt)
    sp_only = torch.zeros((B, T), device=dev, dtype=dt)
    tm_only = torch.ones((B, T), device=dev, dtype=dt)
    y_sp = model(x, image_only_indicator=sp_only)["sample"]
    y_tm = model(x, image_only_indicator=tm_only)["sample"]
    assert y_sp.shape == x.shape and y_tm.shape == x.shape
    assert torch.isfinite(y_sp).all() and torch.isfinite(y_tm).all()

def test_ctx_shapes():
    dev, dt = _device_dtype()
    B, C, T, H, W = 2, 160, 3, 32, 32
    model = _make_model(C=C).to(dev).to(dtype=dt)
    x = _randn((B, C, T, H, W), dev, dt)

    # (1) (B, N, Cctx)
    ctx_seq = _randn((B, 77, 1280), dev, dt)
    y1 = model(x, encoder_hidden_states=ctx_seq)["sample"]
    assert y1.shape == x.shape

    # (2) (B, Cctx)
    ctx_pooled = _randn((B, 1280), dev, dt)
    y2 = model(x, encoder_hidden_states=ctx_pooled)["sample"]
    assert y2.shape == x.shape

def test_no_ctx_works():
    # encoder_hidden_states なしでも動作
    dev, dt = _device_dtype()
    B, C, T, H, W = 1, 160, 2, 32, 32
    model = _make_model(C=C).to(dev).to(dtype=dt)
    x = _randn((B, C, T, H, W), dev, dt)
    y = model(x)["sample"]
    assert y.shape == x.shape

def test_timestep_variants():
    dev, dt = _device_dtype()
    B, C, T, H, W = 2, 160, 5, 16, 16
    model = _make_model(C=C).to(dev).to(dtype=dt)
    x = _randn((B, C, T, H, W), dev, dt)

    # (1) B 長さ
    t1 = torch.tensor([5, 5], device=dev)
    y1 = model(x, timestep=t1)["sample"]

    # (2) B*T 長さ
    t2 = torch.arange(B * T, device=dev)
    y2 = model(x, timestep=t2)["sample"]

    # (3) None（内部がフォールバック）
    y3 = model(x, timestep=None)["sample"]

    for y in (y1, y2, y3):
        assert y.shape == x.shape
        assert torch.isfinite(y).all()

def test_invalid_indicator_raises():
    # 不正形状の indicator は明確にエラー
    dev, dt = _device_dtype()
    B, C, T, H, W = 1, 160, 3, 16, 16
    model = _make_model(C=C).to(dev).to(dtype=dt)
    x = _randn((B, C, T, H, W), dev, dt)
    bad = torch.zeros((B, T + 1), device=dev, dtype=dt)
    try:
        model(x, image_only_indicator=bad)
        assert False, "expected ValueError for bad indicator shape"
    except ValueError:
        pass

def test_amp_autocast():
    dev, dt = _device_dtype()
    if dev.type != "cuda":
        return  # AMPはCUDA前提で簡略
    B, C, T, H, W = 2, 160, 3, 32, 32
    model = _make_model(C=C).to(dev).to(dtype=torch.float16)
    x = torch.randn(B, C, T, H, W, device=dev, dtype=torch.float16).requires_grad_(True)
    ctx = torch.randn(B, 77, 1280, device=dev, dtype=torch.float16)
    with torch.amp.autocast('cuda', dtype=torch.float16):
        y = model(x, encoder_hidden_states=ctx)["sample"]
        loss = y.mean()
    loss.backward()
    assert torch.isfinite(x.grad).all()

def test_bfloat16_autocast():
    # bfloat16 でも最低限の forward が通るか（GPU 環境のみ）
    dev, _ = _device_dtype()
    if dev.type != "cuda":
        return
    # BF16 は Ampere(sm_80) 以降のみ対応。未満ならスキップ。
    major, minor = torch.cuda.get_device_capability(0)
    if major < 8:
        return
    B, C, T, H, W = 1, 160, 2, 32, 32
    model = _make_model(C=C).to(dev).to(dtype=torch.bfloat16)
    x = torch.randn(B, C, T, H, W, device=dev, dtype=torch.bfloat16)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        y = model(x)["sample"]
    assert y.shape == x.shape
    assert torch.isfinite(y).all()

def test_long_T_with_chunk():
    dev, dt = _device_dtype()
    B, C, T, H, W = 1, 64, 64, 8, 8   # トリトンに優しい小さめ空間で長いT
    model = _make_model(C=C, d_model=64, ctx_dim=256, chunk=16, keep_spatial=True).to(dev).to(dtype=dt)
    x = _randn((B, C, T, H, W), dev, dt)
    ctx = _randn((B, 32, 256), dev, dt)
    y = model(x, encoder_hidden_states=ctx)["sample"]
    assert y.shape == x.shape
    assert torch.isfinite(y).all()

def test_mem_eff_toggle_runs():
    # メモリ効率パスの ON/OFF どちらも実行可能であることを確認
    # 注意: 非メモリ効率パス（CUDAのcausal_conv1d）では L または B_eff*L が 8 の倍数である必要があるため、
    #       形状は B_eff=B*H*W と L=T の積が 8 の倍数になるように設定する。
    dev, dt = _device_dtype()
    # CUDA の非メモリ効率パスでは T を 8 の倍数にする必要がある
    B, C, T, H, W = 1, 160, 8, 16, 16
    for flag in (False, True) if torch.cuda.is_available() else (False,):
        model = _make_model(C=C, d_model=C, ctx_dim=128, chunk=8, keep_spatial=True, mem_eff=flag).to(dev).to(dtype=dt)
        x = _randn((B, C, T, H, W), dev, dt)
        y = model(x)["sample"]
        assert y.shape == x.shape
        assert torch.isfinite(y).all()

def test_train_eval_mode_consistency():
    # train/eval の切替で API と有限性が変わらない
    dev, dt = _device_dtype()
    B, C, T, H, W = 1, 160, 2, 16, 16
    model = _make_model(C=C).to(dev).to(dtype=dt)
    x = _randn((B, C, T, H, W), dev, dt)
    model.train()
    y_tr = model(x)["sample"]
    model.eval()
    y_ev = model(x)["sample"]
    assert y_tr.shape == x.shape and y_ev.shape == x.shape
    assert torch.isfinite(y_tr).all() and torch.isfinite(y_ev).all()

def test_perf_microbench():
    # 軽いパフォーマンス計測（失敗条件は設けない／情報表示のみ）
    if not torch.cuda.is_available():
        return
    dev = torch.device("cuda")
    B, C, T, H, W = 1, 160, 8, 64, 64
    model = _make_model(C=C, chunk=8, keep_spatial=True, mem_eff=True).to(dev).to(dtype=torch.float16)
    x = torch.randn(B, C, T, H, W, device=dev, dtype=torch.float16)
    torch.cuda.synchronize()
    # warmup
    for _ in range(2):
        _ = model(x)["sample"].sum().item()
        torch.cuda.synchronize()
    iters = 5
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(x)["sample"].sum()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    avg = (t1 - t0) / iters
    print(f"[perf] avg_forward_time={avg*1000:.2f}ms over {iters} iters @ {B}x{C}x{T}x{H}x{W}")

if __name__ == "__main__":
    # 直接実行用
    if not torch.cuda.is_available():
        print("[SKIP] CUDA が無い環境では Mamba2 の Triton カーネルが使えないためテストをスキップします。")
        sys.exit(0)
    test_basic_forward()
    test_tuple_return()
    test_indicator_variants()
    test_indicator_extremes_and_paths()
    test_ctx_shapes()
    test_no_ctx_works()
    test_timestep_variants()
    test_invalid_indicator_raises()
    test_amp_autocast()
    test_bfloat16_autocast()
    test_long_T_with_chunk()
    test_mem_eff_toggle_runs()
    test_train_eval_mode_consistency()
    test_perf_microbench()
    print("[OK] all tests passed")
