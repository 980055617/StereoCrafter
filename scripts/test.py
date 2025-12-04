if __name__ == "__main__":
    from mamba_ssm import Mamba2
    import torch
    mamba = Mamba2(
        d_model=320,
        d_state=256,
        d_conv=4,
        expand=4,
        chunk_size=32,
    ).cuda()
    
    # Execute small tensor first
    x = torch.randn(30720, 8, 320).cuda()
    x = mamba(x)
    print(x.shape)  # (30720, 8, 320) - Works


# =============================================
# 使い方メモ（疑似コード）
# ---------------------------------------------
# from blocks.mamba_spatiotemporal import MambaSpatioTemporalModel
# m = MambaSpatioTemporalModel(
#     in_channels=160,
#     d_model=160,
#     cross_attention_dim=1280,
#     d_state=256,
#     headdim=64,
#     expand=2,
#     temporal_chunk_size=64,
#     use_mem_eff_path=True,
#     num_groups_gn=32,
#     keep_spatial_mixer=True,
# )
# x = torch.randn(2, 160, 3, 64, 64)
# ctx = torch.randn(2, 77, 1280)
# t = torch.tensor([10, 10])
# out = m(x, encoder_hidden_states=ctx, timestep=t, image_only_indicator=None, return_dict=True)
# print(out["sample"].shape)  # (2,160,3,64,64)