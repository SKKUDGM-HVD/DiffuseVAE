python main/train_e2e_stage3.py +dataset=celeba64/train \
                      dataset.ddpm.data.root='/root/workspace/sharing/lgp-mlu-dev/share/celaba64/50k' \
                      dataset.ddpm.data.name='celeba' \
                      dataset.ddpm.data.norm=True \
                      dataset.ddpm.data.hflip=True \
                      dataset.ddpm.model.dim=128 \
                      dataset.ddpm.model.dropout=0.1 \
                      dataset.ddpm.model.attn_resolutions=\'16,\' \
                      dataset.ddpm.model.n_residual=2 \
                      dataset.ddpm.model.dim_mults=\'1,2,2,2,4\' \
                      dataset.ddpm.model.n_heads=8 \
                      dataset.ddpm.training.type='form1' \
                      dataset.ddpm.training.cfd_rate=0.0 \
                      dataset.ddpm.training.epochs=500 \
                      dataset.ddpm.training.z_cond=True \
                      dataset.ddpm.training.log_step=50 \
                      dataset.ddpm.training.batch_size=32 \
                      dataset.ddpm.training.vae_chkpt_path=\'/root/workspace/skku/DGM/DiffuseVAE/vae_celeba64_alpha=1.0/checkpoints/test.ckpt\' \
                      dataset.ddpm.training.device=\'gpu:0\' \
                      dataset.ddpm.training.results_dir=\'/root/workspace/skku/DGM/DiffuseVAE/diffusevae_celeba64_e2e_st3_nheads=8_dropout=0.1/\' \
                      dataset.ddpm.training.workers=1 \
                      dataset.ddpm.training.chkpt_prefix=\'celeba64_e2d_nheads=8_dropout=0.1\'

					  