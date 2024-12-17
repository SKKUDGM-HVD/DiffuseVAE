#python main/test.py reconstruct --device cuda:0 \
#                                 --dataset celeba \
#                                 --image-size 64 \
#                                 --save-path /root/workspace/skku/DGM/DiffuseVAE/result_vae_64_recon/ \
#                                 --write-mode numpy \
#                                 /root/workspace/skku/DGM/DiffuseVAE/vae_celeba64_alpha=1.0/checkpoints/test.ckpt \
#                                 /root/workspace/sharing/lgp-mlu-dev/share/celaba64/50k/




python main/test.py sample --device gpu:0 \
                             --image-size 64 \
                             --seed 0 \
                             --num-samples 32 \
                             --save-path /root/workspace/skku/DGM/DiffuseVAE/result_vae_64_sample/ \
                             --write-mode image \
                             512 \
                             /root/workspace/skku/DGM/DiffuseVAE/vae_celeba64_alpha=1.0/checkpoints/test.ckpt \
