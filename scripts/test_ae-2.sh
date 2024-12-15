# python main/test.py reconstruct --device cuda:0 \
#                                 --dataset celeba \
#                                 --image-size 64 \
#                                 --save-path ~/vae_celeba64_recons/ \
#                                 --write-mode numpy \
#                                 /root/workspace/skku/DGM/DiffuseVAE/result/vae_cmhq128_alpha=1.0/checkpoints/test.ckpt \
#                                 /root/workspace/sharing/lgp-mlu-dev/share/img-align-celeba/img-align-celeba/test

# python main/test.py reconstruct --device cuda:0 \
#                                 --dataset ffhq \
#                                 --image-size 128 \
#                                 --num-samples 64 \
#                                 --save-path /root/workspace/skku/DGM/DiffuseVAE/result/ \
#                                 --write-mode image \
#                                 /root/workspace/skku/DGM/DiffuseVAE/result/vae_cmhq128_alpha=1.0/checkpoints/test.ckpt \
#                                /root/workspace/sharing/lgp-mlu-dev/share/ffhq

#python main/test.py sample --device cuda:0 \
#                             --image-size 32 \
#                             --seed 0 \
#                             --num-samples 50000 \
#                             --save-path /root/workspace/skku/DGM/DiffuseVAE/result/vae_samples_cifar10_deleteme/ \
#                             --write-mode image \
#                             512 \
#                             /root/workspace/skku/DGM/DiffuseVAE/result/vae_cmhq128_alpha=1.0/checkpoints/test.ckpt \

 python main/test.py sample --device gpu:0 \
                             --image-size 128 \
                             --seed 0 \
                             --num-samples 64 \
                             --save-path /root/workspace/skku/DGM/DiffuseVAE/result2/ \
                             --write-mode image \
                             1024 \
                             /root/workspace/skku/DGM/DiffuseVAE/result/vae_cmhq128_alpha=1.0/checkpoints/test.ckpt \


# python main/test.py reconstruct --device gpu:0 \
#                                 --dataset afhq \
#                                 --image-size 128 \
#                                 --save-path ~/reconstructions/afhq_reconsv2/ \
#                                 --write-mode numpy \
#                                 ~/vae_afhq_alpha\=1.0/checkpoints/vae-afhq_alpha=1.0-epoch=1499-train_loss=0.0000.ckpt \
#                                 ~/datasets/afhq/

# python main/test.py sample --device gpu:0 \
#                             --image-size 128 \
#                             --seed 0 \
#                             --num-samples 64 \
#                             --save-path ~/afhq_vae_samples1/ \
#                             --write-mode image \
#                             1024 \
#                             ~/vae_afhq_alpha\=1.0/checkpoints/vae-afhq_alpha=1.0-epoch=1499-train_loss=0.0000.ckpt \

# python main/test.py reconstruct --device gpu:0 \
#                            --dataset celebamaskhq \
#                            --num-samples 16 \
#                            --save-path ~/vae_alpha_1_0_samples/ \
#                            ~/checkpoints/cmhq/vae-epoch\=189-train_loss\=0.00.ckpt \
#                            ~/datasets/CelebAMask-HQ/
