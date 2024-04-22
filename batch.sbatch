#!/bin/bash

####

# Define slurm job parameters

####

#SBATCH --job-name=VQ-GAN
#SBATCH --cpus-per-task=2
#SBATCH --partition=student
#SBATCH --gpus=1
#SBATCH --time 31-0
#SBATCH --nodelist=node-gpu-01
#SBATCH --error=job.%J.%x.err
#SBATCH --output=job.%J.%x.out

module load cuda/11.3

#experiment_1 train VQ-GAN

PL_TORCH_DISTRIBUTED_BACKEND=gloo python /train/train_vqgan.py dataset=synthrad2023 \
model=vq_gan_3d model.gpus=1 \
model.default_root_dir_postfix='checkpoint' model.precision=16 model.embedding_dim=8 \
model.n_hiddens=16 model.downsample=[2,2,2] model.num_workers=32 model.gradient_clip_val=1.0 \
model.lr=3e-4 model.discriminator_iter_start=10000 model.perceptual_weight=4 \
model.image_gan_weight=1 model.video_gan_weight=1 model.gan_feat_weight=4 \
model.batch_size=1 model.n_codes=16384 model.accumulate_grad_batches=1