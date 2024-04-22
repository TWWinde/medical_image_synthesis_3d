#!/bin/bash -l

#Slurm parameters
#SBATCH --job-name=VQ-GAN
#SBATCH --output=VQ-GAN%j.%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=6-23:00:00
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --qos=batch
# SBATCH --nodes=1
#SBATCH --gpus=rtx_a5000:1
# SBATCH --gpus=geforce_rtx_2080ti:1
# SBATCH --gpus=geforce_gtx_titan_x:1

# Activate everything you need

#conda activate /anaconda3/envs/myenv
pyenv activate venv
module load cuda
# Run your python code

#experiment_1 train VQ-GAN
PL_TORCH_DISTRIBUTED_BACKEND=gloo python /misc/no_backups/s1449/medical_image_synthesis_3d/train/train_vqgan.py dataset=synthrad2023 \
model=vq_gan_3d model.gpus=1 \
model.default_root_dir_postfix='flair' model.precision=8 model.embedding_dim=8 \
model.n_hiddens=16 model.downsample=[2,2,2] model.num_workers=32 model.gradient_clip_val=1.0 \
model.lr=3e-4 model.discriminator_iter_start=10000 model.perceptual_weight=4 \
model.image_gan_weight=1 model.video_gan_weight=1 model.gan_feat_weight=4 \
model.batch_size=1 model.n_codes=16384 model.accumulate_grad_batches=1

