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
python /misc/no_backups/s1449/medical_image_synthesis_3d/train/train_vqgan.py dataset_name synthrad2023 \
--name vq_gan_3d --gpu_ids 0,1 \
 --precision 16 --embedding_dim 8 \
--n_hiddens 16 --downsample [2,2,2] --num_workers 32 --gradient_clip_val 1.0 \
--lr 0.0003 --discriminator_iter_start 10000 --perceptual_weight 4 \
--image_gan_weight 1 --video_gan_weight 1 --gan_feat_weight 4 \
--batch_size 1 --n_codes 16384 --accumulate_grad_batches 1

