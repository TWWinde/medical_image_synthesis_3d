#!/bin/bash -l

#Slurm parameters
#SBATCH --job-name=3DVQGAN
#SBATCH --output=VQ-GAN%j.%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=6-23:00:00
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --gpus=3
#SBATCH --qos=batch
# SBATCH --gpus=rtx_a5000:1
# SBATCH --gpus=geforce_rtx_2080ti:1
# SBATCH --gpus=geforce_gtx_titan_x:1

# Activate everything you need

#conda activate /anaconda3/envs/myenv
pyenv activate venv
module load cuda
# Run your python code


#experiment_1 train VQ-GAN
python /misc/no_backups/d1502/medical_image_synthesis_3d/train/train_vqgan.py --dataset_name SynthRAD2023 \
--name vq_gan_3d --gpu_ids 0,1,2 --embedding_dim 8 \
--lr 0.0003 --discriminator_iter_start 10000 --perceptual_weight 4 \
--image_gan_weight 1 --video_gan_weight 1 --gan_feat_weight 4 \
--batch_size 3 --n_codes 16384

