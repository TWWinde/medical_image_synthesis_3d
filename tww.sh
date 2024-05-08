#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --qos=batch
# SBATCH --gpus=rtx_a5000:1
# SBATCH --gpus=rtx_a6000:1
# SBATCH --gpus=geforce_rtx_2080ti:1
# SBATCH --gpus=geforce_gtx_titan_x:1

# Activate everything you need

#conda activate /anaconda3/envs/myenv
module load cuda
pyenv activate venv

# Run your python code
python /misc/no_backups/d1502/medical_image_synthesis_3d/dataloaders/preparation.py