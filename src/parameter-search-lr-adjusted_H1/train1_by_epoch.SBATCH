#!/bin/bash
#SBATCH --job-name=train1_AL_ld_by_epoch
#SBATCH --open-mode=append
#SBATCH --output=%j_%x.out
#SBATCH --error=%j_%x.err
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --gres=gpu:v100:1  ## To request specific GPU (v100 or rtx8000)
#SBATCH --time=04:00:00
#SBATCH --mem=20G
#SBATCH -c 4


module purge
module load python/intel/3.8.6


singularity exec --nv --bind /scratch/yz5880 --overlay /scratch/yz5880/overlay-25GB-500K.ext3:ro /scratch/yz5880/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
python3 /scratch/yz5880/ALP/src/parameter-search-lr-adjusted_H1/train.py --data_dir UCLA-protest/ --batch_size 10 --lr 0.0002 --print_freq 100 --epochs 100 --cuda --method_id 0 --heuristic_id 5 --workers 8 --num_label_samples 400 --num_samples_added 1 --loss_decrease_model model_heuristic.pt --lr_mode 1
"
