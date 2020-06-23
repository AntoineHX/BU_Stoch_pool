#!/bin/bash
#SBATCH --gres=gpu:1 #gpu:v100l:1    # https://docs.computecanada.ca/wiki/Using_GPUs_with_Slurm
#SBATCH --cpus-per-task=6 #6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M  #32000M       # Memory proportional to CPUs: 32000 Cedar, 64000 Graham.
#SBATCH --account=def-mpederso
#SBATCH --time=1:00:00
#SBATCH --job-name=MyLeNetMatNormalNoceil
#SBATCH --output=log/%x-%j.out
#SBATCH --mail-user=harle.collette.antoine@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# Setup
source ~/stoch_pool/bin/activate

#Execute
# echo $(pwd) = /home/antoh/projects/def-mpederso/antoh/stoch/jobs
cd ../

time python main.py \
    -n MyLeNetMatNormalNoceil \
    -ep 10 \
    -sc cosine \
    -lr 5e-2 \
    -pf _noCrop