#!/bin/bash
#SBATCH --gres=gpu:1 #v100l:1    # https://docs.computecanada.ca/wiki/Using_GPUs_with_Slurm
#SBATCH --cpus-per-task=6 #6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M  #32000M       # Memory proportional to CPUs: 32000 Cedar, 64000 Graham.
#SBATCH --account=def-mpederso
#SBATCH --time=10:00:00
#SBATCH --job-name=Benchmark-MyLeNetMatStochBUNoceil
#SBATCH --output=log/%x-%j.out
#SBATCH --mail-user=harle.collette.antoine@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=1-3



# Setup
source ~/virtual_env/stoch_pool/bin/activate

#Execute
cd ../

time python main.py \
    -n MyLeNetMatStochBUNoceil \
    -ep 50 \
    -sc cosine \
    -lr 5e-2 \
    -rf 'res/benchmark_NoCeil/' \
    -k 4 \
    -pf __k4_$SLURM_ARRAY_TASK_ID