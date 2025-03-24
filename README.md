# Diffusion-and-Guidance

You need to edit just ddpm.py file without changing the function names and signatues of functions outside `__main__`. You can add more options to the argparser as required for your experiments. For example, class labels and guidance scales for part 2.

To setup the environment, follow these steps:

```
conda create --name cs726env python=3.8 -y
conda activate cs726env
```
Install the dependencies
```
pip install -r requirements.txt
```
To install torch, you can follow the steps [here](https://pytorch.org/get-started/locally/). You'll need to know the cuda version on the server. Use `nvitop` command to know the version first. If you have cuda version 12.4, you can just do:

```
pip install torch
```

In case multiple GPUs are present in the system, we recommend using the environment variable `CUDA_VISIBLE_DEVICES` when running your scripts. For example, below command ensures that your script runs on 7th GPU. 

```
CUDA_VISIBLE_DEVICES=7 python ddpm.py --mode train --dataset moons
```

CUDA error messages can often be cryptic and difficult to debug. In such cases, the following command can be quite useful:
```
CUDA_VISIBLE_DEVICE=-1 python ddpm.py --mode train --dataset moons
```
This forces the script to run exclusively on the CPU.



Run the DDPM file as 

1. For training
python ddpm.py \
  --mode train \
  --dataset moons \               # Dataset: moons/circles/8gaussians
  --n_steps 1000 \                # Diffusion steps (match for sampling)
  --lr 0.001 \                    # Learning rate
  --batch_size 256 \              # Batch size
  --epochs 500 \                  # Training epochs
  --n_dim 2 \                     # Data dimension
  --type linear \                 # Noise schedule: linear/cosine
  --lbeta 0.0001 \                # Start beta (linear only)
  --ubeta 0.02 \                  # End beta (linear only)
  --guidance_scale 7.0 \          # For conditional models (0=unconditional)
  --label_dropout 0.1 \           # Classifier-free guidance dropout
  --run_name exps5/ddpm_trained \ # Output directory
  --visualize                     # Enable visualizations

2. Sampling
# Unconditional Sampling (DDPM)
python ddpm.py \
  --mode sample \
  --n_samples 10000 \             # Number of samples
  --run_name exps5/ddpm_trained \ # Must match training dir
  --seed 42                       # Reproducibility

# Conditional Sampling with Guidance
python ddpm.py \
  --mode sample \
  --n_samples 5000 \
  --guidance_scale 7.0 \          # CFG scale (>0)
  --class_label 0 \               # Target class (for conditional)
  --ddim_steps 50 \               # For DDIM acceleration
  --eta 0.0 \                     # DDIM eta (0=deterministic)
  --run_name exps5/ddpm_trained


3. Visualization
# Generate Reverse Process GIFs
python ddpm.py \
  --mode visualize \
  --n_samples 1000 \
  --ddim_steps_list 50 40 30 20 10 1 \ # Multiple step counts
  --eta 0.0 \
  --run_name exps5/ddpm_trained \
  --visualize

4 . Comparision between DDPM and DDIM
python ddpm.py \
  --mode compare \
  --n_samples 5000 \
  --ddim_steps 50 \               # Steps for accelerated sampling
  --eta 0.0 \                     # Deterministic DDIM
  --run_name exps5/ddpm_trained

5. Classification
python ddpm.py \
  --mode classify \
  --classifier_type cfg \         # cfg/mlp
  --n_samples 10000 \             # Evaluation samples
  --num_steps 50 \                # Classifier steps
  --run_name exps5/ddpm_trained





  

