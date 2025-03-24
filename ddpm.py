import torch
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch import nn
import argparse
import torch.nn.functional as F
import utils
import dataset
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import time
from scipy.stats import wasserstein_distance 
import math
import imageio.v2 as imageio
import glob

class NoiseScheduler():
    """
    Noise scheduler for the DDPM model

    Args:
        num_timesteps: int, the number of timesteps
        type: str, the type of scheduler to use
        **kwargs: additional arguments for the scheduler

    This object sets up all the constants like alpha, beta, sigma, etc. required for the DDPM model
    
    """
    def __init__(self, num_timesteps=50, type="linear", **kwargs):
        self.num_timesteps = num_timesteps
        self.type = type

        if type == "linear":
            
            self.init_linear_schedule(kwargs['beta_start'], kwargs['beta_end'])
        elif type == "cosine":
            s = kwargs.get('s', 0.008)
            self.init_cosine_schedule(s)
        else:
            raise NotImplementedError(f"{type} scheduler is not implemented")


    def init_linear_schedule(self, beta_start, beta_end):
        """
        Precompute quantities required for training and sampling
        """
        print("for Linear")
        self.betas = torch.linspace(beta_start, beta_end, self.num_timesteps, dtype=torch.float32).clamp(1e-6, 0.02)   
        
        print(self.betas)
        self.alphas = 1 - self.betas
        # print(self.alphas)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        print(self.alphas_cumprod)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        print(self.alphas_cumprod_prev)
    
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        # print(self.sqrt_alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        # print(self.sqrt_one_minus_alphas_cumprod)
        
        
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod + 1e-6) 
        # print(self.posterior_variance)

    def init_cosine_schedule(self, s=0.008):
        print("for Cosine ")
        steps = self.num_timesteps + 1
        print(steps)
        x = torch.linspace(0, self.num_timesteps, steps)
        print(x)
        alphas_cumprod = torch.cos(((x / self.num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        print(alphas_cumprod)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        print(alphas_cumprod)
        self.betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        print(self.betas)
        self.betas = torch.clamp(self.betas, 0.0001, 0.02)
        print(self.betas)
    
        self.alphas = 1 - self.betas
        print(self.alphas)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        print(self.alphas_cumprod)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        print(self.alphas_cumprod_prev)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        print(self.sqrt_alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        print(self.sqrt_one_minus_alphas_cumprod)
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod + 1e-6)
        print(self.posterior_variance)
        
        
    def __len__(self):
        return self.num_timesteps
    
# class DDPM(nn.Module):
#     def __init__(self, n_dim=3, n_steps=200, time_embed_dim=128, hidden_dim=256,  noise_scheduler=None):
#         """
#         Noise prediction network for the DDPM

#         Args:
#             n_dim: int, the dimensionality of the data
#             n_steps: int, the number of steps in the diffusion process
#             time_embed_dim: int, the dimension of the time embedding
#             hidden_dim: int, the dimension of the hidden layers
#         """
#         # sup# The above code appears to be a snippet of Python code defining a class or a function.
#         # However, most of the code is commented out using the '#' symbol, so it is not currently
#         # executing.
#         # super().__init__()
#         # self.noise_scheduler = noise_scheduler
    
#         # self.time_embed = nn.Sequential(
#         #     nn.Linear(1, time_embed_dim),
#         #     nn.ReLU(),                                      
#         #     nn.Linear(time_embed_dim, time_embed_dim),
#         # )
        
    
#         # self.model = nn.Sequential(
#         #     nn.Linear(n_dim + time_embed_dim, hidden_dim),
#         #     nn.LayerNorm(hidden_dim),
#         #     nn.ReLU(),
#         #     nn.Linear(hidden_dim, hidden_dim),
#         #     nn.LayerNorm(hidden_dim),
#         #     nn.ReLU(),
#         #     nn.Linear(hidden_dim, hidden_dim),
#         #     nn.LayerNorm(hidden_dim),
#         #     nn.ReLU(),
#         #     nn.Linear(hidden_dim, n_dim)
#         # )
#         super().__init__()
#         self.noise_scheduler = noise_scheduler
        
#         # Add this block (critical fix)
#         self.time_embed = nn.Sequential(
#             nn.Linear(1, time_embed_dim),
#             nn.ReLU(),
#             nn.Linear(time_embed_dim, time_embed_dim)
#         )
        
#         self.model = nn.Sequential(
#             nn.Linear(n_dim + time_embed_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(),
#             self._make_residual_block(hidden_dim),
#             self._make_residual_block(hidden_dim),
#             nn.Linear(hidden_dim, n_dim)
#         )

#     # Helper function for residual blocks
#     def _make_residual_block(self, dim):
#         return nn.Sequential(
#             nn.Linear(dim, dim),
#             nn.LayerNorm(dim),
#             nn.ReLU()
#         )    
        
#     def forward(self, x, t):
#         """
#         Args:
#             x: torch.Tensor, the input data tensor [batch_size, n_dim]
#             t: torch.Tensor, the timestep tensor [batch_size]

#         Returns:
#             torch.Tensor, the predicted noise tensor [batch_size, n_dim]
#         """
        
#         t = t.float().view(-1, 1) / self.noise_scheduler.num_timesteps    
    
#         t_emb = self.time_embed(t)
        
        
#         x_input = torch.cat([x, t_emb], dim=1)

#         noise_pred = self.model(x_input)
        
#         return noise_pred


# class DDPM(nn.Module):
#     def __init__(self, n_dim=3, n_steps=200, time_embed_dim=128, hidden_dim=256, noise_scheduler=None):
#         super().__init__()
#         self.noise_scheduler = noise_scheduler
        
#         self.time_embed = nn.Sequential(
#             nn.Linear(1, time_embed_dim),
#             nn.ReLU(),
#             nn.Linear(time_embed_dim, time_embed_dim)
#         )
        
#         # Use explicit sequential layers (matches original training)
#         self.model = nn.Sequential(
#             nn.Linear(n_dim + time_embed_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, n_dim)
#         )

#     def forward(self, x, t):
#         # t = t.float().view(-1, 1) / self.noise_scheduler.num_timesteps    
#         # t_emb = self.time_embed(t)
#         half_dim = t // 2
#         emb = math.log(10000) / (half_dim - 1)
#         emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
#         emb = t.unsqueeze(-1) * emb.unsqueeze(0)
#         t_emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
#         x_input = torch.cat([x, t_emb], dim=1)
#         return self.model(x_input)
        
    
class DDPM(nn.Module):
    def __init__(self, n_dim=3, n_steps=200, time_embed_dim=128, hidden_dim=256, noise_scheduler=None):
        super().__init__()
        self.noise_scheduler = noise_scheduler
        self.time_embed_dim = time_embed_dim  
        
        self.model = nn.Sequential(
            nn.Linear(n_dim + time_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_dim)
        )

    def forward(self, x, t):
        half_dim = self.time_embed_dim // 2  
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        t_emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        x_input = torch.cat([x, t_emb], dim=1)
        return self.model(x_input)
        
class ConditionalDDPM(nn.Module):
    def __init__(self, n_dim=3, n_classes=2, n_steps=200, 
                time_embed_dim=128, label_embed_dim=64, hidden_dim=256,noise_scheduler=None):
        super().__init__()
        self.n_classes = n_classes
        self.noise_scheduler = noise_scheduler
        
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        self.label_embed = nn.Embedding(n_classes, label_embed_dim)
        
    
        self.model = nn.Sequential(
            nn.Linear(n_dim + time_embed_dim + label_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_dim)
        )
        
    def forward(self, x, t, y=None):
    
        t = t.float().view(-1, 1) / noise_scheduler.num_timesteps
        
    
        t_emb = self.time_embed(t)
        
    
        if y is None:
            y_emb = torch.zeros(x.shape[0], self.label_embed.embedding_dim).to(x.device)
        else:
            assert (y >= 0).all() and (y < self.n_classes).all(), "Invalid class indices"
            y_emb = self.label_embed(y)
        
        
        x_input = torch.cat([x, t_emb, y_emb], dim=1)
        return self.model(x_input)
    
def trainConditional(model, noise_scheduler, dataloader, optimizer, epochs, 
                    run_name, label_dropout=0.1):
    device = next(model.parameters()).device
    losses = []
    
    for epoch in range(epochs):
        epoch_losses = []
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            x, y = batch
            x, y = x.to(device), y.to(device)
            batch_size = x.shape[0]
            
        
            mask = torch.rand(batch_size) < label_dropout
            y[mask] = -1
            cond_mask = ~mask 
            
            
            x_cond, y_cond, x_uncond = x[cond_mask], y[cond_mask], x[mask]
            
            t = torch.randint(0, len(noise_scheduler), (batch_size,), device=device)
            t_cond, t_uncond = t[cond_mask], t[mask]
            noise = torch.randn_like(x)
            
            if len(x_cond) > 0:
                alpha_cumprod = noise_scheduler.alphas_cumprod.to(device)
                alpha_t_cond = alpha_cumprod[t_cond].view(-1, 1)
                noisy_x_cond = torch.sqrt(alpha_t_cond) * x_cond + torch.sqrt(1 - alpha_t_cond) * noise[cond_mask]
                noise_pred_cond = model(noisy_x_cond, t_cond, y_cond)
                loss_cond = F.mse_loss(noise_pred_cond, noise[cond_mask])
            else:
                loss_cond = 0
            
            if len(x_uncond) > 0:
                alpha_t_uncond = alpha_cumprod[t_uncond].view(-1, 1)
                noisy_x_uncond = torch.sqrt(alpha_t_uncond) * x_uncond + torch.sqrt(1 - alpha_t_uncond) * noise[mask]
                noise_pred_uncond = model(noisy_x_uncond, t_uncond, y=None)
                loss_uncond = F.mse_loss(noise_pred_uncond, noise[mask])
            else:
                loss_uncond = 0
            
            loss = (loss_cond + loss_uncond) / (len(x_cond) + len(x_uncond) + 1e-8)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            progress_bar.set_postfix(loss=loss.item())
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            save_path = f"{run_name}/model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), save_path)
        final_path = f"{run_name}/model.pth"
        torch.save(model.state_dict(), final_path)
    torch.save(model.state_dict(), f"{run_name}/model.pth")   
        
class ClassifierDDPM():
    def __init__(self, model: ConditionalDDPM, noise_scheduler: NoiseScheduler, num_steps=50, n_classes=2):
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.n_classes = n_classes
        self.num_steps = num_steps

    def predict_proba(self, x, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            
        device = next(self.model.parameters()).device
        x = x.to(device)
        batch_size = x.shape[0]
        losses = torch.zeros(batch_size, self.n_classes).to(device)
        
        with torch.no_grad():
            for c in range(self.n_classes):
                for step in range(self.num_steps):
                    t = torch.randint(0, len(self.noise_scheduler), (batch_size,), device=device)
                    
                    noise = torch.randn_like(x)
                    
                    alpha_cumprod = self.noise_scheduler.alphas_cumprod.to(device)
                    alpha_t = alpha_cumprod[t].view(-1, 1)
                    noisy_x = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise
                    
                    y = torch.full((batch_size,), c, device=device, dtype=torch.long)
                    
                    noise_pred = self.model(noisy_x, t, y)
                    
                    mse = F.mse_loss(noise_pred, noise, reduction='none').mean(dim=1)
                    losses[:, c] += mse
        
        avg_losses = losses / self.num_steps
        probs = F.softmax(-avg_losses, dim=1)
        return probs.cpu().numpy()
    
    def predict(self, x, seed=None):
        """Returns class predictions instead of probabilities"""
        probs = self.predict_proba(x, seed=seed)
        return np.argmax(probs, axis=1)

# ------------------------------------------------------------------------------------------------------------------------------

def evaluate_guidance_scales(model, noise_scheduler, n_samples=1000, class_label=0, scales=[0, 0.5, 1.0, 3.0, 5.0, 7.0], num_runs=3):
    """
    Generate samples with different guidance scales and evaluate their class consistency
    
    Args:
        model: ConditionalDDPM model
        noise_scheduler: NoiseScheduler
        n_samples: Number of samples to generate for each scale
        class_label: Target class label
        scales: List of guidance scales to evaluate
        num_runs: Number of times to repeat each evaluation for statistical significance
    
    Returns:
        dict: Results containing accuracy for each scale
    """
    results = {scale: [] for scale in scales}
    classifier = ClassifierDDPM(model, noise_scheduler, num_steps=50)
    
    for run in range(num_runs):
        run_seed = 42 + run * 100
        
        print(f"\nRun {run+1}/{num_runs}:")
        for scale in scales:
            print(f"Generating samples with guidance_scale={scale}...")
            samples, _ = sampleCFG(
                model, n_samples, noise_scheduler, 
                guidance_scale=scale, 
                class_label=class_label,
                seed=run_seed
            )
            
            eval_seed = run_seed + 1
            preds = classifier.predict(samples, seed=eval_seed)
            accuracy = (preds == class_label).mean()
            
            results[scale].append(accuracy)
            print(f"Guidance Scale {scale}: Accuracy = {accuracy*100:.1f}%")
    
    summary = {}
    for scale in scales:
        mean_acc = np.mean(results[scale])
        std_acc = np.std(results[scale])
        summary[scale] = (mean_acc, std_acc)
        print(f"\nGuidance Scale {scale}: Mean Accuracy = {mean_acc*100:.1f}% ± {std_acc*100:.1f}%")
    
    return results, summary

# def visualize_reverse_process(model, noise_scheduler, n_samples=1000, run_name="debug"):
#     # Generate samples with intermediate steps
#     intermediates = sample(model, n_samples, noise_scheduler, return_intermediate=True)
    
#     # Create visualization grid
#     n_steps = len(intermediates)
#     cols = 5
#     rows = int(np.ceil(n_steps/cols))
    
#     plt.figure(figsize=(20, 4*rows))
#     plt.suptitle(f"Reverse Diffusion Process ({noise_scheduler.num_timesteps} steps)", y=1.02)
    
#     for i, x in enumerate(intermediates):
#         plt.subplot(rows, cols, i+1)
#         plt.scatter(x[:, 0].cpu().numpy(), x[:, 1].cpu().numpy(), s=1, alpha=0.5)
#         plt.title(f"Step {noise_scheduler.num_timesteps - i*100}")
#         plt.grid(True, alpha=0.3)
#         plt.xlim(-3, 3)
#         plt.ylim(-3, 3)
    
#     plt.tight_layout()
#     plt.savefig(f"{run_name}/reverse_process.png", bbox_inches='tight')
#     plt.close()
#     print(f"Saved reverse process visualization to {run_name}/reverse_process.png")
    
    
@torch.no_grad()
def visualize_reverse_ddpm(model, noise_scheduler, n_samples=1000, save_dir="reverse_process"):
    """Sampling with visualization of intermediate steps"""
    device = next(model.parameters()).device
    os.makedirs(save_dir, exist_ok=True)
    
    # Initial noise
    x = torch.randn(n_samples, model.model[-1].out_features).to(device)
    steps = list(range(noise_scheduler.num_timesteps-1, -1, -1))
    
    plt.figure(figsize=(12, 12))
    
    for idx, t in enumerate(tqdm(steps, desc="Reverse Process")):
        # Run one denoising step
        timesteps = torch.full((n_samples,), t, device=device, dtype=torch.long)
        noise_pred = model(x, timesteps)
        
        # DDPM reverse step calculation
        alpha = noise_scheduler.alphas[t]
        alpha_cumprod = noise_scheduler.alphas_cumprod[t]
        beta = noise_scheduler.betas[t]
        
        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
            
        x = (1/torch.sqrt(alpha)) * (x - ((1 - alpha)/torch.sqrt(1 - alpha_cumprod)) * noise_pred) + torch.sqrt(beta)*noise

        # Visualize every 100 steps (for 1000-step process)
        if idx % 100 == 0 or t == 0:
            plt.clf()
            plt.scatter(x[:, 0].cpu(), x[:, 1].cpu(), s=1, alpha=0.6,
                    c=np.linspace(0,1,len(x)), cmap='viridis')
            plt.title(f"Reverse Step: {t} ({(noise_scheduler.num_timesteps-t)/noise_scheduler.num_timesteps*100:.1f}% denoised)")
            plt.xlim(-3, 3)
            plt.ylim(-3, 3)
            plt.grid(True, alpha=0.3)
            plt.savefig(f"{save_dir}/step_{t:04d}.png", dpi=120, bbox_inches='tight')
    
    # Create animation
    files = sorted(glob.glob(f"{save_dir}/step_*.png"), 
                key=lambda x: int(x.split('_')[-1].split('.')[0]))
    images = []
    for f in files[::-1]:
        images.append(imageio.imread(f))
    
    imageio.mimsave(f"{save_dir}/reverse_process.gif", images, fps=2)

@torch.no_grad()
def visualize_reverse_ddim(model, noise_scheduler, n_samples=1000, base_save_dir="reverse_ddim", steps_list=[50, 40, 30, 20, 10, 1], eta=0.0):
    device = next(model.parameters()).device
    os.makedirs(base_save_dir, exist_ok=True)
    T = noise_scheduler.num_timesteps

    for steps in steps_list:
        save_dir = os.path.join(base_save_dir, f"steps_{steps}")
        os.makedirs(save_dir, exist_ok=True)

        # Generate timestep sequence for current step count
        step_inc = max(1, T // steps)
        times = torch.arange(T-1, -1, -step_inc).tolist()
        if 0 not in times:
            times.append(0)
        times = sorted(times, reverse=True)  # From T-1 to 0

        x = torch.randn(n_samples, model.model[-1].out_features).to(device)
        plt.figure(figsize=(12, 12))

        for i, t in enumerate(tqdm(times, desc=f"DDIM Sampling (Steps={steps})")):
            # Visualization
            plt.clf()
            plt.scatter(x[:, 0].cpu(), x[:, 1].cpu(), s=1, alpha=0.6, cmap='viridis')
            plt.title(f"Steps: {steps} | Progress: {i+1}/{len(times)} | t={t}")
            plt.xlim(-3, 3)
            plt.ylim(-3, 3)
            plt.grid(True, alpha=0.3)
            plt.savefig(f"{save_dir}/step_{i:04d}.png", dpi=120, bbox_inches='tight')

            # DDIM Step Calculation
            if i < len(times)-1:
                t_prev = times[i+1]
                timestep = torch.full((n_samples,), t, device=device, dtype=torch.long)
                noise_pred = model(x, timestep)

                alpha = noise_scheduler.alphas_cumprod[t]
                alpha_prev = noise_scheduler.alphas_cumprod_prev[t_prev] if t_prev >=0 else 1.0

                pred_x0 = (x - torch.sqrt(1-alpha)*noise_pred)/torch.sqrt(alpha)
                dir_xt = torch.sqrt(torch.clamp(1-alpha_prev - (eta**2)*(1-alpha), min=1e-6)) * noise_pred
                noise = eta * torch.sqrt(1-alpha) * torch.randn_like(x)
                x = torch.sqrt(alpha_prev)*pred_x0 + dir_xt + noise

        # Create GIF
        images = []
        files = sorted(glob.glob(f"{save_dir}/step_*.png"), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        for f in files:
            images.append(imageio.imread(f))
        imageio.mimsave(f"{save_dir}/reverse_process.gif", images, fps=4)
        print(f"Saved {steps}-step GIF to {save_dir}/reverse_process.gif")
    
def visualize_guidance_effect(model, noise_scheduler, n_samples=500, class_label=0, scales=[0, 3.0, 7.0]):
    """
    Generate and visualize samples with different guidance scales
    
    Args:
        model: ConditionalDDPM model
        noise_scheduler: NoiseScheduler
        n_samples: Number of samples to generate for each scale
        class_label: Target class label
        scales: List of guidance scales to evaluate
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    
    fig, axes = plt.subplots(1, len(scales), figsize=(5*len(scales), 5))
    if len(scales) == 1:
        axes = [axes]
    
    for i, scale in enumerate(scales):
        print(f"Generating samples with guidance_scale={scale}...")
        samples, _ = sampleCFG(
            model, n_samples, noise_scheduler, 
            guidance_scale=scale, 
            class_label=class_label,
            seed=42  
        )
        
        x = samples.cpu().numpy()
        
        if class_label == 0:
            cmap = LinearSegmentedColormap.from_list('custom_cmap', ['lightblue', 'darkblue'])
        else:
            cmap = LinearSegmentedColormap.from_list('custom_cmap', ['lightgreen', 'darkgreen'])
        
        axes[i].scatter(x[:, 0], x[:, 1], c=np.ones(n_samples), 
                        cmap=cmap, alpha=0.6, s=10)
        axes[i].set_title(f"Guidance Scale: {scale}")
        axes[i].set_xlim(-3, 3)
        axes[i].set_ylim(-3, 3)
        axes[i].set_aspect('equal')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"guidance_scale_comparison_class{class_label}.png", dpi=300)
    plt.close()
    
    print(f"Visualization saved to guidance_scale_comparison_class{class_label}.png")

# ------------------------------------------------------------------------------------------------------------------------
    
    
def train(model, noise_scheduler, dataloader, optimizer, epochs, run_name):
    """
    Train the model and save the model and necessary plots

    Args:
        model: DDPM, model to train
        noise_scheduler: NoiseScheduler, scheduler for the noise
        dataloader: torch.utils.data.DataLoader, dataloader for the dataset
        optimizer: torch.optim.Optimizer, optimizer to use
        epochs: int, number of epochs to train the model
        run_name: str, path to save the model
    """
    device = next(model.parameters()).device
    losses = []
    
    for epoch in range(epochs):
        epoch_losses = []
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx , batch in enumerate(progress_bar):
            x, _ = batch  
            x = x.to(device)
            batch_size = x.shape[0]
            
            t = torch.randint(0, len(noise_scheduler), (batch_size,), device=device)
            
            noise = torch.randn_like(x)
            
        
            alpha_cumprod = noise_scheduler.alphas_cumprod.to(device)     # change2
            alpha_cumprod = alpha_cumprod.to(device)
            alpha_t = alpha_cumprod[t].view(-1, 1)
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
            
            noisy_x = sqrt_alpha_t * x + sqrt_one_minus_alpha_t * noise
            
            noise_pred = model(noisy_x, t)
            
            loss = F.mse_loss(noise_pred, noise)
            
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            progress_bar.set_postfix(loss=loss.item())

            if batch_idx % 100 == 0:
                print(f"Batch {batch} | Noise Prediction Error: {loss:.4f}")
        # Calculate average loss for the epoch
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            torch.save(model.state_dict(), f"{run_name}/model_epoch_{epoch+1}.pth")
    
    torch.save(model.state_dict(), f"{run_name}/model.pth")
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(f"{run_name}/loss_curve.png")
    plt.close()
    
    print(f"Training completed. Model saved to {run_name}")

@torch.no_grad()
def sample(model, n_samples, noise_scheduler, initial_noise=None, return_intermediate=False):
    device = next(model.parameters()).device
    n_dim = model.model[-1].out_features 
    
    if initial_noise is not None:
        x = initial_noise.to(device)
        n_samples = initial_noise.shape[0]  
    else:
        x = torch.randn(n_samples, n_dim).to(device)
    
    intermediates = [x.cpu()] if return_intermediate else None
    save_interval = max(1, noise_scheduler.num_timesteps // 10)
    
    for t in tqdm(range(noise_scheduler.num_timesteps - 1, -1, -1), desc="Sampling"):
        timesteps = torch.full((n_samples,), t, device=device, dtype=torch.long)
        noise_pred = model(x, timesteps)
        alpha = noise_scheduler.alphas[t]
        beta = noise_scheduler.betas[t]
        alphas_cumprod = noise_scheduler.alphas_cumprod[t]
        
        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
            
        x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / 
             torch.sqrt(1 - noise_scheduler.alphas_cumprod[t])) * noise_pred) + \
             torch.sqrt(beta) * noise
        
        # if return_intermediate and t%100 == 0:
        #     intermediates.append(x.cpu())

        if return_intermediate and (t % save_interval == 0 or t == 0):
            intermediates.append(x.cpu())
    return intermediates if return_intermediate else x


@torch.no_grad()
def sampleCFG(model, n_samples, noise_scheduler, guidance_scale, class_label, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        
    device = next(model.parameters()).device
    n_dim = model.model[-1].out_features
    x = torch.randn(n_samples, n_dim).to(device)
    
    intermediates = [x.cpu()]
    
    for t in tqdm(range(noise_scheduler.num_timesteps-1, -1, -1), desc="CFG Sampling"):
        timesteps = torch.full((n_samples,), t, device=device, dtype=torch.long)
        
        if class_label is not None:
            y = torch.full((n_samples,), class_label, device=device, dtype=torch.long)
            noise_pred_cond = model(x, timesteps, y=y)
            
            noise_pred_uncond = model(x, timesteps, y=None)
            
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            noise_pred = model(x, timesteps, y=None)
        
        alpha = noise_scheduler.alphas[t].to(device)
        alpha_cumprod = noise_scheduler.alphas_cumprod[t].to(device)
        beta = noise_scheduler.betas[t].to(device)
        
        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)  
        
        sqrt_alpha = torch.sqrt(alpha)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod)
    
        # x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / 
        #      torch.sqrt(1 - alpha_cumprod)) * noise_pred) + \
        #      torch.sqrt(beta) * noise
        
        x = (1/sqrt_alpha) * (x - ((1 - alpha)/sqrt_one_minus_alpha_cumprod) * noise_pred) + torch.sqrt(beta) * noise
        
        if t % 20 == 0:
            intermediates.append(x.cpu())
    
    return x, intermediates


# @torch.no_grad()
# def sample_ddim(model, n_samples, noise_scheduler, eta=0.0, steps=50, return_intermediate=False , initial_noise=None):
#     """DDIM sampling with proper time step handling"""
#     device = next(model.parameters()).device
#     n_dim = model.model[-1].out_features
    
#     if initial_noise is not None:
#         x = initial_noise.to(device)
#     else:
#         x = torch.randn(n_samples, n_dim).to(device)
    
    
#     T = noise_scheduler.num_timesteps
#     times_float = torch.linspace(T-1, 0, steps+1)
#     times = torch.round(times_float).to(dtype=torch.long)
#     times[-1] = 0  
#     times = torch.unique(times)  
#     times = sorted(times, reverse=True)    
#     # Use THE SAME noise scheduler as training
#     T = noise_scheduler.num_timesteps
#     step_inc = T // steps
    
#     # Create time subsequence aligned with training scheduler
#     times = torch.arange(0, T, step_inc)
#     times = list(reversed(times.int().tolist()))
    
#     x = torch.randn(n_samples, n_dim).to(device)
#     intermediates = []
    
#     # Precompute all alpha values
#     alpha_cumprods = noise_scheduler.alphas_cumprod.to(device)
    
#     for i in tqdm(range(len(times)), desc="DDIM Sampling"):
#         t = times[i]
#         t_prev = times[i+1] if i < len(times)-1 else -1
        
#         # Model prediction (critical to use correct time embedding)
#         timestep = torch.full((n_samples,), t, device=device, dtype=torch.long)
#         noise_pred = model(x, timestep)
        
#         # Get alpha values for current and previous steps
#         alpha_cumprod = alpha_cumprods[t]
#         alpha_cumprod_prev = alpha_cumprods[t_prev] if t_prev >= 0 else torch.tensor(1.0)
        
#         # DDIM update parameters
#         sigma = eta * torch.sqrt((1 - alpha_cumprod_prev)/(1 - alpha_cumprod)) * \
#                 torch.sqrt(1 - alpha_cumprod/alpha_cumprod_prev)
        
#         # Predicted initial image (x0)
#         pred_x0 = (x - torch.sqrt(1 - alpha_cumprod) * noise_pred) / torch.sqrt(alpha_cumprod)
        
#         # Direction pointing to xt
#         dir_xt = torch.sqrt(1 - alpha_cumprod_prev - sigma**2) * noise_pred
        
#         # Apply noise for stochastic sampling
#         if eta == 0:
#             noise = 0
#         else:
#             noise = sigma * torch.randn_like(x)
            
#         # Update x with proper device placement
#         x = torch.sqrt(alpha_cumprod_prev) * pred_x0 + dir_xt + noise
        
#         if return_intermediate:
#             intermediates.append(x.cpu())
    
#     return intermediates if return_intermediate else x



@torch.no_grad()
def sample_ddim(model, n_samples, noise_scheduler, eta=0.0, steps=50, return_intermediate=False, initial_noise=None):
    device = next(model.parameters()).device
    n_dim = model.model[-1].out_features
    
    # Use same initial noise for reproducibility
    if initial_noise is not None:
        x = initial_noise.to(device)
    else:
        x = torch.randn(n_samples, n_dim).to(device)
    
    # Generate timestep subsequence aligned with training
    T = noise_scheduler.num_timesteps
    times = torch.linspace(0, T-1, steps+1).flip(0).long().unique().tolist()
    times = list(reversed(times))  # Reverse to go from T-1 to 0
    
    intermediates = []
    alpha_cumprods = noise_scheduler.alphas_cumprod.to(device)
    alpha_cumprod_prev = noise_scheduler.alphas_cumprod_prev.to(device)
    
    for i in tqdm(range(len(times)), desc="DDIM Sampling"):
        t = times[i]
        t_prev = times[i+1] if i < len(times)-1 else -1
        
        # Model prediction
        timestep = torch.full((n_samples,), t, device=device, dtype=torch.long)
        pred_noise = model(x, timestep)
        
        # Alpha values for current and previous steps
        alpha = alpha_cumprods[t]
        alpha_prev = alpha_cumprod_prev[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
        
        # DDIM update rule
        pred_x0 = (x - torch.sqrt(1 - alpha) * pred_noise) / torch.sqrt(alpha)
        dir_xt = torch.sqrt(torch.clamp(1 - alpha_prev - (eta**2) * (1 - alpha), min=1e-6)) * pred_noise
        noise = eta * torch.sqrt(1 - alpha) * torch.randn_like(x)
        
        x = torch.sqrt(alpha_prev) * pred_x0 + dir_xt + noise
        
        if return_intermediate:
            intermediates.append(x.cpu())
    
    return intermediates if return_intermediate else x


def compare_ddpm_ddim(model, original_noise_scheduler, n_samples=1000, steps=50):
    """Compare DDPM and DDIM sampling with visualization"""
    # DDPM sampling with original steps
    
    initial_noise = torch.randn(n_samples, model.model[-1].out_features)
    noise_scheduler = original_noise_scheduler
    ddpm_samples = sample(model, n_samples, noise_scheduler,initial_noise=initial_noise)
    
    print(ddpm_samples.shape)
    print(ddpm_samples[:5])
    # DDIM sampling with reduced steps
    ddim_samples = sample_ddim(model, n_samples, noise_scheduler, eta=0, steps=steps,)
    
    # Visualization
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(ddpm_samples[:, 0].cpu(), ddpm_samples[:, 1].cpu(), 
                alpha=0.5, label=f'DDPM ({noise_scheduler.num_timesteps} steps)')
    plt.title("Original DDPM Sampling")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.scatter(ddim_samples[:, 0].cpu(), ddim_samples[:, 1].cpu(),
                alpha=0.5, label=f'DDIM ({steps} steps)', color='orange')
    plt.title("DDIM Accelerated Sampling")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('ddpm_vs_ddim.png')
    # plt.savefig(f'{run_name}/ddpm_vs_ddim_enhanced.png')
    plt.show()
    
    # Quantitative comparison (FID would be better but needs image data)
    print(f"DDPM Sample Mean: {ddpm_samples.mean(0).cpu().detach().numpy()}")
    print(f"DDIM Sample Mean: {ddim_samples.mean(0).cpu().detach().numpy()}")
    print(f"DDPM Sample Std: {ddpm_samples.std(0).cpu().detach().numpy()}")
    print(f"DDIM Sample Std: {ddim_samples.std(0).cpu().detach().numpy()}")
    
def interpolate_ddim(model, z1, z2, noise_scheduler, steps=10):
    """Latent space interpolation with DDIM"""
    alphas = torch.linspace(0, 1, steps)
    return torch.stack([sample_ddim(model, 1, noise_scheduler, initial_noise=a*z1 + f(1-a)*z2) for a in alphas])

    
def sampleSVDD(model, n_samples, noise_scheduler, reward_scale, reward_fn):
    """
    Sample from the SVDD model

    Args:
        model: ConditionalDDPM
        n_samples: int
        noise_scheduler: NoiseScheduler
        reward_scale: float
        reward_fn: callable, takes in a batch of samples torch.Tensor:[n_samples, n_dim] and returns torch.Tensor[n_samples]

    Returns:
        torch.Tensor, samples from the model [n_samples, n_dim]
    """
    pass


def compare_reverse_processes(model, noise_scheduler, ddim_steps=50, n_samples=1000, run_name="comparison"):
    # Generate samples with intermediate steps
    ddpm_intermediates = sample(model, n_samples, noise_scheduler, return_intermediate=True)
    ddim_intermediates = sample_ddim(model, n_samples, noise_scheduler, eta=0, 
                                steps=ddim_steps, return_intermediate=True)

    # Create visualization figure
    plt.figure(figsize=(20, 10))
    plt.suptitle("Reverse Process Comparison: DDPM vs DDIM", y=1.02, fontsize=16)

    # DDPM Visualization
    ddpm_steps_to_show = [0, 200, 400, 600, 800, 999]  # Corresponding to 1000, 800,...1
    for i, idx in enumerate(ddpm_steps_to_show):
        plt.subplot(2, 6, i+1)
        x = ddpm_intermediates[min(idx, len(ddpm_intermediates)-1)]
        plt.scatter(x[:, 0].cpu().numpy(), x[:, 1].cpu().numpy(), s=1, alpha=0.5)
        plt.title(f"DDPM Step {noise_scheduler.num_timesteps - idx}")
        plt.grid(True, alpha=0.3)
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)

    # DDIM Visualization
    ddim_steps_to_show = [0, 10, 20, 30, 40, 49]  # Corresponding to 50, 40,...1
    for i, idx in enumerate(ddim_steps_to_show):
        plt.subplot(2, 6, i+7)
        x = ddim_intermediates[min(idx, len(ddim_intermediates)-1)]
        plt.scatter(x[:, 0].cpu().numpy(), x[:, 1].cpu().numpy(), s=1, alpha=0.5, color='orange')
        plt.title(f"DDIM Step {ddim_steps - idx}")
        plt.grid(True, alpha=0.3)
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)

    plt.tight_layout()
    plt.savefig(f"{run_name}/reverse_process_comparison.png", bbox_inches='tight')
    plt.close()
    print(f"Saved comparison visualization to {run_name}/reverse_process_comparison.png")
    
    
    
def compare_ddpm_ddim(model, original_noise_scheduler, n_samples=1000, ddim_steps=50):
    """Enhanced DDPM vs DDIM comparison with quantitative metrics"""
    
    # Ensure we use THE SAME noise scheduler for both methods
    noise_scheduler = original_noise_scheduler  # Use trained scheduler
    
    # Generate shared initial noise for fair comparison
    initial_noise = torch.randn(n_samples, model.model[-1].out_features)

    # 1. DDPM Sampling
    start = time.time()
    ddpm_samples = sample(model, n_samples, noise_scheduler, initial_noise=initial_noise)
    ddpm_time = time.time() - start

    # 2. DDIM Sampling (with same initial noise)
    start = time.time()
    ddim_samples = sample_ddim(model, n_samples, noise_scheduler, eta=0, steps=ddim_steps, initial_noise=initial_noise)
    ddim_time = time.time() - start

    # Convert to numpy for analysis
    ddpm_np = ddpm_samples.cpu().numpy().flatten()
    ddim_np = ddim_samples.cpu().numpy().flatten()

    # 3. Quantitative Metrics
    print("\n=== Distribution Statistics ===")
    print(f"DDPM Samples - Mean: {np.mean(ddpm_np):.4f}, Std: {np.std(ddpm_np):.4f}")
    print(f"DDIM Samples - Mean: {np.mean(ddim_np):.4f}, Std: {np.std(ddim_np):.4f}")

    # Wasserstein distance
    w_dist = wasserstein_distance(ddpm_np, ddim_np)
    print(f"\nWasserstein Distance: {w_dist:.4f}")

    # Timing
    print("\n=== Timing ===")
    print(f"DDPM ({noise_scheduler.num_timesteps} steps): {ddpm_time:.2f}s")
    print(f"DDIM ({ddim_steps} steps): {ddim_time:.2f}s")
    print(f"Speedup Factor: {ddpm_time/ddim_time:.1f}x")

    # 4. Enhanced Visualization
    plt.figure(figsize=(16, 8))

    # DDPM Plot
    plt.subplot(1, 2, 1)
    plt.hist2d(ddpm_samples[:, 0].cpu().numpy(), ddpm_samples[:, 1].cpu().numpy(),bins=100, cmap='viridis', range=[[-3, 3], [-3, 3]])
    plt.colorbar()
    plt.title(f"DDPM ({noise_scheduler.num_timesteps} steps)")

    # DDIM Plot
    plt.subplot(1, 2, 2)
    plt.hist2d(ddim_samples[:, 0].cpu().numpy(), ddim_samples[:, 1].cpu().numpy(),bins=100, cmap='viridis', range=[[-3, 3], [-3, 3]])
    plt.colorbar()
    plt.title(f"DDIM ({ddim_steps} steps)")

    plt.savefig('ddpm_vs_ddim_enhanced.png')
    plt.close()
    
    compare_reverse_processes(model, original_noise_scheduler, ddim_steps=ddim_steps, n_samples=n_samples,run_name=run_name)
    
########################### Visualization #############################

def visualize_dataset(dataset_name, data_X, data_y, noise_scheduler, model=None, run_name=None):
    """
    Visualize the dataset and optionally the forward diffusion process or denoising process.
    
    Args:
        dataset_name: str, name of the dataset
        data_X: torch.Tensor, data points
        data_y: torch.Tensor, labels (if available)
        noise_scheduler: NoiseScheduler, noise scheduler instance
        model: Optional[DDPM], if provided, will visualize the denoising process using the model
        run_name: Optional[str], directory to save visualizations
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation  
    import numpy as np
    import os
    
    if run_name:
        viz_dir = os.path.join(run_name, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
    else:
        viz_dir = 'visualizations'
        os.makedirs(viz_dir, exist_ok=True)
    
    data_dim = data_X.shape[1]
    
    if data_dim > 3:
        print(f"Data dimensionality ({data_dim}) is too high for direct visualization.")
        return
    
    plt.figure(figsize=(10, 10))
    
    if data_dim == 2:
        # 2D visualization
        plt.scatter(data_X[:, 0].cpu().numpy(), data_X[:, 1].cpu().numpy(), s=1, alpha=0.5)
        plt.title(f"{dataset_name} Dataset")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.grid(True)
        plt.savefig(os.path.join(viz_dir, f"{dataset_name}_dataset.png"))
        plt.close()
        
    
    elif data_dim == 3:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data_X[:, 0].cpu().numpy(), data_X[:, 1].cpu().numpy(), data_X[:, 2].cpu().numpy(), s=1, alpha=0.5)
        ax.set_title(f"{dataset_name} Dataset")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_zlabel("Dimension 3")
        plt.savefig(os.path.join(viz_dir, f"{dataset_name}_dataset.png"))
        plt.close()
        
        print("3D diffusion visualization not implemented for brevity.")

@torch.no_grad()
def debug_sample(model, n_samples, noise_scheduler, return_intermediate=True):
    device = next(model.parameters()).device
    x = torch.randn(n_samples, model.model[-1].out_features).to(device)
    
    intermediates = []
    pred_noises = []
    pred_x0s = []
    
    for t in tqdm(range(noise_scheduler.num_timesteps-1, -1, -1), desc="Debug Sampling"):
        # Current timestep setup
        timesteps = torch.full((n_samples,), t, device=device, dtype=torch.long)
        
        # Model predictions
        pred_noise = model(x, timesteps)
        
        # Calculate predicted x0
        alpha_t = noise_scheduler.alphas_cumprod[t].to(device)
        pred_x0 = (x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
        
        # Store debug info
        if return_intermediate:
            intermediates.append(x.cpu())
            pred_noises.append(pred_noise.cpu())
            pred_x0s.append(pred_x0.cpu())
        
        # DDPM reverse process step
        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
            
        x = (1 / torch.sqrt(noise_scheduler.alphas[t])) * (
            x - ((1 - noise_scheduler.alphas[t])/torch.sqrt(1 - noise_scheduler.alphas_cumprod[t])) * pred_noise
        ) + torch.sqrt(noise_scheduler.betas[t]) * noise
    
    return {
        'samples': x.cpu(),
        'intermediates': intermediates,
        'pred_noises': pred_noises,
        'pred_x0s': pred_x0s
    }

#######################################################################
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['train', 'sample', 'visualize', 'classify' , 'compare'], default='sample')
    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--lbeta", type=float, default=0.0001)
    parser.add_argument("--ubeta", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dataset", type=str, default='moons')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_dim", type=int, default=None)
    parser.add_argument("--visualize", action='store_true', help="Enable visualization")
    parser.add_argument("--type", type=str, default="linear", help="Noise schedule type: 'linear' or 'cosine'")
    parser.add_argument("--guidance_scale", type=float, default=0.0)
    parser.add_argument("--class_label", type=int, default=None)
    parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--run_name", type=str, default="exps5/ddpm_default", help="Directory to save/load models")
    parser.add_argument("--ddim_steps", type=int, default=50, help="Number of steps for DDIM sampling")
    parser.add_argument("--eta", type=float, default=0.0,help="DDIM eta parameter (0=deterministic)")
    parser.add_argument('--ddim_steps_list', nargs='+', type=int, default=[50, 40, 30, 20, 10, 1], help='List of DDIM step counts to visualize')
    
    args = parser.parse_args()
    
    utils.seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # run_name = f'exps5/ddpm_{args.n_dim}_{args.n_steps}_{args.lbeta}_{args.ubeta}_{args.epochs}_{args.batch_size}_{args.lr}_{args.dataset}'
    run_name = args.run_name
    os.makedirs(run_name, exist_ok=True)

    data_X, data_y = dataset.load_dataset(args.dataset)
    data_X = (data_X - data_X.mean(0)) / (data_X.std(0) + 1e-8)
    if data_y is None:
        data_y = torch.zeros(len(data_X)) 


    if args.n_dim is None:
        args.n_dim = data_X.shape[1]
        print(f"Auto-detected data dimension: {args.n_dim}")

    
    # run_name = f'exps5/ddpm_{args.n_dim}_{args.n_steps}_{args.lbeta}_{args.ubeta}_{args.epochs}_{args.batch_size}_{args.lr}_{args.dataset}'
    # os.makedirs(run_name, exist_ok=True)
    
    model = DDPM(n_dim=args.n_dim, n_steps=args.n_steps)
    
    if args.type == "linear":
        noise_scheduler = NoiseScheduler(
            num_timesteps=args.n_steps,
            type=args.type,
            beta_start=args.lbeta,
            beta_end=args.ubeta
        )
    elif args.type == "cosine":
        noise_scheduler = NoiseScheduler(
            num_timesteps=args.n_steps,
            type=args.type
        )

    plt.figure(figsize=(10, 6))
    plt.plot(noise_scheduler.betas.cpu(), label='beta')
    plt.plot(noise_scheduler.alphas.cpu(), label='alpha')
    plt.plot(noise_scheduler.alphas_cumprod.cpu(), label='alpha_cumprod')
    plt.title("Noise Schedule Parameters")
    plt.legend()
    plt.savefig(f"{run_name}/schedule_curves.png")
    plt.close()
    
    model = model.to(device)
    
    
    if args.mode == 'train':
        if args.guidance_scale > 0:  
            model = ConditionalDDPM(n_dim=args.n_dim, n_classes=args.n_classes,noise_scheduler=noise_scheduler)
        else:  
            model = DDPM(n_dim=args.n_dim,noise_scheduler=noise_scheduler)
        
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data_X, data_y), 
            batch_size=args.batch_size, 
            shuffle=True
        )
        
        if args.visualize:
            visualize_dataset(args.dataset, data_X, data_y, noise_scheduler, run_name=run_name)
        
        # Train the model
        if args.guidance_scale > 0:
            trainConditional(model, noise_scheduler, dataloader, optimizer, args.epochs, run_name)
        else:
            train(model, noise_scheduler, dataloader, optimizer, args.epochs, run_name)
        
        # Visualize after training
        if args.visualize:
            visualize_dataset(args.dataset, data_X, data_y, noise_scheduler, model=model, run_name=run_name)

        model.eval()
    
        # Generate debug samples
        debug_output = debug_sample(model, 1000, noise_scheduler)
        
        # Visualize reverse process
        print("\nGenerating reverse process visualization...")
        visualize_reverse_ddpm(
        model=model,
        noise_scheduler=noise_scheduler,
        n_samples=5000,
        save_dir=f"{run_name}/reverse_vis"
    )
        
        # Compare with ground truth
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.scatter(data_X[:, 0].cpu().numpy(), data_X[:, 1].cpu().numpy(), s=1, alpha=0.5)
        plt.title("Real Data")
        
        plt.subplot(122)
        plt.scatter(debug_output['samples'][:, 0], debug_output['samples'][:, 1], s=1, alpha=0.5)
        plt.title("Generated Samples")
        plt.savefig(f"{run_name}/comparison.png")
        plt.close()
        
    elif args.mode == 'sample':
        

        if (args.guidance_scale and args.guidance_scale > 0) or args.class_label is not None:
            model = ConditionalDDPM(n_dim=args.n_dim, n_classes=args.n_classes,noise_scheduler=noise_scheduler).to(device)
        else:
            model = DDPM(n_dim=args.n_dim,noise_scheduler=noise_scheduler).to(device)

        model.load_state_dict(torch.load(f'{run_name}/model.pth', map_location=device))
        
        
        if args.class_label is not None :
            #
            samples_conditional = sampleCFG(model, args.n_samples, noise_scheduler,  guidance_scale=0.0, class_label=args.class_label)
        
            samples_guided = sampleCFG(model, args.n_samples, noise_scheduler, 
                                    guidance_scale=7.0, class_label=args.class_label)
            

            visualize_reverse_ddpm(model, noise_scheduler, args.n_samples, run_name)
            torch.save(samples_conditional, f'{run_name}/samples_conditional_{args.class_label}.pth')
            torch.save(samples_guided, f'{run_name}/samples_guided_{args.class_label}.pth')
        
            classifier = ClassifierDDPM(model, noise_scheduler)
            cond_preds = classifier.predict(samples_conditional)
            guided_preds = classifier.predict(samples_guided)
            
            cond_acc = (cond_preds == args.class_label).mean()
            guided_acc = (guided_preds == args.class_label).mean()
            print(f"\nConditional Sampling Accuracy: {cond_acc*100:.1f}%")
            print(f"Guided Sampling Accuracy: {guided_acc*100:.1f}%")
            
        else:
            
            samples = sample(model, args.n_samples, noise_scheduler)
            visualize_reverse_ddpm(model, noise_scheduler, args.n_samples, run_name)
            torch.save(samples, f'{run_name}/samples_unconditional.pth')
            
        intermediates = sample(model, args.n_samples, noise_scheduler, return_intermediate=True)
    
        # Visualize
        plt.figure(figsize=(12, 6))
        for i, x in enumerate(intermediates):
            if i % 100 == 0:
                plt.scatter(x[:, 0].cpu().numpy(), x[:, 1].cpu().numpy(), 
                            alpha=0.5, label=f"Step {i}")
        plt.legend()
        plt.savefig(f"{run_name}/sampling_steps.png")
        plt.close()
        
    elif args.mode == 'classify':
        
        test_X, test_y = dataset.load_dataset(args.dataset)
        test_loader = DataLoader(
            torch.utils.data.TensorDataset(test_X, test_y), 
            batch_size=args.batch_size
        )
        
        classifier = ClassifierDDPM(model, noise_scheduler, num_steps=20)
        all_preds = []
        all_labels = []
        
        for x, y in test_loader:
            x = x.to(device)
            preds = classifier.predict(x)
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
        
        cfg_accuracy = accuracy_score(all_labels, all_preds)
        print(f"\nCFG Classifier Accuracy: {cfg_accuracy*100:.2f}%")
        
        

        mlp_classifier = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300)
        mlp_classifier.fit(test_X.cpu().numpy(), all_labels)  
        mlp_preds = mlp_classifier.predict(test_X.numpy())
        mlp_accuracy = accuracy_score(all_labels, mlp_preds)
        print(f"MLP Classifier Accuracy: {mlp_accuracy*100:.2f}%")

    # elif args.mode == 'sample':
    #     if args.guidance_scale is not None or args.class_label is not None:
    #         model = ConditionalDDPM(n_dim=args.n_dim, n_classes=args.n_classes, 
    #                             noise_scheduler=noise_scheduler).to(device)
    #     else:
    #         model = DDPM(n_dim=args.n_dim, noise_scheduler=noise_scheduler).to(device)

    #     model.load_state_dict(torch.load(f'{run_name}/model.pth', map_location=device))
        
    #     if args.class_label is not None:
    #         # For single guidance scale evaluation
    #         if args.guidance_scale is not None:
    #             # Conditional Sampling
    #             samples_conditional, _ = sampleCFG(
    #                 model, args.n_samples, noise_scheduler, 
    #                 guidance_scale=0.0, class_label=args.class_label,
    #                 seed=args.seed
    #             )
                
    #             # Guided Sampling
    #             samples_guided, _ = sampleCFG(
    #                 model, args.n_samples, noise_scheduler, 
    #                 guidance_scale=args.guidance_scale, class_label=args.class_label,
    #                 seed=args.seed+1  # Use different seed
    #             )
                
    #             # Save samples
    #             torch.save(samples_conditional, 
    #                     f'{run_name}/samples_conditional_{args.class_label}.pth')
    #             torch.save(samples_guided, 
    #                     f'{run_name}/samples_guided_{args.class_label}_gs{args.guidance_scale}.pth')
                
    #             # Evaluate class consistency
    #             classifier = ClassifierDDPM(model, noise_scheduler, num_steps=50)
                
    #             # Use different seeds for evaluation
    #             cond_preds = classifier.predict(samples_conditional, seed=args.seed+10)
    #             guided_preds = classifier.predict(samples_guided, seed=args.seed+11)
                
    #             cond_acc = (cond_preds == args.class_label).mean()
    #             guided_acc = (guided_preds == args.class_label).mean()
                
    #             print(f"\nConditional Sampling Accuracy: {cond_acc*100:.1f}%")
    #             print(f"Guided Sampling (scale={args.guidance_scale}) Accuracy: {guided_acc*100:.1f}%")
                
    #             # Create visualization
    #             visualize_guidance_effect(model, noise_scheduler, n_samples=1000, 
    #                                     class_label=args.class_label,
    #                                     scales=[0.0, args.guidance_scale])
    #         else:
    #             # Comprehensive evaluation of multiple guidance scales
    #             scales = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    #             results, summary = evaluate_guidance_scales(
    #                 model, noise_scheduler, 
    #                 n_samples=args.n_samples,
    #                 class_label=args.class_label,
    #                 scales=scales,
    #                 num_runs=3
    #             )
                
    #             # Save results
    #             np.save(f'{run_name}/guidance_scale_results_{args.class_label}.npy', results)
                
    #             # Create visualization of select scales
    #             visualize_guidance_effect(model, noise_scheduler, n_samples=1000, 
    #                                     class_label=args.class_label,
    #                                     scales=[0.0, 3.0, 7.0])
    #     else:
    #         # Standard unconditional sampling
    #         samples = sample(model, args.n_samples, noise_scheduler)
    #         torch.save(samples, f'{run_name}/samples_unconditional.pth')

    elif args.mode == 'visualize':
        
        if os.path.exists(f'{run_name}/model.pth'):
            model.load_state_dict(torch.load(f'{run_name}/model.pth', weights_only=True))
            model.eval()
        else:
            print(f"Warning: No trained model found at {run_name}/model.pth")
            model = None
        
        viz_dir = os.path.join(run_name, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        print(f"Visualizing dataset: {args.dataset}")
        if args.n_dim <= 3:  
            data_X = data_X.to(device)
            if data_y is not None:
                data_y = data_y.to(device)
            
            visualize_dataset(args.dataset, data_X, data_y, noise_scheduler, model=model, run_name=run_name)
            
            samples_path = f'{run_name}/samples_{args.seed}_{args.n_samples}.pth'
            if os.path.exists(samples_path):
                print(f"Found samples at {samples_path}, visualizing...")
                samples = torch.load(samples_path)
                
                plt.figure(figsize=(10, 10))
                if args.n_dim == 2:
                    plt.scatter(samples[:, 0].cpu().numpy(), samples[:, 1].cpu().numpy(), s=1, alpha=0.5)
                    plt.title(f"{args.dataset} Generated Samples")
                    plt.grid(True)
                    plt.savefig(f"{viz_dir}/{args.dataset}_generated_samples.png")
                    plt.close()
                    
                    plt.figure(figsize=(16, 8))
                    plt.subplot(1, 2, 1)
                    plt.scatter(data_X[:, 0].cpu().numpy(), data_X[:, 1].cpu().numpy(), s=1, alpha=0.5)
                    plt.title("Original Data")
                    plt.grid(True)
                    
                    plt.subplot(1, 2, 2)
                    plt.scatter(samples[:, 0].cpu().numpy(), samples[:, 1].cpu().numpy(), s=1, alpha=0.5)
                    plt.title("Generated Samples")
                    plt.grid(True)
                    
                    plt.tight_layout()
                    plt.savefig(f"{viz_dir}/{args.dataset}_comparison.png")
                    plt.close()
                
                elif args.n_dim == 3:
                    fig = plt.figure(figsize=(10, 10))
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(samples[:, 0].cpu().numpy(), samples[:, 1].cpu().numpy(), samples[:, 2].cpu().numpy(), s=1, alpha=0.5)
                    plt.title(f"{args.dataset} Generated Samples (3D)")
                    plt.savefig(f"{viz_dir}/{args.dataset}_generated_samples_3d.png")
                    plt.close()
            else:
                print(f"No samples found at {samples_path}")
            
        else:
            print(f"Visualization not implemented for n_dim={args.n_dim}")
            
    # elif args.mode == 'compare':
    #     # Load pretrained model
    #     model = DDPM(n_dim=args.n_dim).to(device)
    #     model.load_state_dict(torch.load(f'{run_name}/model.pth', map_location=device))
        
    #     # Initialize proper noise scheduler for comparison
    #     noise_scheduler = NoiseScheduler(
    #         num_timesteps=1000,
    #         type='linear',
    #         beta_start=0.0001,
    #         beta_end=0.02
    #     )
        
    #     # 1. Visual comparison
    #     compare_ddpm_ddim(model, noise_scheduler, n_samples=1000, steps=50)
        
    #     # 2. Timing tests
    #     print("\nTiming Comparison:")
    #     start = time.time()
    #     sample(model, 1000, noise_scheduler)
    #     ddpm_time = time.time()-start
    #     print(f"DDPM time ({noise_scheduler.num_timesteps} steps): {ddpm_time:.2f}s")
        
    #     start = time.time()
    #     sample_ddim(model, 1000, noise_scheduler, steps=50)
    #     ddim_time = time.time()-start
    #     print(f"DDIM time (50 steps): {ddim_time:.2f}s")
    #     print(f"Speedup factor: {ddpm_time/ddim_time:.1f}x")
        
    #     # 3. Interpolation comparison
    #     print("\nInterpolation Comparison:")
    #     z1 = torch.randn(1, args.n_dim).to(device)
    #     z2 = torch.randn(1, args.n_dim).to(device)
    #     interp_samples = interpolate_ddim(model, z1, z2, noise_scheduler, steps=10)
        
    #     # Visualize interpolation
    #     plt.figure(figsize=(10, 5))
    #     for i, sample in enumerate(interp_samples):
    #         plt.subplot(1, 10, i+1)
    #         plt.scatter(sample[:, 0].cpu(), sample[:, 1].cpu(), s=10)
    #         plt.axis('off')
    #     plt.suptitle("DDIM Latent Space Interpolation")
    #     plt.savefig('ddim_interpolation.png')
    #     plt.close()
    # print(f"Completed {args.mode} mode for {args.dataset} dataset")

    elif args.mode == 'compare':
        # Load pretrained model
        model = DDPM(n_dim=args.n_dim, noise_scheduler=noise_scheduler).to(device)
        model.load_state_dict(torch.load(f'{run_name}/model.pth', map_location=device))
        
        # Use THE SAME noise scheduler used during training
        noise_scheduler = NoiseScheduler(
            num_timesteps=args.n_steps,
            type=args.type,
            beta_start=args.lbeta,
            beta_end=args.ubeta
        )
        
        # compare_ddpm_ddim(model = model , noise_scheduler = noise_scheduler, n_samples=args.n_samples,ddim_steps=args.ddim_steps )
        compare_ddpm_ddim(
        model=model,
        original_noise_scheduler=noise_scheduler,  # Changed parameter name
        n_samples=args.n_samples,
        ddim_steps=args.ddim_steps
    )
        
        # DDIM visualization
        # visualize_reverse_ddim(
        #     model=model,
        #     noise_scheduler=noise_scheduler,
        #     n_samples=args.n_samples,
        #     save_dir=f"{run_name}/ddim_vis",
        #     steps=args.ddim_steps,
        #     eta=args.eta
        # )
        
        visualize_reverse_ddim(
            model=model,
            noise_scheduler=noise_scheduler,
            n_samples=args.n_samples,
            base_save_dir=f"{run_name}/ddim_vis",  # Changed parameter name
            steps_list=[50, 40, 30, 20, 10, 1],    # New list parameter
            eta=args.eta
        )
        
        # DDPM visualization
        visualize_reverse_ddpm(
            model=model,
            noise_scheduler=noise_scheduler,
            n_samples=args.n_samples,
            save_dir=f"{run_name}/ddpm_vis"
        )