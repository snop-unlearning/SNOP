import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
from tqdm import tqdm
import os
import time

from vae_utils import (
    set_seed,
    get_mnist_transform,
    split_indices,
    vae_loss_function,
    hyperparams,
    evaluate_vae,
    generate_samples,
    load_vae_model
)

# --- Hyperparameters ---
LATENT_DIM = hyperparams["latent_dim"]
IMAGE_CHANNELS = hyperparams["image_channels"]
INIT_CHANNELS = hyperparams["init_channels"]
# IMAGE_SIZE is also available from hyperparams in vae_utils

# --- SSD Class (Adapted for VAE and using I_D) ---
class SSD_VAE:
    def __init__(self, model, alpha=0.9, lambd=0.8):
        # Store the original model's state to ensure Fisher is always on the pristine model
        self.original_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        self.model_for_fisher = copy.deepcopy(model) # Use a dedicated copy for Fisher calculations
        self.alpha = alpha
        self.lambd = lambd

    def compute_importances_vae(self, dataloader, device='cuda', desc_prefix="SSD Fisher"):
        importances = {}
        # Ensure the model for Fisher is on the correct device and in eval mode
        self.model_for_fisher.to(device)
        self.model_for_fisher.eval()

        for name, param in self.model_for_fisher.named_parameters():
            if param.requires_grad:
                importances[name] = torch.zeros_like(param.data, device=device)

        num_samples = 0
        dataset_size = len(dataloader.dataset)
        if dataset_size == 0: # Guard against empty dataset for description
            print(f"Warning: {desc_prefix} Dataloader is empty. Importances will be zero.")
            for name in importances: importances[name].zero_()
            return importances

        desc_text = f"{desc_prefix} (VAE Recon Loss) for {dataset_size} samples"

        for inputs, _ in tqdm(dataloader, desc=desc_text, leave=False):
            if inputs.numel() == 0: continue
            inputs = inputs.to(device)
            num_samples += inputs.size(0)

            self.model_for_fisher.zero_grad(set_to_none=True)
            recon_batch, mu, logvar = self.model_for_fisher(inputs)
            # We only need reconstruction loss for Fisher, as per many unlearning papers focusing on sample influence
            recon_loss, _ = vae_loss_function(recon_batch, inputs, mu, logvar)
            # Average loss per sample for stable gradient magnitudes
            loss_for_grad = recon_loss / inputs.size(0) if inputs.size(0) > 0 else recon_loss
            loss_for_grad.backward()

            for name, param in self.model_for_fisher.named_parameters():
                if param.grad is not None and name in importances:
                    importances[name] += param.grad.data.clone().pow(2) # Accumulate squared gradients

        self.model_for_fisher.zero_grad(set_to_none=True) # Clear grads from the Fisher model

        if num_samples > 0:
            for name in importances:
                importances[name] /= num_samples # Normalize by number of samples
        else: # Should be caught by dataset_size check, but for safety
            print(f"Warning: {desc_prefix} No samples processed. Importances will be zero.")
            for name in importances: importances[name].zero_()
        return importances

    def unlearn(self, full_original_train_loader, forget_loader, device='cuda'):
        start_time = time.time()
        # Load original model state into the model_for_fisher to ensure pristine weights
        self.model_for_fisher.load_state_dict(self.original_model_state)
        # self.model_for_fisher.to(device) # Already done in compute_importances_vae
        # self.model_for_fisher.eval() # Already done in compute_importances_vae

        if len(full_original_train_loader.dataset) == 0:
            print("Warning: SSD Full Original Train loader is empty. Skipping unlearning.")
            # Return a copy of the model with original weights if unlearning cannot proceed
            unlearned_model_pristine = copy.deepcopy(self.model_for_fisher)
            unlearned_model_pristine.load_state_dict(self.original_model_state)
            return unlearned_model_pristine.to(device)

        if len(forget_loader.dataset) == 0:
            print("Warning: SSD Forget loader is empty. Skipping unlearning.")
            unlearned_model_pristine = copy.deepcopy(self.model_for_fisher)
            unlearned_model_pristine.load_state_dict(self.original_model_state)
            return unlearned_model_pristine.to(device)

        print("SSD: Computing Fisher Information for full original training data (I_D)...")
        importance_D = self.compute_importances_vae(full_original_train_loader, device, desc_prefix="SSD FullTrain Fisher")
        print("SSD: Computing Fisher Information for forget set data (I_Df)...")
        importance_Df = self.compute_importances_vae(forget_loader, device, desc_prefix="SSD Forget Fisher")

        # Create the unlearned model by copying the original state first
        unlearned_model = copy.deepcopy(self.model_for_fisher) # Start with a fresh copy of the original model
        unlearned_model.load_state_dict(self.original_model_state)
        unlearned_model.to(device)
        unlearned_model.eval() # Set to eval for consistency during modification, can be set to train later if needed

        params_dampened_count = 0
        total_params_considered = 0
        print("SSD: Applying selective synaptic dampening...")
        with torch.no_grad():
            for name, param in unlearned_model.named_parameters():
                if not param.requires_grad: continue
                total_params_considered +=1
                if name in importance_D and name in importance_Df:
                    imp_D_param = importance_D[name].to(param.device)
                    imp_Df_param = importance_Df[name].to(param.device)
                    epsilon = 1e-12 # For division stability

                    # Selection Criterion: I_Df > alpha * I_D
                    # FIM values are non-negative.
                    mask = imp_Df_param > (self.alpha * imp_D_param)

                    if mask.sum() > 0:
                        params_dampened_count +=1
                        beta = torch.ones_like(param.data)

                        # Dampening Factor: beta_val = lambda * I_D / I_Df
                        # Add epsilon to the denominator I_Df to prevent division by zero
                        numerator_damp = self.lambd * imp_D_param[mask]
                        denominator_damp = imp_Df_param[mask] + epsilon # Denominator is I_Df
                        damp_values = numerator_damp / denominator_damp

                        # Cap dampening factor at 1: min(beta_val, 1)
                        beta[mask] = torch.min(damp_values, torch.ones_like(damp_values))
                        param.data.mul_(beta) # Apply dampening

        if total_params_considered > 0:
             print(f"SSD: Dampened parameters in {params_dampened_count} out of {total_params_considered} trainable parameter tensors/groups.")
        else:
            print("SSD: No trainable parameters found to dampen.")
        unlearning_time = time.time() - start_time
        print(f"--- SSD VAE Unlearning Complete (Time: {unlearning_time:.2f}s) ---")
        return unlearned_model

# --- Main Execution Logic for SSD ---
def run_ssd_unlearning():
    set_seed(42) # Set seed at the beginning
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    kl_weight_for_eval = 1.0 # Used in evaluate_vae

    model_path = 'vae_mnist_pretrained.pth'
    if not (os.path.exists(model_path) and os.path.getsize(model_path) > 0):
        print(f"Error: Pretrained VAE model not found or is empty at '{model_path}'. Exiting."); exit(1)

    model = load_vae_model(
        model_path,
        device,
        latent_dim=LATENT_DIM,
        image_channels=IMAGE_CHANNELS,
        init_channels=INIT_CHANNELS,
        image_size=hyperparams["image_size"]
    )

    classes_to_forget = [5]
    ssd_alpha = 10          # Selection threshold (I_Df > alpha * I_D)
    ssd_lambda = 1.0        # Dampening strength (lambda * I_D / I_Df)
    retain_fraction = 1
    forget_fraction = 1
    
    print(f"\n--- SSD VAE Unlearning Configuration ---")
    print(f"Classes to Forget: {classes_to_forget}")
    print(f"SSD Hyperparams: alpha={ssd_alpha}, lambda={ssd_lambda}")

    print("\n--- Data Preparation (MNIST) ---")
    transform_mnist = get_mnist_transform(image_size=hyperparams["image_size"])
    try:
        full_trainset_mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
        testset_mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
    except Exception as e:
        print(f"Error loading MNIST dataset: {e}"); exit(1)

    # For SSD:
    # 1. I_D is computed on the full_trainset_mnist.
    # 2. I_Df is computed on the forget portion of full_trainset_mnist.
    train_forget_indices, train_retain_indices = split_indices(full_trainset_mnist, classes_to_forget, retain_fraction=retain_fraction, forget_fraction=forget_fraction) # We only need forget_indices from train for I_Df
    
    ssd_full_dataset = Subset(full_trainset_mnist, np.concatenate((train_retain_indices, train_forget_indices)))
    ssd_forget_dataset = Subset(full_trainset_mnist, train_forget_indices)
    # The full_trainset_mnist itself will be used for the full_original_train_loader

    # Test set splitting for evaluation
    test_forget_indices, test_retain_indices = split_indices(testset_mnist, classes_to_forget)
    test_forget_dataset = Subset(testset_mnist, test_forget_indices)
    test_retain_dataset = Subset(testset_mnist, test_retain_indices)

    print(f"Sizes for SSD Importance Calculation: FullTrainSet (for I_D)={len(ssd_full_dataset)}, ForgetSet (for I_Df)={len(ssd_forget_dataset)}")
    print(f"Sizes for Test Evaluation: TestRetain={len(test_retain_dataset)}, TestForget={len(test_forget_dataset)}")

    batch_size = 128
    num_workers = min(os.cpu_count(), 4) if os.cpu_count() else 2 # Use fewer workers if CPU count is low
    pin_memory = True if device.type == 'cuda' else False

    # DataLoaders for SSD importance calculation
    # Loader for I_D (full original training data)
    full_original_train_loader = DataLoader(ssd_full_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    # Loader for I_Df (forget portion of original training data)
    ssd_forget_loader = DataLoader(ssd_forget_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    
    test_retain_loader_eval = DataLoader(test_retain_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_forget_loader_eval = DataLoader(test_forget_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    print("\n--- Original VAE Model Evaluation (Test Set) ---")
    original_model_test_retain_metrics = evaluate_vae(model, test_retain_loader_eval, device, kl_weight_for_eval, "Original Test Retain")
    original_model_test_forget_metrics = evaluate_vae(model, test_forget_loader_eval, device, kl_weight_for_eval, "Original Test Forget")
    print("Original Model Test Metrics:")
    print(f"  Retain -> Recon: {original_model_test_retain_metrics['recon_loss']:.2f}, KLD: {original_model_test_retain_metrics['kld']:.2f}")
    print(f"  Forget -> Recon: {original_model_test_forget_metrics['recon_loss']:.2f}, KLD: {original_model_test_forget_metrics['kld']:.2f}")
    generate_samples(model, LATENT_DIM, n_samples=25, device=device, filename="original_samples_for_ssd_run_32x32.png")

    print("\n--- SSD VAE Unlearning ---")
    ssd_unlearner = SSD_VAE(model, alpha=ssd_alpha, lambd=ssd_lambda)
    
    import time
    start_time = time.time()
    
    unlearned_model_ssd = ssd_unlearner.unlearn(
        full_original_train_loader=full_original_train_loader, # Pass the loader for the full training set for I_D
        forget_loader=ssd_forget_loader,
        device=device
    )

    print(f"SSD Unlearning took {time.time() - start_time:.2f} seconds.")

    print("\n--- SSD Unlearned VAE Model Evaluation (Test Set) ---")
    ssd_unlearned_test_retain_metrics = evaluate_vae(unlearned_model_ssd, test_retain_loader_eval, device, kl_weight_for_eval, "SSD Unlearned Test Retain")
    ssd_unlearned_test_forget_metrics = evaluate_vae(unlearned_model_ssd, test_forget_loader_eval, device, kl_weight_for_eval, "SSD Unlearned Test Forget")
    print("SSD Unlearned Model Test Metrics:")
    print(f"  Retain -> Recon: {ssd_unlearned_test_retain_metrics['recon_loss']:.2f}, KLD: {ssd_unlearned_test_retain_metrics['kld']:.2f}")
    print(f"  Forget -> Recon: {ssd_unlearned_test_forget_metrics['recon_loss']:.2f}, KLD: {ssd_unlearned_test_forget_metrics['kld']:.2f}")
    generate_samples(unlearned_model_ssd, LATENT_DIM, n_samples=25, device=device, filename="ssd_unlearned_samples_32x32.png")

    print("\n--- VAE Performance Summary (Test Set Recon Loss) ---")
    print(f"Original Model: Retain Recon={original_model_test_retain_metrics['recon_loss']:.2f}, Forget Recon={original_model_test_forget_metrics['recon_loss']:.2f}")
    print(f"SSD Unlearned:  Retain Recon={ssd_unlearned_test_retain_metrics['recon_loss']:.2f}, Forget Recon={ssd_unlearned_test_forget_metrics['recon_loss']:.2f}")
    retain_recon_delta_ssd = ssd_unlearned_test_retain_metrics['recon_loss'] - original_model_test_retain_metrics['recon_loss']
    forget_recon_delta_ssd = ssd_unlearned_test_forget_metrics['recon_loss'] - original_model_test_forget_metrics['recon_loss']
    print(f"  SSD Change:  Retain Δ={retain_recon_delta_ssd:+.2f}, Forget Δ={forget_recon_delta_ssd:+.2f}")
    print("\nDesired outcome: Forget Recon Loss should increase significantly, Retain Recon Loss should increase minimally.")

    torch.save(unlearned_model_ssd.state_dict(), "unlearned_vae_ssd.pth")

if __name__ == "__main__":
    
    run_ssd_unlearning()