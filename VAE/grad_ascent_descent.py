# grad_ascent_descent_vae.py
import torch
import copy
import torchvision
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import os
import time
from itertools import cycle

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

LATENT_DIM = hyperparams["latent_dim"]
IMAGE_CHANNELS = hyperparams["image_channels"]
INIT_CHANNELS = hyperparams["init_channels"]
# IMAGE_SIZE is also available from hyperparams in vae_utils

class GradAscentDescentVAE:
    def __init__(self, kl_weight=1.0, retain_loss_weight=1.0, forget_loss_weight=1.0):
        self.kl_weight = kl_weight
        self.alpha = retain_loss_weight  # Retain loss weight (for descent)
        self.beta = forget_loss_weight   # Forget loss weight (for ascent)

    def unlearn(self,
                original_model,
                train_retain_loader,
                train_forget_loader,
                val_retain_loader,
                val_forget_loader,
                epochs,
                lr, # This will be the reduced LR
                device,
                patience=7):
        start_time = time.time()

        if len(train_retain_loader.dataset) == 0:
            print("Warning: GradAD Train Retain loader is empty. Skipping unlearning.")
            return copy.deepcopy(original_model).to(device)
        if len(train_forget_loader.dataset) == 0:
            print("Warning: GradAD Train Forget loader is empty. Skipping unlearning.")
            return copy.deepcopy(original_model).to(device)


        unlearned_model = copy.deepcopy(original_model)
        unlearned_model.to(device)
        optimizer = torch.optim.AdamW(unlearned_model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=patience // 2, verbose=True, threshold=1e-4)

        best_val_retain_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        best_epoch = -1

        iter_train_forget_loader = cycle(train_forget_loader)

        print(f"Starting GradAscentDescent Unlearning for {epochs} epochs.")
        print(f"Retain loss weight (alpha): {self.alpha}, Forget loss weight (beta): {self.beta}, KL weight: {self.kl_weight}")
        print(f"Unlearning LR: {lr}")

        for epoch in range(epochs):
            epoch_start_time = time.time()
            unlearned_model.train()
            epoch_total_loss_val = 0.0
            epoch_retain_loss_val = 0.0
            epoch_forget_loss_for_logging_val = 0.0 # For logging full VAE loss on forget
            num_batches_processed = 0

            with tqdm(train_retain_loader, desc=f"Epoch {epoch+1}/{epochs} Training", leave=False, dynamic_ncols=True) as pbar:
                for retain_inputs, _ in pbar:
                    if retain_inputs.numel() == 0: continue
                    retain_inputs = retain_inputs.to(device)

                    forget_inputs, _ = next(iter_train_forget_loader)
                    if forget_inputs.numel() == 0: continue
                    forget_inputs = forget_inputs.to(device)

                    optimizer.zero_grad()

                    # Retain pass (descent on full VAE loss)
                    retain_recon, retain_mu, retain_logvar = unlearned_model(retain_inputs)
                    retain_bce, retain_kld = vae_loss_function(retain_recon, retain_inputs, retain_mu, retain_logvar)
                    avg_retain_bce = retain_bce / retain_inputs.size(0) if retain_inputs.size(0) > 0 else retain_bce
                    avg_retain_kld = retain_kld / retain_inputs.size(0) if retain_inputs.size(0) > 0 else retain_kld
                    current_retain_loss_objective = avg_retain_bce + self.kl_weight * avg_retain_kld

                    # Forget pass (ascent on BCE_forget only)
                    forget_recon, forget_mu, forget_logvar = unlearned_model(forget_inputs)
                    forget_bce, forget_kld = vae_loss_function(forget_recon, forget_inputs, forget_mu, forget_logvar)
                    # Objective for forget set: Maximize reconstruction error
                    avg_forget_bce_objective = forget_bce / forget_inputs.size(0) if forget_inputs.size(0) > 0 else forget_bce
                    
                    # For logging purposes, calculate the full VAE loss on forget data
                    avg_forget_kld_for_logging = forget_kld / forget_inputs.size(0) if forget_inputs.size(0) > 0 else forget_kld
                    current_forget_vae_loss_for_logging = avg_forget_bce_objective.item() + self.kl_weight * avg_forget_kld_for_logging.item()
                    
                    # Combined loss: L = alpha * L_retain_objective - beta * BCE_forget_objective
                    # Optimizer minimizes L. So it minimizes L_retain_objective and maximizes BCE_forget_objective.
                    total_loss_to_minimize = self.alpha * current_retain_loss_objective - self.beta * avg_forget_bce_objective

                    total_loss_to_minimize.backward()
                    torch.nn.utils.clip_grad_norm_(unlearned_model.parameters(), max_norm=1.0)
                    optimizer.step()

                    epoch_total_loss_val += total_loss_to_minimize.item()
                    epoch_retain_loss_val += current_retain_loss_objective.item()
                    epoch_forget_loss_for_logging_val += current_forget_vae_loss_for_logging
                    num_batches_processed += 1
                    
                    pbar.set_postfix(
                        total_L=f"{total_loss_to_minimize.item():.4f}",
                        ret_L_obj=f"{current_retain_loss_objective.item():.4f}",
                        fgt_BCE_obj=f"{avg_forget_bce_objective.item():.4f}", # Log the part of forget loss we optimize
                        fgt_VAE_log=f"{current_forget_vae_loss_for_logging:.4f}" # Log full VAE loss on forget
                    )
            
            avg_epoch_total_loss = epoch_total_loss_val / num_batches_processed if num_batches_processed > 0 else 0
            avg_epoch_retain_loss = epoch_retain_loss_val / num_batches_processed if num_batches_processed > 0 else 0
            avg_epoch_forget_loss_logged = epoch_forget_loss_for_logging_val / num_batches_processed if num_batches_processed > 0 else 0

            val_retain_metrics = evaluate_vae(unlearned_model, val_retain_loader, device, self.kl_weight, "Val Retain")
            val_forget_metrics = evaluate_vae(unlearned_model, val_forget_loader, device, self.kl_weight, "Val Forget")
            epoch_end_time = time.time()

            print(f"Epoch {epoch+1}/{epochs} Summary - Train Avg: TotalLoss={avg_epoch_total_loss:.4f}, RetainLossObj={avg_epoch_retain_loss:.4f}, ForgetVAELogged={avg_epoch_forget_loss_logged:.4f}")
            print(f"  Val Retain -> Recon: {val_retain_metrics['recon_loss']:.2f}, KLD: {val_retain_metrics['kld']:.2f}, Total: {val_retain_metrics['total_loss']:.4f}")
            print(f"  Val Forget -> Recon: {val_forget_metrics['recon_loss']:.2f}, KLD: {val_forget_metrics['kld']:.2f}, Total: {val_forget_metrics['total_loss']:.4f}")
            print(f"  Epoch Time: {epoch_end_time - epoch_start_time:.2f}s")

            scheduler.step(val_retain_metrics['total_loss'])

            current_val_retain_loss = val_retain_metrics['total_loss']
            if current_val_retain_loss < best_val_retain_loss - 1e-5:
                print(f"-> New best validation retain loss: {current_val_retain_loss:.4f} (was {best_val_retain_loss:.4f})")
                best_val_retain_loss = current_val_retain_loss
                best_model_state = copy.deepcopy(unlearned_model.state_dict())
                best_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"  -> No improvement in validation retain loss. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after epoch {epoch+1}.")
                break
        
        unlearning_end_time = time.time()
        print(f"\n--- GradAscentDescent Unlearning Finished (Time: {unlearning_end_time - start_time:.2f}s) ---")
        if best_model_state is not None:
            print(f"Loading best model state from Epoch {best_epoch} (Val Retain Total Loss: {best_val_retain_loss:.4f})")
            unlearned_model.load_state_dict(best_model_state)
        else:
            print("Warning: No improvement detected during training or patience=0. Using model from last epoch.")
        
        torch.save(unlearned_model.state_dict(), 'vae_mnist_gad.pth')    
        return unlearned_model

# --- Main Execution Logic for Gradient Ascent/Descent ---
def run_grad_ad_unlearning():
    set_seed(42) # Set seed at the beginning
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Configurable Parameters ---
    kl_weight_param = 1.0
    retain_loss_weight = 1.0
    forget_loss_weight = 1.0  # Weight for maximizing forget BCE
    unlearning_lr = 1e-5      # REDUCED LEARNING RATE
    unlearning_epochs = 40    # Might need more epochs with smaller LR
    unlearning_patience = 5   # Adjusted patience
    val_split_fraction = 0.1
    retain_split_fraction = 0.1
    forget_split_fraction = 0.1
    classes_to_forget = [1]
    batch_size = 128
    num_workers = min(os.cpu_count(), 4) if os.cpu_count() else 2
    # --- End Configurable Parameters ---

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

    print(f"\n--- GradAscentDescent VAE Unlearning Configuration ---")
    print(f"Classes to Forget: {classes_to_forget}")
    print(f"KL Weight: {kl_weight_param}")
    print(f"Retain Loss Weight (alpha): {retain_loss_weight}")
    print(f"Forget Loss Weight (beta): {forget_loss_weight}")
    print(f"Unlearning LR: {unlearning_lr}, Epochs: {unlearning_epochs}, Patience: {unlearning_patience}")

    print("\n--- Data Preparation (MNIST) ---")
    transform_mnist = get_mnist_transform(image_size=hyperparams["image_size"])
    try:
        full_trainset_mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
        testset_mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
    except Exception as e:
        print(f"Error loading MNIST dataset: {e}"); exit(1)

    # 1. Create Validation Set from Full Training Data
    num_total_train = len(full_trainset_mnist)
    indices = list(range(num_total_train))
    np.random.shuffle(indices)
    split_idx = int(np.floor(val_split_fraction * num_total_train))
    val_indices, unlearn_train_indices = indices[:split_idx], indices[split_idx:]
    
    val_dataset_full = Subset(full_trainset_mnist, val_indices)
    unlearn_train_dataset_full = Subset(full_trainset_mnist, unlearn_train_indices)

    # 2. Split Validation Set into Retain/Forget
    val_forget_indices, val_retain_indices = split_indices(val_dataset_full, classes_to_forget)
    val_forget_dataset = Subset(val_dataset_full, val_forget_indices)
    val_retain_dataset = Subset(val_dataset_full, val_retain_indices)

    # 3. Split Unlearning Training Set into Retain/Forget (for the unlearning algorithm)
    train_forget_indices, train_retain_indices = split_indices(unlearn_train_dataset_full, classes_to_forget, retain_fraction=retain_split_fraction, forget_fraction=forget_split_fraction)
    train_forget_dataset_unlearn = Subset(unlearn_train_dataset_full, train_forget_indices)
    train_retain_dataset_unlearn = Subset(unlearn_train_dataset_full, train_retain_indices)
    
    # 4. Test set splitting for final evaluation
    test_forget_indices, test_retain_indices = split_indices(testset_mnist, classes_to_forget)
    test_forget_dataset_eval = Subset(testset_mnist, test_forget_indices)
    test_retain_dataset_eval = Subset(testset_mnist, test_retain_indices)

    print(f"Dataset Sizes:")
    print(f"  Unlearning Train: Retain={len(train_retain_dataset_unlearn)}, Forget={len(train_forget_dataset_unlearn)}")
    print(f"  Validation:       Retain={len(val_retain_dataset)}, Forget={len(val_forget_dataset)}")
    print(f"  Test Evaluation:  Retain={len(test_retain_dataset_eval)}, Forget={len(test_forget_dataset_eval)}")

    pin_memory = True if device.type == 'cuda' else False

    # DataLoaders for Unlearning
    train_retain_loader = DataLoader(train_retain_dataset_unlearn, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True) # drop_last=True for stable batch sizes
    train_forget_loader = DataLoader(train_forget_dataset_unlearn, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    
    # DataLoaders for Validation
    val_retain_loader = DataLoader(val_retain_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    val_forget_loader = DataLoader(val_forget_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # DataLoaders for Final Test Evaluation
    test_retain_loader_eval = DataLoader(test_retain_dataset_eval, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_forget_loader_eval = DataLoader(test_forget_dataset_eval, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    print("\n--- Original VAE Model Evaluation (Test Set) ---")
    original_model_test_retain_metrics = evaluate_vae(model, test_retain_loader_eval, device, kl_weight_param, "Original Test Retain")
    original_model_test_forget_metrics = evaluate_vae(model, test_forget_loader_eval, device, kl_weight_param, "Original Test Forget")
    print("Original Model Test Metrics:")
    print(f"  Retain -> Recon: {original_model_test_retain_metrics['recon_loss']:.2f}, KLD: {original_model_test_retain_metrics['kld']:.2f}, Total: {original_model_test_retain_metrics['total_loss']:.2f}")
    print(f"  Forget -> Recon: {original_model_test_forget_metrics['recon_loss']:.2f}, KLD: {original_model_test_forget_metrics['kld']:.2f}, Total: {original_model_test_forget_metrics['total_loss']:.2f}")
    generate_samples(model, LATENT_DIM, n_samples=25, device=device, filename="grad_ad_original_samples_32x32.png")

    print("\n--- GradAscentDescent VAE Unlearning ---")
    grad_ad_unlearner = GradAscentDescentVAE(
        kl_weight=kl_weight_param,
        retain_loss_weight=retain_loss_weight,
        forget_loss_weight=forget_loss_weight
    )
    import time
    start_time = time.time()
    
    unlearned_model_grad_ad = grad_ad_unlearner.unlearn(
        original_model=model,
        train_retain_loader=train_retain_loader,
        train_forget_loader=train_forget_loader,
        val_retain_loader=val_retain_loader,
        val_forget_loader=val_forget_loader,
        epochs=unlearning_epochs,
        lr=unlearning_lr,
        device=device,
        patience=unlearning_patience
    )
    end_time = time.time()
    print(f"GradAscentDescent Unlearning Time: {end_time - start_time:.2f}s")

    print("\n--- GradAscentDescent Unlearned VAE Model Evaluation (Test Set) ---")
    grad_ad_unlearned_test_retain_metrics = evaluate_vae(unlearned_model_grad_ad, test_retain_loader_eval, device, kl_weight_param, "GradAD Unlearned Test Retain")
    grad_ad_unlearned_test_forget_metrics = evaluate_vae(unlearned_model_grad_ad, test_forget_loader_eval, device, kl_weight_param, "GradAD Unlearned Test Forget")
    print("GradAscentDescent Unlearned Model Test Metrics:")
    print(f"  Retain -> Recon: {grad_ad_unlearned_test_retain_metrics['recon_loss']:.2f}, KLD: {grad_ad_unlearned_test_retain_metrics['kld']:.2f}, Total: {grad_ad_unlearned_test_retain_metrics['total_loss']:.2f}")
    print(f"  Forget -> Recon: {grad_ad_unlearned_test_forget_metrics['recon_loss']:.2f}, KLD: {grad_ad_unlearned_test_forget_metrics['kld']:.2f}, Total: {grad_ad_unlearned_test_forget_metrics['total_loss']:.2f}")
    generate_samples(unlearned_model_grad_ad, LATENT_DIM, n_samples=25, device=device, filename="grad_ad_unlearned_samples_32x32.png")

    print("\n--- VAE Performance Summary (Test Set Total Loss) ---")
    print(f"Original Model: Retain TotalLoss={original_model_test_retain_metrics['total_loss']:.2f}, Forget TotalLoss={original_model_test_forget_metrics['total_loss']:.2f}")
    print(f"GradAD Unlearned:  Retain TotalLoss={grad_ad_unlearned_test_retain_metrics['total_loss']:.2f}, Forget TotalLoss={grad_ad_unlearned_test_forget_metrics['total_loss']:.2f}")
    retain_loss_delta = grad_ad_unlearned_test_retain_metrics['total_loss'] - original_model_test_retain_metrics['total_loss']
    forget_loss_delta = grad_ad_unlearned_test_forget_metrics['total_loss'] - original_model_test_forget_metrics['total_loss']
    print(f"  GradAD Change:  Retain ΔTotalLoss={retain_loss_delta:+.2f}, Forget ΔTotalLoss={forget_loss_delta:+.2f}")
    print("\nDesired outcome: Forget Total Loss should increase significantly, Retain Total Loss should increase minimally or decrease.")

if __name__ == "__main__":
    # Ensure a pretrained model exists. If not, you might want to add a call
    # to a training function here, similar to how snop_vae.py handles it.
    # For this script, we assume 'vae_mnist_pretrained.pth' exists.
    model_path = 'vae_mnist_pretrained.pth'
    if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
        print(f"Error: Pretrained VAE model not found or is empty at '{model_path}'.")
        print("Please ensure a VAE model is trained and saved as 'vae_mnist_pretrained.pth'.")
        print("You can train one using the train_vae function in snop_vae.py, for example.")
        exit(1)
        
    run_grad_ad_unlearning()