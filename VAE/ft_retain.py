import torch
import copy
import torchvision
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import os

from .vae_utils import (
    set_seed,
    get_mnist_transform,
    split_indices,
    ConvVAE,
    vae_loss_function,
    hyperparams,
    evaluate_vae,
    generate_samples,
    load_vae_model,
)

# --- Hyperparameters ---
# These are now primarily sourced from vae_utils.hyperparams, but can be overridden if needed
LATENT_DIM = hyperparams["latent_dim"]
IMAGE_CHANNELS = hyperparams["image_channels"]
INIT_CHANNELS = hyperparams["init_channels"]
# IMAGE_SIZE is also available from hyperparams in vae_utils

# --- Main Execution Logic for Retain Fine-tuning ---
def run_retain_finetuning():
    set_seed(42) # Set seed at the beginning
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Configuration ---
    model_path = 'vae_mnist_pretrained.pth'
    classes_to_forget = [1] # Example: forget class '1'
    kl_weight_train_eval = 1.0 # Weight for KLD in VAE loss, consistent for training and eval

    lr_phase1 = 1e-4
    epochs_phase1 = 20
    patience_phase1 = 5

    lr_phase2 = 5e-5
    epochs_phase2 = 30
    patience_phase2 = 7

    val_split_fraction = 0.1 # Fraction of full_trainset to use for validation
    
    # % of dataset to use for retain 
    retain_split_fraction = 1
    forget_split_fraction = 1
    
    batch_size = 128
    num_workers = min(os.cpu_count(), 4) if os.cpu_count() else 2
    pin_memory = True if device.type == 'cuda' else False

    print(f"\n--- Retain Fine-tuning VAE Configuration ---")
    print(f"Classes to Forget: {classes_to_forget}")
    print(f"KL Divergence Weight (Train & Eval): {kl_weight_train_eval}")
    print(f"Phase 1: LR={lr_phase1}, Epochs={epochs_phase1}, Patience={patience_phase1} (Maximize Forget Loss)")
    print(f"Phase 2: LR={lr_phase2}, Epochs={epochs_phase2}, Patience={patience_phase2} (Minimize Retain Loss)")
    print(f"Validation Split Fraction: {val_split_fraction}")

    # --- Load Original Model ---
    if not (os.path.exists(model_path) and os.path.getsize(model_path) > 0):
        print(f"Error: Pretrained VAE model not found or is empty at '{model_path}'. Exiting."); exit(1)
    
    original_model = load_vae_model(
        model_path,
        device,
        latent_dim=LATENT_DIM,
        image_channels=IMAGE_CHANNELS,
        init_channels=INIT_CHANNELS,
        image_size=hyperparams["image_size"] # Use image_size from hyperparams
    )
    # load_vae_model already prints loading messages and handles .to(device)

    # --- Data Preparation (MNIST) ---
    print("\n--- Data Preparation (MNIST) ---")
    transform_mnist = get_mnist_transform(image_size=hyperparams["image_size"])
    try:
        full_trainset_mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
        testset_mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
    except Exception as e:
        print(f"Error loading MNIST dataset: {e}"); exit(1)

    # 1. Create Validation Set from Full Train Set
    num_total_train = len(full_trainset_mnist)
    indices = list(range(num_total_train))
    np.random.shuffle(indices)
    val_set_size = int(np.floor(val_split_fraction * num_total_train))
    
    val_indices = indices[:val_set_size]
    train_pool_indices = indices[val_set_size:]

    val_dataset = Subset(full_trainset_mnist, val_indices)
    train_pool_dataset = Subset(full_trainset_mnist, train_pool_indices)

    # 2. Split Training Pool for Fine-tuning (Optimizer sees only train_retain_finetune_dataset)
    _, train_pool_retain_indices = split_indices(train_pool_dataset, classes_to_forget, retain_fraction=retain_split_fraction, forget_fraction=forget_split_fraction)
    train_retain_finetune_dataset = Subset(train_pool_dataset, train_pool_retain_indices)

    # 3. Split Validation Set for Phase 1 & 2 Evaluation
    val_forget_indices, val_retain_indices = split_indices(val_dataset, classes_to_forget)
    val_forget_eval_dataset = Subset(val_dataset, val_forget_indices)
    val_retain_eval_dataset = Subset(val_dataset, val_retain_indices)

    # 4. Split Test Set for Final Evaluation
    test_forget_indices, test_retain_indices = split_indices(testset_mnist, classes_to_forget)
    test_forget_final_dataset = Subset(testset_mnist, test_forget_indices)
    test_retain_final_dataset = Subset(testset_mnist, test_retain_indices)

    print(f"Dataset sizes:")
    print(f"  Full Train: {len(full_trainset_mnist)}")
    print(f"  Train Pool (for optimizer source): {len(train_pool_dataset)}")
    print(f"    -> Train Retain (for fine-tuning): {len(train_retain_finetune_dataset)}")
    print(f"  Validation Set (for phase stopping): {len(val_dataset)}")
    print(f"    -> Val Forget (Phase 1 target): {len(val_forget_eval_dataset)}")
    print(f"    -> Val Retain (Phase 2 target): {len(val_retain_eval_dataset)}")
    print(f"  Test Set (for final eval): {len(testset_mnist)}")
    print(f"    -> Test Forget: {len(test_forget_final_dataset)}")
    print(f"    -> Test Retain: {len(test_retain_final_dataset)}")

    # DataLoaders
    train_retain_finetune_loader = DataLoader(train_retain_finetune_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    val_forget_eval_loader = DataLoader(val_forget_eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    val_retain_eval_loader = DataLoader(val_retain_eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_forget_final_loader = DataLoader(test_forget_final_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_retain_final_loader = DataLoader(test_retain_final_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # --- Evaluate Original Model on Test Set ---
    print("\n--- Original VAE Model Evaluation (Test Set) ---")
    original_model_test_retain_metrics = evaluate_vae(original_model, test_retain_final_loader, device, kl_weight_train_eval, "Original Test Retain")
    original_model_test_forget_metrics = evaluate_vae(original_model, test_forget_final_loader, device, kl_weight_train_eval, "Original Test Forget")
    print("Original Model Test Metrics:")
    print(f"  Retain -> Recon: {original_model_test_retain_metrics['recon_loss']:.2f}, KLD: {original_model_test_retain_metrics['kld']:.2f}")
    print(f"  Forget -> Recon: {original_model_test_forget_metrics['recon_loss']:.2f}, KLD: {original_model_test_forget_metrics['kld']:.2f}")
    generate_samples(original_model, LATENT_DIM, n_samples=25, device=device, filename="original_samples_for_finetune_32x32.png")

    # --- Phase 1: Maximize Forget Reconstruction Loss ---
    print("\n--- Phase 1: Fine-tuning to Maximize Forget Reconstruction Loss (on Val Set) ---")
    model_phase1 = copy.deepcopy(original_model)
    optimizer_phase1 = torch.optim.Adam(model_phase1.parameters(), lr=lr_phase1)
    
    best_max_val_forget_loss = -float('inf')
    val_retain_loss_at_max_forget = float('inf')
    model_state_after_phase1 = None
    patience_counter_ph1 = 0
    best_epoch_ph1 = -1

    for epoch in range(epochs_phase1):
        model_phase1.train()
        epoch_train_loss_sum = 0.0
        num_train_batches = 0
        for inputs, _ in tqdm(train_retain_finetune_loader, desc=f"Phase1 Epoch {epoch+1}/{epochs_phase1} Training", leave=False):
            if inputs.numel() == 0: continue
            inputs = inputs.to(device)
            optimizer_phase1.zero_grad()
            recon_batch, mu, logvar = model_phase1(inputs)
            bce, kld = vae_loss_function(recon_batch, inputs, mu, logvar)
            # Loss is per-sample average for stability
            loss = (bce / inputs.size(0)) + kl_weight_train_eval * (kld / inputs.size(0))
            loss.backward()
            optimizer_phase1.step()
            epoch_train_loss_sum += loss.item() * inputs.size(0) # To get sum for averaging later
            num_train_batches += inputs.size(0)

        avg_epoch_train_loss = epoch_train_loss_sum / num_train_batches if num_train_batches > 0 else 0

        # Evaluate on validation sets
        val_retain_metrics_ph1 = evaluate_vae(model_phase1, val_retain_eval_loader, device, kl_weight_train_eval, "Phase1 Val Retain")
        val_forget_metrics_ph1 = evaluate_vae(model_phase1, val_forget_eval_loader, device, kl_weight_train_eval, "Phase1 Val Forget")
        
        current_val_forget_recon_loss = val_forget_metrics_ph1['recon_loss']
        current_val_retain_recon_loss = val_retain_metrics_ph1['recon_loss']

        print(f"Phase1 Epoch {epoch+1}: Train Loss={avg_epoch_train_loss:.2f}, Val Retain Recon={current_val_retain_recon_loss:.2f}, Val Forget Recon={current_val_forget_recon_loss:.2f}")

        if current_val_forget_recon_loss > best_max_val_forget_loss:
            best_max_val_forget_loss = current_val_forget_recon_loss
            val_retain_loss_at_max_forget = current_val_retain_recon_loss
            model_state_after_phase1 = copy.deepcopy(model_phase1.state_dict())
            patience_counter_ph1 = 0
            best_epoch_ph1 = epoch + 1
            print(f"  -> New best max forget recon loss on val: {best_max_val_forget_loss:.2f} (Retain recon: {val_retain_loss_at_max_forget:.2f})")
        else:
            patience_counter_ph1 += 1
            print(f"  -> No improvement in val forget recon loss. Patience: {patience_counter_ph1}/{patience_phase1}")

        if patience_counter_ph1 >= patience_phase1:
            print(f"Early stopping Phase 1 after epoch {epoch+1}.")
            break
    
    if model_state_after_phase1 is None: # Should not happen if epochs > 0
        print("Warning: Phase 1 did not find a best model state. Using last state of original model.")
        model_state_after_phase1 = original_model.state_dict()
        best_max_val_forget_loss = original_model_test_forget_metrics['recon_loss'] # Approx.
        val_retain_loss_at_max_forget = original_model_test_retain_metrics['recon_loss'] # Approx.
    
    print(f"--- Phase 1 Complete. Best Model (Epoch {best_epoch_ph1}): Max Val Forget Recon={best_max_val_forget_loss:.2f}, Val Retain Recon={val_retain_loss_at_max_forget:.2f} ---")


    # --- Phase 2: Minimize Retain Reconstruction Loss ---
    print("\n--- Phase 2: Fine-tuning to Minimize Retain Reconstruction Loss (on Val Set) ---")
    model_phase2 = ConvVAE(latent_dim=LATENT_DIM, image_channels=IMAGE_CHANNELS, init_channels=INIT_CHANNELS).to(device)
    model_phase2.load_state_dict(model_state_after_phase1) # Start from best model of Phase 1
    optimizer_phase2 = torch.optim.Adam(model_phase2.parameters(), lr=lr_phase2)

    best_min_val_retain_loss = float('inf')
    val_forget_loss_at_best_retain = -float('inf')
    final_finetuned_model_state = None
    patience_counter_ph2 = 0
    best_epoch_ph2 = -1

    for epoch in range(epochs_phase2):
        model_phase2.train()
        epoch_train_loss_sum_ph2 = 0.0
        num_train_batches_ph2 = 0
        for inputs, _ in tqdm(train_retain_finetune_loader, desc=f"Phase2 Epoch {epoch+1}/{epochs_phase2} Training", leave=False):
            if inputs.numel() == 0: continue
            inputs = inputs.to(device)
            optimizer_phase2.zero_grad()
            recon_batch, mu, logvar = model_phase2(inputs)
            bce, kld = vae_loss_function(recon_batch, inputs, mu, logvar)
            loss = (bce / inputs.size(0)) + kl_weight_train_eval * (kld / inputs.size(0))
            loss.backward()
            optimizer_phase2.step()
            epoch_train_loss_sum_ph2 += loss.item() * inputs.size(0)
            num_train_batches_ph2 += inputs.size(0)
        
        avg_epoch_train_loss_ph2 = epoch_train_loss_sum_ph2 / num_train_batches_ph2 if num_train_batches_ph2 > 0 else 0

        # Evaluate on validation sets
        val_retain_metrics_ph2 = evaluate_vae(model_phase2, val_retain_eval_loader, device, kl_weight_train_eval, "Phase2 Val Retain")
        val_forget_metrics_ph2 = evaluate_vae(model_phase2, val_forget_eval_loader, device, kl_weight_train_eval, "Phase2 Val Forget")

        current_val_retain_recon_loss_ph2 = val_retain_metrics_ph2['recon_loss']
        current_val_forget_recon_loss_ph2 = val_forget_metrics_ph2['recon_loss']
        
        print(f"Phase2 Epoch {epoch+1}: Train Loss={avg_epoch_train_loss_ph2:.2f}, Val Retain Recon={current_val_retain_recon_loss_ph2:.2f}, Val Forget Recon={current_val_forget_recon_loss_ph2:.2f}")

        if current_val_retain_recon_loss_ph2 < best_min_val_retain_loss:
            best_min_val_retain_loss = current_val_retain_recon_loss_ph2
            val_forget_loss_at_best_retain = current_val_forget_recon_loss_ph2
            final_finetuned_model_state = copy.deepcopy(model_phase2.state_dict())
            patience_counter_ph2 = 0
            best_epoch_ph2 = epoch + 1
            print(f"  -> New best min retain recon loss on val: {best_min_val_retain_loss:.2f} (Forget recon: {val_forget_loss_at_best_retain:.2f})")
        else:
            patience_counter_ph2 += 1
            print(f"  -> No improvement in val retain recon loss. Patience: {patience_counter_ph2}/{patience_phase2}")

        if patience_counter_ph2 >= patience_phase2:
            print(f"Early stopping Phase 2 after epoch {epoch+1}.")
            break
    
    if final_finetuned_model_state is None: # If Phase 2 didn't improve
        print("Warning: Phase 2 did not find a better model state than end of Phase 1. Using model from end of Phase 1.")
        final_finetuned_model_state = model_state_after_phase1
        best_min_val_retain_loss = val_retain_loss_at_max_forget # from Phase 1
        val_forget_loss_at_best_retain = best_max_val_forget_loss # from Phase 1

    print(f"--- Phase 2 Complete. Best Model (Epoch {best_epoch_ph2}): Min Val Retain Recon={best_min_val_retain_loss:.2f}, Val Forget Recon={val_forget_loss_at_best_retain:.2f} ---")

    # --- Load Final Fine-tuned Model and Evaluate on Test Set ---
    final_finetuned_model = ConvVAE(latent_dim=LATENT_DIM, image_channels=IMAGE_CHANNELS, init_channels=INIT_CHANNELS).to(device)
    final_finetuned_model.load_state_dict(final_finetuned_model_state)
    torch.save(final_finetuned_model.state_dict(), "unlearned_vae_ft_r.pth") # Save the final model state

    print("\n--- Final Fine-tuned VAE Model Evaluation (Test Set) ---")
    finetuned_model_test_retain_metrics = evaluate_vae(final_finetuned_model, test_retain_final_loader, device, kl_weight_train_eval, "Final Test Retain")
    finetuned_model_test_forget_metrics = evaluate_vae(final_finetuned_model, test_forget_final_loader, device, kl_weight_train_eval, "Final Test Forget")
    print("Final Fine-tuned Model Test Metrics:")
    print(f"  Retain -> Recon: {finetuned_model_test_retain_metrics['recon_loss']:.2f}, KLD: {finetuned_model_test_retain_metrics['kld']:.2f}")
    print(f"  Forget -> Recon: {finetuned_model_test_forget_metrics['recon_loss']:.2f}, KLD: {finetuned_model_test_forget_metrics['kld']:.2f}")
    generate_samples(final_finetuned_model, LATENT_DIM, n_samples=25, device=device, filename="retain_finetuned_samples_32x32.png")

    # --- Summary ---
    print("\n--- VAE Performance Summary (Test Set Recon Loss) ---")
    print(f"Original Model:           Retain Recon={original_model_test_retain_metrics['recon_loss']:.2f}, Forget Recon={original_model_test_forget_metrics['recon_loss']:.2f}")
    print(f"Retain Fine-tuned Model:  Retain Recon={finetuned_model_test_retain_metrics['recon_loss']:.2f}, Forget Recon={finetuned_model_test_forget_metrics['recon_loss']:.2f}")
    
    retain_recon_delta = finetuned_model_test_retain_metrics['recon_loss'] - original_model_test_retain_metrics['recon_loss']
    forget_recon_delta = finetuned_model_test_forget_metrics['recon_loss'] - original_model_test_forget_metrics['recon_loss']
    print(f"  Change vs Original:     Retain Δ={retain_recon_delta:+.2f}, Forget Δ={forget_recon_delta:+.2f}")

    print("\nIntermediate Validation Metrics (during fine-tuning):")
    print(f"End of Phase 1 (Epoch {best_epoch_ph1}): Max Val Forget Recon={best_max_val_forget_loss:.2f}, Val Retain Recon={val_retain_loss_at_max_forget:.2f}")
    print(f"End of Phase 2 (Epoch {best_epoch_ph2}): Min Val Retain Recon={best_min_val_retain_loss:.2f}, Val Forget Recon={val_forget_loss_at_best_retain:.2f}")
    print("\nDesired outcome: Forget Recon Loss should increase significantly, Retain Recon Loss should ideally stay low or decrease.")

if __name__ == "__main__":
    run_retain_finetuning()