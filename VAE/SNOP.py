import torch
import torchvision
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
from SNOP_vae_methodology import SNOP_VAE

from vae_utils import (
    set_seed,
    get_mnist_transform,
    split_indices,
    hyperparams,
    evaluate_vae,
    generate_samples,
    load_vae_model,
    IMAGE_SIZE
)

LATENT_DIM = hyperparams["latent_dim"]
IMAGE_CHANNELS = hyperparams["image_channels"]
INIT_CHANNELS = hyperparams["init_channels"]
# IMAGE_SIZE is also available from hyperparams in vae_utils


# --- Main Execution Logic ---
def main():
    set_seed(42) # Set seed at the beginning
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    use_amp_main = False # SNOP_VAE in snop_vae_methodology.py handles its own AMP if needed

    # --- VAE Model Loading ---
    latent_dim_main = LATENT_DIM
    init_channels_main = INIT_CHANNELS
    image_channels_main = IMAGE_CHANNELS
    kl_weight_main = 1.0 # Used for evaluation and potentially by SNOP_VAE if it uses vae_loss_function
    image_size_main = IMAGE_SIZE

    model_path = 'vae_mnist_pretrained.pth'
    model = load_vae_model(
        model_path,
        device,
        latent_dim=latent_dim_main,
        image_channels=image_channels_main,
        init_channels=init_channels_main,
        image_size=image_size_main
    )
    # load_vae_model prints messages about loading status.
    # The original script had commented out exits if model not found.
    # If 'vae_mnist_pretrained.pth' is essential, an explicit check after load_vae_model might be needed
    # if load_vae_model doesn't already enforce this (it currently prints a warning and returns a new model).

    # --- Configuration ---
    classes_to_forget = [5]
    val_split_fraction = 0.1 
    train_retain_sampling_fraction = 1 
    train_forget_sampling_fraction = 1 
    use_full_data_for_initial_signals = False # can use if not using full data for optimization

    dynamic_mask_update_freq_main = 1
    post_tune_epochs_main = 2
    post_tune_lr_main = 1e-5

    # --- Hyperparameters for SNOP ---
    alpha_main = 20.0      # Weight for retain loss (Full VAE Loss)
    beta_main = 10.0       # Weight for forget sparsity 
    gamma_main = 0.005   # Weight for retain stability - keep very low
    delta_main = 0.4   # Weight for forget loss
    dampening_main = 0.9
    lr_main = 5e-4        # SNOP LR might need to be smaller for fine-tuning
    epochs_main = 2
    patience_main = 7 
    
    # kl_weight_main is defined above (e.g., 1.0)

    print(f"\n--- VAE Unlearning Configuration ---")
    print(f"Using New VAE Architecture: Latent Dim={latent_dim_main}, Init Channels={init_channels_main}")
    print(f"Classes to Forget: {classes_to_forget}")
    print(f"Validation Split Fraction: {val_split_fraction:.2f}")
    print(f"Unlearning Optimization Data Sampling: Retain={train_retain_sampling_fraction:.2f}, Forget={train_forget_sampling_fraction:.2f}")
    print(f"Use Full Data for Initial Signals: {use_full_data_for_initial_signals}")
    print(f"AMP Enabled: {use_amp_main}")
    print(f"Dynamic Mask Update Freq: {dynamic_mask_update_freq_main}")
    print(f"Post-Unlearning Tune Epochs: {post_tune_epochs_main} (LR: {post_tune_lr_main})")
    print(f"Hyperparameters: alpha={alpha_main}, beta={beta_main}, gamma={gamma_main}, damp={dampening_main}, lr={lr_main}, kl_w={kl_weight_main}")

    # --- Data Loading and Splitting ---
    print("\n--- Data Preparation (MNIST) ---")
    transform_mnist = get_mnist_transform(image_size=image_size_main)

    try:
        full_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
    except Exception as e:
        print(f"Error loading MNIST dataset: {e}")
        exit(1)

    # 1. Create Dedicated Validation Set
    num_train = len(full_trainset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(val_split_fraction * num_train))
    val_indices, train_subset_indices = indices[:split], indices[split:]
    val_dataset = Subset(full_trainset, val_indices)
    train_subset_for_unlearning = Subset(full_trainset, train_subset_indices)

    # 2. Split Validation Set into Retain/Forget
    val_forget_indices, val_retain_indices = split_indices(val_dataset, classes_to_forget)
    val_forget_dataset = Subset(val_dataset, val_forget_indices)
    val_retain_dataset = Subset(val_dataset, val_retain_indices)

    # 3. Split Training Subset Pool into Full Retain/Forget
    full_train_forget_indices, full_train_retain_indices = split_indices(train_subset_for_unlearning, classes_to_forget)
    full_train_forget_dataset = Subset(train_subset_for_unlearning, full_train_forget_indices)
    full_train_retain_dataset = Subset(train_subset_for_unlearning, full_train_retain_indices)

    # 4. Subsample for Optimization Loop
    if len(full_train_retain_indices) > 0:
        num_retain_opt = max(1, int(len(full_train_retain_indices) * train_retain_sampling_fraction))
        opt_retain_indices = np.random.choice(full_train_retain_indices, num_retain_opt, replace=False).tolist()
    else: opt_retain_indices = []
    if len(full_train_forget_indices) > 0:
        num_forget_opt = max(1, int(len(full_train_forget_indices) * train_forget_sampling_fraction))
        opt_forget_indices = np.random.choice(full_train_forget_indices, num_forget_opt, replace=False).tolist()
    else: opt_forget_indices = []

    train_retain_dataset_opt = Subset(train_subset_for_unlearning, opt_retain_indices)
    train_forget_dataset_opt = Subset(train_subset_for_unlearning, opt_forget_indices)

    # 5. Prepare Test Set Splits
    test_forget_indices, test_retain_indices = split_indices(testset, classes_to_forget)
    test_forget_dataset = Subset(testset, test_forget_indices)
    test_retain_dataset = Subset(testset, test_retain_indices)

    print(f"Sizes: FullTrain={num_train}, Val={len(val_dataset)}, TrainSub={len(train_subset_for_unlearning)}")
    print(f"       Val(R/F)={len(val_retain_dataset)}/{len(val_forget_dataset)}")
    print(f"       FullTrainSub(R/F)={len(full_train_retain_dataset)}/{len(full_train_forget_dataset)}")
    print(f"       OptLoop(R/F)={len(train_retain_dataset_opt)}/{len(train_forget_dataset_opt)}")
    print(f"       Test(R/F)={len(test_retain_dataset)}/{len(test_forget_dataset)}")

    # --- Create DataLoaders ---
    batch_size = 128 # Can use larger batch for MNIST
    num_workers = 2
    pin_memory = True if device.type == 'cuda' else False

    # Optimization Loaders
    train_forget_loader_opt = DataLoader(train_forget_dataset_opt, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    train_retain_loader_opt = DataLoader(train_retain_dataset_opt, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    # Validation Loaders
    val_forget_loader = DataLoader(val_forget_dataset, batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    val_retain_loader = DataLoader(val_retain_dataset, batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    # Initial Signal Loaders (Optional: Full or Sampled)
    initial_forget_loader = DataLoader(full_train_forget_dataset if use_full_data_for_initial_signals else train_forget_dataset_opt, batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    initial_retain_loader = DataLoader(full_train_retain_dataset if use_full_data_for_initial_signals else train_retain_dataset_opt, batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    if use_full_data_for_initial_signals: print("Using full train subset data for initial Fisher/Mask calculation.")
    else: print("Using sampled train subset data for initial Fisher/Mask calculation.")
    # Test Loaders
    test_forget_loader = DataLoader(test_forget_dataset, batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_retain_loader = DataLoader(test_retain_dataset, batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # --- Evaluate Original Model ---
    print("\n--- Original VAE Model Evaluation (Test Set) ---")
    original_retain_metrics = evaluate_vae(model, test_retain_loader, device, kl_weight_main, "Original Test Retain")
    original_forget_metrics = evaluate_vae(model, test_forget_loader, device, kl_weight_main, "Original Test Forget")
    print("Original Model Metrics:")
    print(f"  Retain -> Recon: {original_retain_metrics['recon_loss']:.2f}, KLD: {original_retain_metrics['kld']:.2f}")
    print(f"  Forget -> Recon: {original_forget_metrics['recon_loss']:.2f}, KLD: {original_forget_metrics['kld']:.2f}")

    # --- Initialize and Run SNOP ---
    print("\n--- SNOP VAE Unlearning ---")
    snop_vae = SNOP_VAE(model, alpha=alpha_main, beta=beta_main, gamma=gamma_main, delta=delta_main,
                        dampening_constant=dampening_main, kl_weight=kl_weight_main)

    import time
    start_time = time.time()
    
    unlearned_model = snop_vae.unlearn(
        train_retain_loader=train_retain_loader_opt,
        train_forget_loader=train_forget_loader_opt,
        val_retain_loader=val_retain_loader,
        val_forget_loader=val_forget_loader,
        lr=lr_main, epochs=epochs_main, patience=patience_main, device=device,
        kl_weight=kl_weight_main, # Pass KL weight
        initial_forget_loader=initial_forget_loader,
        initial_retain_loader=initial_retain_loader,
        threshold_mode='stddev', percentile=95, # Adjust percentile maybe
        dynamic_mask_update_freq=dynamic_mask_update_freq_main,
        use_amp=use_amp_main,
        post_tune_epochs=post_tune_epochs_main,
        post_tune_lr=post_tune_lr_main
    )
    print(f"Unlearning took {time.time() - start_time:.2f}s")
    torch.save(unlearned_model.state_dict(), 'unlearned_vae_snop.pth')
    # --- Evaluate Unlearned Model ---
    print("\n--- Unlearned VAE Model Evaluation (Test Set) ---")
    unlearned_retain_metrics = evaluate_vae(unlearned_model, test_retain_loader, device, kl_weight_main, "Unlearned Test Retain")
    unlearned_forget_metrics = evaluate_vae(unlearned_model, test_forget_loader, device, kl_weight_main, "Unlearned Test Forget")
    print("Unlearned Model Metrics:")
    print(f"  Retain -> Recon: {unlearned_retain_metrics['recon_loss']:.2f}, KLD: {unlearned_retain_metrics['kld']:.2f}")
    print(f"  Forget -> Recon: {unlearned_forget_metrics['recon_loss']:.2f}, KLD: {unlearned_forget_metrics['kld']:.2f}")

    # --- Performance Summary ---
    print("\n--- VAE Performance Summary ---")
    print(f"Original Model: Retain Recon={original_retain_metrics['recon_loss']:.2f}, Forget Recon={original_forget_metrics['recon_loss']:.2f}")
    print(f"Unlearned Model: Retain Recon={unlearned_retain_metrics['recon_loss']:.2f}, Forget Recon={unlearned_forget_metrics['recon_loss']:.2f}")
    retain_recon_delta = unlearned_retain_metrics['recon_loss'] - original_retain_metrics['recon_loss']
    forget_recon_delta = unlearned_forget_metrics['recon_loss'] - original_forget_metrics['recon_loss']
    print(f"Change: Retain Recon Δ={retain_recon_delta:+.2f}, Forget Recon Δ={forget_recon_delta:+.2f}")
    # Higher forget reconstruction loss is desired (poorer reconstruction of forgotten class)
    # Lower retain reconstruction loss is desired (good reconstruction of retained classes)


    # --- Optional: Mechanistic Validation ---
    print("\n--- Mechanistic Validation (Test Set, VAE) ---")
    try:
        # Recalculate mask based on test forget data using RECON loss only
        validation_mask = snop_vae.identify_critical_circuits(test_forget_loader, device, threshold_mode='percentile', percentile=90)
        validation_results = snop_vae.validate_unlearning(
            unlearned_model=unlearned_model,
            forget_loader=test_forget_loader,
            retain_loader=test_retain_loader,
            critical_mask=validation_mask,
            device=device
        )
        print("\nValidation Results:")
        print(f"  Unlearned Forget Recon: {validation_results['unlearned_forget_recon_loss']:.2f}")
        print(f"  Ablated Forget Recon:   {validation_results['ablated_forget_recon_loss']:.2f}")
        print(f"  Unlearned Retain Recon: {validation_results['unlearned_retain_recon_loss']:.2f}")
        print(f"  Ablated Retain Recon:   {validation_results['ablated_retain_recon_loss']:.2f}")
        print(f"  Forget Recon Effectiveness (vs Ablation): {validation_results['forgetting_effectiveness_recon']:.4f}")
        print(f"  Retain Recon Preservation (vs Ablation): {validation_results['retain_preservation_recon']:.4f}")
    except Exception as e:
        print(f"Could not perform mechanistic validation: {e}")

    # --- Optional: Generate Samples ---
    print("\n--- Generating Samples ---")
    # Make sure to pass the correct latent_dim to generate_samples
    if device.type == 'cuda': # Ensure generation happens on CPU if CUDA unavailable during generation
        try:
            # The model instance here is 'unlearned_model' or 'model' (original)
            # Ensure generate_samples uses the correct latent_dim (latent_dim_main)
            generate_samples(unlearned_model, latent_dim_main, n_samples=25, device=device, filename="unlearned_samples_32x32.png", image_size=32)
            generate_samples(model, latent_dim_main, n_samples=25, device=device, filename="original_samples_32x32.png", image_size=32)
            print("Generated sample images saved.")
        except Exception as e:
            print(f"Could not generate samples: {e}")


if __name__ == "__main__":
    # train_vae(epochs=100,
    #           lr=0.001,
    #           latent_dim=LATENT_DIM,
    #           init_channels=INIT_CHANNELS,
    #           batch_size=128,
    #           save_path='vae_mnist_pretrained.pth')
    
    print("\nINITIAL VAE TRAINING COMPLETE. NOW RUNNING UNLEARNING SCRIPT.\n")
    
    model_path = 'vae_mnist_pretrained.pth'
    if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
        print(f"Error: Pretrained VAE model not found or is empty at '{model_path}'.")
        print("Please ensure 'train_vae' runs successfully and saves the model.")
        exit(1)

    main() # Ensure kl_weight_main in main() is 1.0