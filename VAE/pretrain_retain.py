import torch
import torchvision
from torch.utils.data import DataLoader, Subset, Dataset
import os
import time

from vae_utils import (
    set_seed,
    get_mnist_transform,
    split_indices,
    hyperparams,
    evaluate_vae,
    generate_samples,
    train_vae,
    IMAGE_SIZE
)

# --- Hyperparameters ---

LATENT_DIM = hyperparams['latent_dim']
IMAGE_CHANNELS = hyperparams['image_channels']
INIT_CHANNELS = hyperparams['init_channels']
# IMAGE_SIZE is also available from hyperparams in vae_utils


# --- Main Execution Logic ---
def main():
    set_seed(42) # Set seed at the beginning
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    classes_to_forget = [1]  # Define which class(es) to exclude from training
    kl_weight_for_eval = 1.0 # For final evaluation reporting
    model_save_path = f'vae_mnist_retain_only_forgot_{"_".join(map(str, classes_to_forget))}.pth'

    # --- Train VAE only on Retain Data using the utility function ---
    epochs_train = 100 # Adjust as needed
    lr_train = 0.001
    batch_size_train = 128
    
    start_time = time.time()
    trained_model = train_vae( # Call the utility function
        epochs=epochs_train,
        lr=lr_train,
        latent_dim=LATENT_DIM,
        init_channels=INIT_CHANNELS,
        image_channels=IMAGE_CHANNELS,
        image_size_train=IMAGE_SIZE, # Use IMAGE_SIZE from vae_utils
        batch_size=batch_size_train,
        save_path=model_save_path,
        kl_weight_train=1.0, # KLD weight during training
        device_str=device.type,
        classes_to_exclude=classes_to_forget # Pass classes_to_forget as classes_to_exclude
    )
    print(f"Training took {time.time() - start_time:.2f} seconds.")
    
    if trained_model is None:
        print("Model training failed. Exiting.")
        exit(1)
    
    # --- Evaluation of the "Retain-Only" Trained Model ---
    print(f"\n--- Evaluating Model Trained Only on Retain Data ({model_save_path}) ---")
    
    transform_mnist_eval = get_mnist_transform(image_size=IMAGE_SIZE)
    try:
        testset_mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist_eval)
    except Exception as e:
        print(f"Error loading MNIST test dataset for evaluation: {e}"); exit(1)

    # Split test set for evaluation
    test_forget_indices, test_retain_indices = split_indices(testset_mnist, classes_to_forget)
    
    if not test_retain_indices: print("Warning: Test retain set is empty for evaluation.")
    if not test_forget_indices: print("Warning: Test forget set is empty for evaluation.")

    test_forget_dataset = Subset(testset_mnist, test_forget_indices)
    test_retain_dataset = Subset(testset_mnist, test_retain_indices)
    
    print(f"Sizes for Test Evaluation: TestRetain={len(test_retain_dataset)}, TestForget={len(test_forget_dataset)}")

    eval_batch_size = 256
    num_workers_eval = min(os.cpu_count(), 4) if os.cpu_count() else 2
    pin_memory_eval = True if device.type == 'cuda' else False
    
    test_retain_loader_eval = DataLoader(test_retain_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers_eval, pin_memory=pin_memory_eval)
    test_forget_loader_eval = DataLoader(test_forget_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers_eval, pin_memory=pin_memory_eval)

    # Evaluate
    retain_only_model_test_retain_metrics = evaluate_vae(trained_model, test_retain_loader_eval, device, kl_weight_for_eval, "Retain-Only Model on Test Retain")
    retain_only_model_test_forget_metrics = evaluate_vae(trained_model, test_forget_loader_eval, device, kl_weight_for_eval, "Retain-Only Model on Test Forget")
    
    print("\n--- Retain-Only Trained Model Performance (Test Set) ---")
    print(f"  Retain Set -> Recon: {retain_only_model_test_retain_metrics['recon_loss']:.2f}, KLD: {retain_only_model_test_retain_metrics['kld']:.2f}, Total: {retain_only_model_test_retain_metrics['total_loss']:.2f}")
    print(f"  Forget Set -> Recon: {retain_only_model_test_forget_metrics['recon_loss']:.2f}, KLD: {retain_only_model_test_forget_metrics['kld']:.2f}, Total: {retain_only_model_test_forget_metrics['total_loss']:.2f}")
    
    print("\nNote: For a model trained only on retain data, we expect:")
    print("  - Good (low) reconstruction loss on the Test Retain set.")
    print("  - Potentially higher reconstruction loss on the Test Forget set, as it hasn't seen these classes.")

    # --- Generate Samples ---
    generate_samples(trained_model, LATENT_DIM, n_samples=25, device=device, 
                     filename=f"retain_only_trained_samples_forgot_{'_'.join(map(str, classes_to_forget))}_{IMAGE_SIZE}x{IMAGE_SIZE}.png", 
                     image_size=IMAGE_SIZE)

if __name__ == "__main__":
    main()