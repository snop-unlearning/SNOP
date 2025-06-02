import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision
from torchvision.transforms import transforms
import os
import numpy as np
import random # Added for set_seed

# --- Seeding ---
def set_seed(seed_value=42):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        # Optional: for full reproducibility, but can impact performance
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed_value}")

hyperparams = {
    "latent_dim": 16,
    "image_channels": 1, # MNIST is grayscale
    "init_channels": 8,
    "image_size": 32 # Target image size for MNIST
}

LATENT_DIM = hyperparams["latent_dim"]
IMAGE_CHANNELS = hyperparams["image_channels"]
INIT_CHANNELS = hyperparams["init_channels"]
IMAGE_SIZE = hyperparams["image_size"]


# --- Data Helpers ---
def get_mnist_transform(image_size=IMAGE_SIZE):
    """Returns the standard transformation for MNIST dataset."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

def get_dataset_targets(dataset_obj: Dataset) -> list:
    """
    Gets targets from a PyTorch Dataset or Subset object.
    Handles nested Subsets by traversing to the base dataset.
    """
    if isinstance(dataset_obj, Subset):
        if not dataset_obj.indices: # Check if indices list is empty
            return []
        
        current_dataset = dataset_obj.dataset
        # Start with the indices of the current Subset
        effective_indices = list(dataset_obj.indices) 

        # Traverse up if the underlying dataset is also a Subset
        while isinstance(current_dataset, Subset):
            if not current_dataset.indices: # Parent subset has no indices
                return []
            
            parent_indices_list = list(current_dataset.indices)
            # Map current effective_indices to the indices of the parent Subset
            new_effective_indices = []
            for idx_in_current_subset in effective_indices:
                if 0 <= idx_in_current_subset < len(parent_indices_list):
                    new_effective_indices.append(parent_indices_list[idx_in_current_subset])
            
            effective_indices = new_effective_indices
            if not effective_indices: # If mapping results in no valid indices
                return []
            
            current_dataset = current_dataset.dataset # Move to the parent's dataset

        # Now current_dataset is the base Dataset object, 
        # and effective_indices are the actual indices for this base dataset.
        if hasattr(current_dataset, 'targets'):
            targets_attr = current_dataset.targets
            # Filter effective_indices to be valid for the base dataset's targets
            valid_base_indices = [i for i in effective_indices if 0 <= i < len(targets_attr)]
            if not valid_base_indices:
                return []

            if isinstance(targets_attr, torch.Tensor):
                return targets_attr[torch.tensor(valid_base_indices, dtype=torch.long)].tolist()
            elif isinstance(targets_attr, np.ndarray):
                return targets_attr[np.array(valid_base_indices, dtype=np.int64)].tolist()
            elif isinstance(targets_attr, list):
                # Ensure all indices are valid for the list
                return [targets_attr[i] for i in valid_base_indices]
        
        # Fallback: iterate through the (now base) dataset with the effective_indices
        try:
            # Ensure indices are valid for the length of the base dataset
            valid_base_indices_for_iteration = [i for i in effective_indices if 0 <= i < len(current_dataset)]
            if not valid_base_indices_for_iteration: return []
            return [current_dataset[i][1] for i in valid_base_indices_for_iteration] # Assumes (data, label)
        except Exception:
            # print(f"Warning: Fallback iteration for base dataset targets failed.")
            return []

    elif hasattr(dataset_obj, 'targets'): # Direct dataset object
        targets_attr = dataset_obj.targets
        if isinstance(targets_attr, torch.Tensor):
            return targets_attr.tolist()
        elif isinstance(targets_attr, np.ndarray):
            return targets_attr.tolist()
        elif isinstance(targets_attr, list):
            return list(targets_attr)

    # Final fallback: iterate through the dataset_obj itself
    try:
        if len(dataset_obj) == 0: return []
        return [label for _, label in dataset_obj] # Assumes (data, label)
    except Exception:
        # print("Warning: Could not retrieve targets from dataset using final fallback. Returning empty list.")
        return []

def split_indices(dataset, classes_to_forget, retain_fraction=1.0, forget_fraction=1.0):
    """Splits dataset indices into forget and retain sets."""
    labels = np.array(get_dataset_targets(dataset))
    if labels.ndim == 0 or len(labels) == 0:
        return [], []
    forget_mask = np.isin(labels, list(classes_to_forget))
    retain_mask = ~forget_mask
    
    forget_indices_all = np.where(forget_mask)[0].tolist()
    retain_indices_all = np.where(retain_mask)[0].tolist()
    
    sample_forget_size = int(forget_fraction * len(forget_indices_all))
    sample_retain_size = int(retain_fraction * len(retain_indices_all))

    # Ensure sample size is not greater than available indices and not negative
    sample_forget_size = max(0, min(sample_forget_size, len(forget_indices_all)))
    sample_retain_size = max(0, min(sample_retain_size, len(retain_indices_all)))
    
    forget_indices_sampled = []
    if sample_forget_size > 0 and len(forget_indices_all) > 0:
        forget_indices_sampled = np.random.choice(forget_indices_all, size=sample_forget_size, replace=False).tolist()
    
    retain_indices_sampled = []
    if sample_retain_size > 0 and len(retain_indices_all) > 0:
        retain_indices_sampled = np.random.choice(retain_indices_all, size=sample_retain_size, replace=False).tolist()
        
    # if not forget_indices_sampled and not retain_indices_sampled and (len(forget_indices_all) > 0 or len(retain_indices_all) > 0) :
    #     print("Warning: No indices selected for forget or retain sets, though original sets were non-empty.")
        
    return forget_indices_sampled, retain_indices_sampled

# --- VAE Model Definition ---
class ConvVAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, image_channels=IMAGE_CHANNELS, init_channels=INIT_CHANNELS, image_size=IMAGE_SIZE):
        super(ConvVAE, self).__init__()
        self.latent_dim = latent_dim
        self.image_channels = image_channels
        self.init_channels = init_channels
        self.image_size = image_size # Store image size
        kernel_size = 4
        
        # Calculate the size of the feature map after the final encoder layer
        # This depends on the number of downsampling layers and kernel/stride/padding
        # For 32x32 input, 4 conv layers with stride 2, padding 1 (except last with padding 0)
        # 32 -> 16 (enc1) -> 8 (enc2) -> 4 (enc3) -> 1 (enc4, kernel 4, stride 2, padding 0)
        # So, final feature map size is 1x1 if input is 32x32.
        # If image_size changes, this needs to be robust.
        # Let's assume a fixed number of downsampling layers (4 in this case)
        # final_feature_map_size = image_size // (2**4) # This is too simple if padding/kernel varies
        # For the current architecture (fixed for 32x32 input leading to 1x1 feature map before FC):
        self.final_enc_fm_size = 1 # Feature map spatial dimension after enc4
        self.fc_input_dim = 64 * self.final_enc_fm_size * self.final_enc_fm_size


        # encoder
        self.enc1 = nn.Conv2d(
            in_channels=self.image_channels, out_channels=self.init_channels, kernel_size=kernel_size,
            stride=2, padding=1
        ) # Assuming 32x32 -> 16x16
        self.enc2 = nn.Conv2d(
            in_channels=self.init_channels, out_channels=self.init_channels*2, kernel_size=kernel_size,
            stride=2, padding=1
        ) # 16x16 -> 8x8
        self.enc3 = nn.Conv2d(
            in_channels=self.init_channels*2, out_channels=self.init_channels*4, kernel_size=kernel_size,
            stride=2, padding=1
        ) # 8x8 -> 4x4
        self.enc4 = nn.Conv2d(
            in_channels=self.init_channels*4, out_channels=64, kernel_size=kernel_size, # Output channels for fc_input_dim
            stride=2, padding=0 # This makes it 1x1 for a 4x4 input
        ) # 4x4 -> 1x1

        self.fc1 = nn.Linear(self.fc_input_dim, 128) # Adjusted based on fc_input_dim
        self.fc_mu = nn.Linear(128, self.latent_dim)
        self.fc_log_var = nn.Linear(128, self.latent_dim)
        self.fc2 = nn.Linear(self.latent_dim, self.fc_input_dim) # Output for decoder

        # decoder
        self.dec1 = nn.ConvTranspose2d(
            in_channels=64, out_channels=self.init_channels*8, kernel_size=kernel_size, # from fc_input_dim (e.g. 64)
            stride=1, padding=0
        ) # 1x1 -> 4x4
        self.dec2 = nn.ConvTranspose2d(
            in_channels=self.init_channels*8, out_channels=self.init_channels*4, kernel_size=kernel_size,
            stride=2, padding=1
        ) # 4x4 -> 8x8
        self.dec3 = nn.ConvTranspose2d(
            in_channels=self.init_channels*4, out_channels=self.init_channels*2, kernel_size=kernel_size,
            stride=2, padding=1
        ) # 8x8 -> 16x16
        self.dec4 = nn.ConvTranspose2d(
            in_channels=self.init_channels*2, out_channels=self.image_channels, kernel_size=kernel_size,
            stride=2, padding=1
        ) # 16x16 -> 32x32 (or original image_size)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + (eps * std)

    def decode(self, z):
        # z_transformed = self.fc2(z) # fc2 maps latent_dim to fc_input_dim (e.g., 64)
        # z_reshaped = z_transformed.view(-1, 64, self.final_enc_fm_size, self.final_enc_fm_size) # Reshape to match dec1 input
        h = self.fc2(z).view(-1, 64, self.final_enc_fm_size, self.final_enc_fm_size)
        h = F.relu(self.dec1(h))
        h = F.relu(self.dec2(h))
        h = F.relu(self.dec3(h))
        return torch.sigmoid(self.dec4(h)) # Sigmoid for image pixel values [0,1]

    def forward(self, x):
        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        h = F.relu(self.enc3(h))
        h = F.relu(self.enc4(h))
        # Adaptive avg pool ensures that even if image_size changes slightly,
        # this part will try to produce a fixed size output before flattening,
        # but fc_input_dim must be correct.
        # For this architecture, enc4 output is already 64x1x1.
        h_flat = h.view(h.shape[0], -1) # Flatten after enc4
        
        hidden = F.relu(self.fc1(h_flat)) # Added ReLU here for consistency
        mu = self.fc_mu(hidden)
        logvar = self.fc_log_var(hidden)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

# --- VAE Loss Function ---
def vae_loss_function(recon_x, x, mu, logvar, kl_weight=1.0): # Added kl_weight parameter
    """
    Calculates the VAE loss.
    recon_x: Reconstructed input.
    x: Original input.
    mu: Latent mean.
    logvar: Latent log variance.
    kl_weight: Weight for the KL divergence term.
    Returns summed BCE and summed KLD for the batch.
    """
    # Ensure x and recon_x are flattened for BCE calculation
    BCE = F.binary_cross_entropy(recon_x.reshape(recon_x.size(0), -1),
                                 x.reshape(x.size(0), -1),
                                 reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE, kl_weight * KLD # Apply kl_weight to KLD term

# --- Model Loading ---
def load_vae_model(model_path, device, latent_dim=LATENT_DIM, image_channels=IMAGE_CHANNELS, init_channels=INIT_CHANNELS, image_size=IMAGE_SIZE):
    """Loads a VAE model from a checkpoint."""
    model = ConvVAE(latent_dim=latent_dim, image_channels=image_channels, init_channels=init_channels, image_size=image_size)
    if model_path and os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        print(f"Loading VAE model from {model_path}")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("VAE model loaded successfully.")
        except Exception as e:
            print(f"Error loading VAE model from {model_path}: {e}. Using initial weights.")
    else:
        print(f"Warning: VAE model checkpoint not found at '{model_path}' or is empty. Using initial model weights.")
    model.to(device)
    return model

# --- VAE Evaluation ---
def evaluate_vae(model, dataloader, device, kl_weight_eval=1.0, desc="Evaluating VAE"):
    """
    Evaluates VAE performance.
    Returns a dictionary with 'recon_loss', 'kld', and 'total_loss' (all per sample).
    """
    model.eval()
    total_recon_loss_sum = 0.0 # Sum of BCE losses over all samples
    total_kld_sum = 0.0        # Sum of KLD losses over all samples
    total_samples_processed = 0

    if not hasattr(dataloader, 'dataset') or len(dataloader.dataset) == 0:
        # print(f"Warning: {desc} - Dataloader's dataset is empty. Returning zero losses.")
        return {'recon_loss': 0.0, 'kld': 0.0, 'total_loss': 0.0}

    with torch.no_grad():
        with tqdm(dataloader, desc=desc, leave=False, dynamic_ncols=True) as pbar:
            for batch_data in pbar:
                inputs = batch_data[0] if isinstance(batch_data, (list, tuple)) else batch_data
                if inputs.numel() == 0: continue

                inputs = inputs.to(device)
                recon_batch, mu, logvar = model(inputs)
                
                # vae_loss_function returns summed BCE and KLD for the batch
                bce_batch_sum, kld_batch_sum_unweighted = vae_loss_function(recon_batch, inputs, mu, logvar, kl_weight=1.0) # Get unweighted KLD sum
                
                total_recon_loss_sum += bce_batch_sum.item()
                total_kld_sum += kld_batch_sum_unweighted.item() # Accumulate unweighted KLD sum
                current_batch_size = inputs.size(0)
                total_samples_processed += current_batch_size

                if total_samples_processed > 0:
                    pbar.set_postfix(
                        avg_recon_sample=f"{(total_recon_loss_sum / total_samples_processed):.4f}",
                        avg_kld_sample=f"{(total_kld_sum / total_samples_processed):.4f}"
                    )

    if total_samples_processed == 0:
        avg_recon_per_sample = 0.0
        avg_kld_per_sample = 0.0
    else:
        avg_recon_per_sample = total_recon_loss_sum / total_samples_processed
        avg_kld_per_sample = total_kld_sum / total_samples_processed

    # Apply kl_weight_eval for the final reported total loss
    avg_total_loss_per_sample = avg_recon_per_sample + kl_weight_eval * avg_kld_per_sample

    return {
        'recon_loss': avg_recon_per_sample, # Per sample
        'kld': avg_kld_per_sample,          # Per sample (unweighted by kl_weight_eval here, but weighted for total)
        'total_loss': avg_total_loss_per_sample # Per sample
    }

# --- Function to Generate Samples ---
def generate_samples(model, latent_dim, n_samples=25, device='cuda', filename="generated_samples.png", image_size=IMAGE_SIZE):
    """Generates samples from the VAE and saves them to a file."""
    model.eval()
    with torch.no_grad():
        # Ensure z is created on the same device as the model's parameters
        model_device = next(model.parameters()).device
        z = torch.randn(n_samples, latent_dim).to(model_device)
        generated_images = model.decode(z).cpu() 

    # Ensure images are in [0, 1] range for saving, sigmoid in decode should handle this.
    # Clamping can be added if necessary: generated_images.clamp_(0, 1)
    grid = torchvision.utils.make_grid(generated_images.view(n_samples, IMAGE_CHANNELS, image_size, image_size), 
                                       nrow=int(n_samples**0.5)) # Make a square grid
    torchvision.utils.save_image(grid, filename)
    # print(f"Generated samples saved to {filename}") # Can be verbose

# --- VAE Training Function (Example, can be expanded) ---
def train_vae(epochs=100,
              lr=1e-3,
              latent_dim=LATENT_DIM,
              init_channels=INIT_CHANNELS,
              image_channels=IMAGE_CHANNELS,
              image_size_train=IMAGE_SIZE,
              batch_size=128,
              save_path='vae_mnist_pretrained.pth',
              kl_weight_train=1.0, # KL weight during training
              device_str=None,
              classes_to_exclude=None # List of classes to exclude from training
              ):
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"\n--- Training VAE on {device} ---")
    print(f"Epochs: {epochs}, LR: {lr}, Latent Dim: {latent_dim}, Init Channels: {init_channels}, Image Size: {image_size_train}")
    print(f"KL Weight (Training): {kl_weight_train}")
    if classes_to_exclude:
        print(f"Excluding classes from training: {classes_to_exclude}")

    transform = get_mnist_transform(image_size=image_size_train)
    
    train_dataset_full = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    val_dataset_full = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    if classes_to_exclude:
        # Filter training set
        train_indices_to_keep = [i for i, target in enumerate(train_dataset_full.targets) if target not in classes_to_exclude]
        train_dataset_filtered = Subset(train_dataset_full, train_indices_to_keep)
        print(f"Original training set size: {len(train_dataset_full)}, Filtered training set size: {len(train_dataset_filtered)}")
        # Filter validation set
        val_indices_to_keep = [i for i, target in enumerate(val_dataset_full.targets) if target not in classes_to_exclude]
        val_dataset_filtered = Subset(val_dataset_full, val_indices_to_keep)
        print(f"Original validation set size: {len(val_dataset_full)}, Filtered validation set size: {len(val_dataset_filtered)}")
    else:
        train_dataset_filtered = train_dataset_full
        val_dataset_filtered = val_dataset_full
        print(f"Training on full dataset (no classes excluded). Train size: {len(train_dataset_filtered)}, Val size: {len(val_dataset_filtered)}")

    if len(train_dataset_filtered) == 0:
        print("Error: Training dataset is empty after filtering. Cannot train.")
        return None

    train_loader = DataLoader(train_dataset_filtered, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_dataset_filtered, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    model = ConvVAE(latent_dim=latent_dim, 
                    image_channels=image_channels, 
                    init_channels=init_channels,
                    image_size=image_size_train).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss_per_sample = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss_sum_samples = 0.0
        train_recon_sum_samples = 0.0
        train_kld_sum_samples = 0.0
        num_train_samples_epoch = 0

        with tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{epochs}", leave=False) as pbar_train:
            for inputs, _ in pbar_train:
                inputs = inputs.to(device)
                optimizer.zero_grad()
                recon_batch, mu, logvar = model(inputs)
                
                bce_batch_sum, kld_batch_sum_weighted = vae_loss_function(recon_batch, inputs, mu, logvar, kl_weight=kl_weight_train)
                total_loss_batch_sum = bce_batch_sum + kld_batch_sum_weighted
                
                total_loss_batch_sum.backward()
                optimizer.step()

                train_loss_sum_samples += total_loss_batch_sum.item()
                train_recon_sum_samples += bce_batch_sum.item() # Unweighted BCE sum
                train_kld_sum_samples += (kld_batch_sum_weighted.item() / kl_weight_train if kl_weight_train > 0 else kld_batch_sum_weighted.item()) # Store unweighted KLD sum for consistent reporting
                num_train_samples_epoch += inputs.size(0)

                if num_train_samples_epoch > 0:
                    pbar_train.set_postfix(
                        avg_loss_sample=f"{(train_loss_sum_samples / num_train_samples_epoch):.4f}"
                    )
        
        avg_train_loss_sample = train_loss_sum_samples / num_train_samples_epoch if num_train_samples_epoch else 0
        avg_train_recon_sample = train_recon_sum_samples / num_train_samples_epoch if num_train_samples_epoch else 0
        avg_train_kld_sample = train_kld_sum_samples / num_train_samples_epoch if num_train_samples_epoch else 0
        print(f"Train Epoch {epoch+1}/{epochs} - Avg Loss/Sample: {avg_train_loss_sample:.4f} (Recon: {avg_train_recon_sample:.4f}, KLD: {avg_train_kld_sample:.4f})")

        # Validation using evaluate_vae (which returns per-sample losses)
        val_metrics = evaluate_vae(model, val_loader, device, kl_weight_eval=kl_weight_train, desc=f"Valid Epoch {epoch+1}/{epochs}")
        current_val_total_loss_sample = val_metrics['total_loss']
        print(f"Valid Epoch {epoch+1}/{epochs} - Avg Loss/Sample: {current_val_total_loss_sample:.4f} (Recon: {val_metrics['recon_loss']:.4f}, KLD: {val_metrics['kld']:.4f})")

        if current_val_total_loss_sample < best_val_loss_per_sample:
            best_val_loss_per_sample = current_val_total_loss_sample
            print(f"  New best validation loss/sample: {best_val_loss_per_sample:.4f}. Saving model to {save_path}")
            torch.save(model.state_dict(), save_path)
    
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        print(f"Loading best model from {save_path} (Val Loss/Sample: {best_val_loss_per_sample:.4f})")
        model.load_state_dict(torch.load(save_path, map_location=device))
    else:
        print(f"Warning: No model was saved or model file is empty at {save_path}.")

    print("--- VAE Training Complete ---")
    if device.type == 'cuda' or not torch.cuda.is_available():
        try:
            generate_samples(model, latent_dim, n_samples=25, device=device, 
                             filename=f"{os.path.splitext(save_path)[0]}_samples.png", 
                             image_size=image_size_train)
            print("Generated samples from trained VAE saved.")
        except Exception as e:
            print(f"Could not generate samples from trained VAE: {e}")
    return model


if __name__ == "__main__":
    # Example usage
    set_seed(42)  # Set seed for reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Train the VAE model
    trained_vae_model = train_vae(
        epochs=10,
        lr=1e-3,
        latent_dim=LATENT_DIM,
        init_channels=INIT_CHANNELS,
        image_channels=IMAGE_CHANNELS,
        image_size_train=IMAGE_SIZE,
        batch_size=128,
        save_path='vae_mnist_pretrained.pth',
        kl_weight_train=1.0,
        device_str=device.type
    )
    
    if trained_vae_model:
        print("VAE model training completed successfully.")
    else:
        print("VAE model training failed.")