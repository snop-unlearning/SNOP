import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
from tqdm import tqdm
import os
import time
import copy # Added for deepcopy
from torch.amp import GradScaler, autocast # For Mixed Precision

# Assuming resnet.py is in the same directory or accessible
from resnet import ResNet18ForCIFAR100, test_transform
from utils import set_seed, split_indices, evaluate, load_model_checkpoint

# --- Unlearning Function ---
def gradient_ascent_descent_unlearn(
    model,
    train_retain_loader,
    train_forget_loader,
    val_retain_loader,
    val_forget_loader,
    lr=1e-4,
    epochs=10,
    lambda_forget=1.0,
    patience=3,
    device='cuda',
    use_amp=True
):
    unlearned_model = copy.deepcopy(model).to(device)
    optimizer = optim.AdamW(unlearned_model.parameters(), lr=lr, weight_decay=1e-4)
    # Scheduler aims to minimize validation forget accuracy
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=max(1, patience // 2), verbose=True, threshold=1e-4)
    criterion = nn.CrossEntropyLoss()

    amp_enabled = use_amp and (device.type == 'cuda')
    scaler = GradScaler(enabled=amp_enabled)
    print(f"Gradient Ascent/Descent Unlearning | AMP: {'Enabled' if amp_enabled else 'Disabled'}")

    best_val_forget_acc = float('inf') # Lower is better for forget set accuracy
    best_val_retain_acc_at_best_forget = -float('inf') # Higher is better
    best_model_state = None
    patience_counter = 0
    best_epoch = -1

    if len(train_retain_loader.dataset) == 0:
        print("Warning: Train retain loader is empty. Unlearning will be ineffective. Returning original model.")
        return model
    if len(train_forget_loader.dataset) == 0:
        print("Warning: Train forget loader is empty. Effective lambda_forget will be 0. Proceeding with retain-only fine-tuning.")
        # lambda_forget will naturally have no effect if loss_forget is always 0

    for epoch in range(epochs):
        epoch_start_time = time.time()
        unlearned_model.train()
        total_epoch_loss = 0.0
        total_retain_loss_epoch = 0.0
        total_forget_loss_epoch = 0.0
        num_batches = 0

        iter_forget_loader = iter(train_forget_loader)

        with tqdm(train_retain_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False, dynamic_ncols=True) as pbar:
            for retain_inputs, retain_labels in pbar:
                retain_inputs, retain_labels = retain_inputs.to(device), retain_labels.to(device)

                forget_inputs, forget_labels = None, None
                if len(train_forget_loader.dataset) > 0: # Only try to get forget data if loader is not empty
                    try:
                        forget_inputs, forget_labels = next(iter_forget_loader)
                    except StopIteration:
                        iter_forget_loader = iter(train_forget_loader) # Reset iterator
                        forget_inputs, forget_labels = next(iter_forget_loader)
                    forget_inputs, forget_labels = forget_inputs.to(device), forget_labels.to(device)

                optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=amp_enabled, device_type='cuda'):
                    loss_retain = criterion(unlearned_model(retain_inputs), retain_labels.long())

                    loss_forget = torch.tensor(0.0, device=device)
                    if forget_inputs is not None and lambda_forget > 0:
                        loss_forget = criterion(unlearned_model(forget_inputs), forget_labels.long())
                    
                    combined_loss = loss_retain - (lambda_forget * loss_forget)
                
                scaler.scale(combined_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unlearned_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                total_epoch_loss += combined_loss.item()
                total_retain_loss_epoch += loss_retain.item()
                total_forget_loss_epoch += loss_forget.item()
                num_batches += 1
                if num_batches % 20 == 0:
                    pbar.set_postfix(loss=f"{combined_loss.item():.4f}", ret_L=f"{loss_retain.item():.2f}", fgt_L=f"{loss_forget.item():.2f}")
        
        avg_epoch_loss = total_epoch_loss / num_batches if num_batches > 0 else 0.0
        avg_retain_loss = total_retain_loss_epoch / num_batches if num_batches > 0 else 0.0
        avg_forget_loss = total_forget_loss_epoch / num_batches if num_batches > 0 else 0.0


        val_retain_acc = evaluate(unlearned_model, val_retain_loader, device, "Val Retain")
        val_forget_acc = evaluate(unlearned_model, val_forget_loader, device, "Val Forget")
        epoch_end_time = time.time()

        print(f"Epoch {epoch+1}/{epochs} - AvgLoss: {avg_epoch_loss:.4f} (R:{avg_retain_loss:.2f}|F:{avg_forget_loss:.2f}), "
              f"Val RetainAcc: {val_retain_acc:.4f}, Val ForgetAcc: {val_forget_acc:.6f} "
              f"(Time: {epoch_end_time - epoch_start_time:.2f}s, LR: {optimizer.param_groups[0]['lr']:.2e})")

        scheduler.step(val_forget_acc)

        if val_forget_acc < best_val_forget_acc - 1e-6: # Significantly better forget acc
            print(f"-> New best val forget acc: {val_forget_acc:.6f} (was {best_val_forget_acc:.6f}). Retain acc: {val_retain_acc:.4f}")
            best_val_forget_acc = val_forget_acc
            best_val_retain_acc_at_best_forget = val_retain_acc
            best_model_state = copy.deepcopy(unlearned_model.state_dict())
            best_epoch = epoch + 1
            patience_counter = 0
        elif abs(val_forget_acc - best_val_forget_acc) < 1e-6: # Forget acc is similar
            if val_retain_acc > best_val_retain_acc_at_best_forget + 1e-5: # Retain acc improved
                print(f"  -> Val forget acc tied ({val_forget_acc:.6f}), val retain acc improved: {val_retain_acc:.4f} (was {best_val_retain_acc_at_best_forget:.4f})")
                best_val_retain_acc_at_best_forget = val_retain_acc
                best_model_state = copy.deepcopy(unlearned_model.state_dict())
                best_epoch = epoch + 1
                # Don't reset patience here, primary goal is forget acc
            else:
                patience_counter += 1
                print(f"  -> No improvement by criterion. Patience: {patience_counter}/{patience}")
        else: # Forget acc worsened
            patience_counter += 1
            print(f"  -> Val forget acc worse ({val_forget_acc:.6f} > {best_val_forget_acc:.6f}). Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"Early stopping triggered after epoch {epoch+1}.")
            break
            
    if best_model_state:
        print(f"Loading best model from epoch {best_epoch} (Val Forget Acc: {best_val_forget_acc:.6f}, Val Retain Acc: {best_val_retain_acc_at_best_forget:.4f})")
        unlearned_model.load_state_dict(best_model_state)
    else:
        print("No improvement found or patience not met. Using model from the last epoch.")

    return unlearned_model

# --- Main Function ---
def main():
    set_seed(42) # Default seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    use_amp_main = True if device.type == 'cuda' else False

    # --- Model Loading ---
    model_path_final = 'resnet18_cifar100_final.pth'
    model_path_best = 'resnet18_cifar100_best.pth'
    model = load_model_checkpoint(ResNet18ForCIFAR100, model_path_best, model_path_final, device)
    # The utility function load_model_checkpoint prints warnings if no model is loaded.

    # --- Configuration ---
    classes_to_forget = [54] 
    val_split_fraction = 0.1 
    train_retain_fraction = 1
    train_forget_fraction = 1
    
    lr_unlearn = 1e-5
    epochs_unlearn = 20 
    lambda_forget_unlearn = 1 # Critical hyperparameter
    patience_unlearn = 7       # Increased patience

    print(f"\n--- Configuration ---")
    print(f"Classes to Forget: {classes_to_forget}")
    print(f"Validation Split Fraction for Unlearning: {val_split_fraction:.2f}")
    print(f"Unlearning LR: {lr_unlearn}, Epochs: {epochs_unlearn}, Lambda_Forget: {lambda_forget_unlearn}, Patience: {patience_unlearn}")
    print(f"AMP Enabled: {use_amp_main}")

    # --- Data Loading and Splitting ---
    print("\n--- Data Preparation ---")
    try:
        full_trainset_orig = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=test_transform)
        testset_orig = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    except Exception as e:
        print(f"Error loading CIFAR-100: {e}"); exit(1)

    num_total_train = len(full_trainset_orig)
    indices = list(range(num_total_train))
    np.random.shuffle(indices)
    split_idx = int(np.floor(val_split_fraction * num_total_train))
    
    unlearn_val_indices = indices[:split_idx]
    unlearn_train_indices = indices[split_idx:]

    # Datasets for unlearning loop's validation
    val_dataset_unlearn_loop = Subset(full_trainset_orig, unlearn_val_indices)
    # Dataset pool for unlearning loop's training
    train_pool_unlearn_loop = Subset(full_trainset_orig, unlearn_train_indices)

    val_forget_idx_local, val_retain_idx_local = split_indices(val_dataset_unlearn_loop, classes_to_forget)
    val_forget_dataset_unlearn = Subset(val_dataset_unlearn_loop, val_forget_idx_local)
    val_retain_dataset_unlearn = Subset(val_dataset_unlearn_loop, val_retain_idx_local)
    
    train_forget_idx_local, train_retain_idx_local = split_indices(train_pool_unlearn_loop, classes_to_forget, retain_fraction=train_retain_fraction, forget_fraction=train_forget_fraction)
    train_forget_dataset_unlearn = Subset(train_pool_unlearn_loop, train_forget_idx_local)
    train_retain_dataset_unlearn = Subset(train_pool_unlearn_loop, train_retain_idx_local)

    # Datasets for final testing
    test_forget_indices, test_retain_indices = split_indices(testset_orig, classes_to_forget)
    test_forget_dataset_final = Subset(testset_orig, test_forget_indices)
    test_retain_dataset_final = Subset(testset_orig, test_retain_indices)

    print(f"Sizes: FullTrainOrig={num_total_train}, UnlearnValSet={len(val_dataset_unlearn_loop)}, UnlearnTrainPool={len(train_pool_unlearn_loop)}")
    print(f"       UnlearnVal(R/F)={len(val_retain_dataset_unlearn)}/{len(val_forget_dataset_unlearn)}")
    print(f"       UnlearnTrain(R/F)={len(train_retain_dataset_unlearn)}/{len(train_forget_dataset_unlearn)}")
    print(f"       Test(R/F)={len(test_retain_dataset_final)}/{len(test_forget_dataset_final)}")

    # --- Create DataLoaders ---
    batch_size = 64 
    num_workers = min(os.cpu_count(), 4) if os.cpu_count() else 2 # Safer num_workers
    pin_memory = True if device.type == 'cuda' else False

    train_retain_loader = DataLoader(train_retain_dataset_unlearn, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    train_forget_loader = DataLoader(train_forget_dataset_unlearn, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=False) # drop_last=True if it is very small
    
    val_retain_loader = DataLoader(val_retain_dataset_unlearn, batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    val_forget_loader = DataLoader(val_forget_dataset_unlearn, batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    test_retain_loader = DataLoader(test_retain_dataset_final, batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_forget_loader = DataLoader(test_forget_dataset_final, batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # --- Evaluate Original Model on Test Data ---
    print("\n--- Original Model Evaluation (Test Set) ---")
    original_retain_acc = evaluate(model, test_retain_loader, device, "Original Test Retain")
    original_forget_acc = evaluate(model, test_forget_loader, device, "Original Test Forget")
    print(f"Original Model -> Test Retain Acc: {original_retain_acc:.4f}, Test Forget Acc: {original_forget_acc:.4f}")

    # --- Perform Unlearning ---
    print("\n--- Gradient Ascent/Descent Unlearning ---")
    start_time = time.time()
    unlearned_model = gradient_ascent_descent_unlearn(
        model=model,
        train_retain_loader=train_retain_loader,
        train_forget_loader=train_forget_loader,
        val_retain_loader=val_retain_loader,
        val_forget_loader=val_forget_loader,
        lr=lr_unlearn,
        epochs=epochs_unlearn,
        lambda_forget=lambda_forget_unlearn,
        patience=patience_unlearn,
        device=device,
        use_amp=use_amp_main
    )
    unlearning_duration = time.time() - start_time
    print(f"Unlearning finished in {unlearning_duration:.2f}s")
    torch.save(unlearned_model.state_dict(), 'resnet18_cifar100_grad_acdc.pth')

    # --- Evaluate Unlearned Model on Test Data ---
    print("\n--- Unlearned Model Evaluation (Test Set) ---")
    unlearned_retain_acc = evaluate(unlearned_model, test_retain_loader, device, "Unlearned Test Retain")
    unlearned_forget_acc = evaluate(unlearned_model, test_forget_loader, device, "Unlearned Test Forget")
    print(f"Unlearned Model -> Test Retain Acc: {unlearned_retain_acc:.4f}, Test Forget Acc: {unlearned_forget_acc:.4f}")

    # --- Performance Summary ---
    print("\n--- Performance Summary (Test Set) ---")
    print(f"Original Model: Retain Acc={original_retain_acc:.4f}, Forget Acc={original_forget_acc:.4f}")
    print(f"Unlearned Model: Retain Acc={unlearned_retain_acc:.4f}, Forget Acc={unlearned_forget_acc:.4f}")
    
    retain_delta = unlearned_retain_acc - original_retain_acc
    forget_delta = unlearned_forget_acc - original_forget_acc
    print(f"Change: Retain Δ={retain_delta:+.4f}, Forget Δ={forget_delta:+.4f}")

    if original_retain_acc > 1e-5: # Avoid division by zero
        print(f"Retain Performance Ratio: {unlearned_retain_acc / original_retain_acc:.4f}")
    if original_forget_acc > 1e-5:
        forgetting_rate = max(0.0, (original_forget_acc - unlearned_forget_acc) / original_forget_acc)
        print(f"Forgetting Rate: {forgetting_rate:.4f}")

if __name__ == "__main__":
    try:
        if 'ResNet18ForCIFAR100' not in globals(): raise ImportError("ResNet not defined")
    except ImportError as e:
        print(f"Missing dependency: {e}. Ensure resnet.py is available.")
        exit(1)
    
    model_path_final = 'resnet18_cifar100_final.pth'
    model_path_best = 'resnet18_cifar100_best.pth'
    if not os.path.exists(model_path_final) and not os.path.exists(model_path_best):
         print(f"Warning: Model weights not found at '{model_path_final}' or '{model_path_best}'. The script will run with initial model weights if these files are missing, and results may not be meaningful for unlearning.")

    main()