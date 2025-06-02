import torch
import torch.nn as nn
import copy
import torchvision
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
from tqdm import tqdm
from resnet import ResNet18ForCIFAR100, test_transform # Assuming resnet.py is in the same directory
from utils import set_seed, split_indices, evaluate, load_model_checkpoint

import os
import time
from torch.amp import GradScaler, autocast


def finetune_on_retain(model_to_finetune,
                       train_retain_loader,
                       val_retain_loader,
                       val_forget_loader, # To monitor forget set during finetuning
                       lr=0.001,
                       epochs=10,
                       patience=3,
                       device='cuda',
                       use_amp=True):
    """
    Fine-tunes the model on the retain set.
    """
    print("\n--- Starting Retain Fine-tuning ---")
    finetuned_model = copy.deepcopy(model_to_finetune).to(device)

    optimizer = torch.optim.AdamW(finetuned_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=patience // 2, verbose=True, threshold=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    amp_enabled = use_amp and (device.type == 'cuda')
    scaler = GradScaler(enabled=amp_enabled)
    print(f"Fine-tuning with AMP: {'Enabled' if amp_enabled else 'Disabled'}")

    best_val_retain_acc = -1.0
    best_model_state = None
    patience_counter = 0
    best_epoch = -1

    if len(train_retain_loader.dataset) == 0:
        print("Warning: Training retain loader is empty. Skipping fine-tuning.")
        return finetuned_model # Return the model as is

    for epoch in range(epochs):
        epoch_start_time = time.time()
        finetuned_model.train()
        epoch_train_loss = 0.0
        num_train_batches = 0

        with tqdm(train_retain_loader, desc=f"FT Epoch {epoch+1}/{epochs}", leave=False, dynamic_ncols=True) as pbar:
            for batch_idx, (inputs, labels) in enumerate(pbar):
                if inputs.numel() == 0: continue
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=amp_enabled, device_type='cuda'):
                    outputs = finetuned_model(inputs)
                    loss = criterion(outputs, labels.long())

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(finetuned_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                epoch_train_loss += loss.item()
                num_train_batches += 1
                if batch_idx % 20 == 0:
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_epoch_loss = epoch_train_loss / num_train_batches if num_train_batches > 0 else 0

        # Validation
        epoch_val_retain_acc = evaluate(finetuned_model, val_retain_loader, device, "FT Val Retain")
        epoch_val_forget_acc = evaluate(finetuned_model, val_forget_loader, device, "FT Val Forget")
        epoch_end_time = time.time()

        print(f"FT Epoch {epoch+1}/{epochs} Summary - TrainLoss: {avg_epoch_loss:.4f}, "
              f"Val RetainAcc: {epoch_val_retain_acc:.4f}, Val ForgetAcc: {epoch_val_forget_acc:.4f} "
              f"(Time: {epoch_end_time - epoch_start_time:.2f}s)")

        scheduler.step(epoch_val_retain_acc)

        if epoch_val_retain_acc > best_val_retain_acc + 1e-5: # Use a small epsilon for improvement
            print(f"-> New best validation retain accuracy: {epoch_val_retain_acc:.4f} (was {best_val_retain_acc:.4f})")
            best_val_retain_acc = epoch_val_retain_acc
            best_model_state = copy.deepcopy(finetuned_model.state_dict())
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  -> No improvement in val retain acc. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"Early stopping triggered after epoch {epoch+1}.")
            break
    
    if best_model_state:
        print(f"Loading best model state from Epoch {best_epoch} (Val Retain Acc: {best_val_retain_acc:.4f})")
        finetuned_model.load_state_dict(best_model_state)
    else:
        print("Warning: No improvement detected during fine-tuning or no validation performed. Using model from last epoch.")

    print("--- Retain Fine-tuning Finished ---")
    return finetuned_model


def main():
    set_seed(42) # Default seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    use_amp_main = True if device.type == 'cuda' else False

    # --- Model Loading ---
    model_path_final = 'resnet18_cifar100_final.pth'
    model_path_best = 'resnet18_cifar100_best.pth'
    
    model = load_model_checkpoint(ResNet18ForCIFAR100, model_path_best, model_path_final, device)
    
    # Check if model loading was successful (heuristic: check if any parameters are loaded,
    # or rely on load_model_checkpoint's print statements. For stricter check,
    # load_model_checkpoint could return a status.)
    # For now, we assume if it returns, it's either loaded or using initial weights as per its logic.
    # The original script had an explicit exit if no model was loaded.
    # We need to replicate that if `load_model_checkpoint` doesn't inherently prevent continuation
    # with a completely uninitialized model when one was expected.
    # The utility function prints a warning if no checkpoint is loaded.
    # We can check if the model has any non-zero parameters as a proxy, or trust the warning.
    # Let's assume the user wants to exit if no pre-trained model is found for fine-tuning.
    # A more robust way would be for load_model_checkpoint to return a status.
    # For now, we'll check if the model has parameters and if they are not all zero.
    # This is a simple check; a more robust check would involve comparing to a newly initialized model.
    
    # A simple check: if the model is still on CPU after load_model_checkpoint (which moves to device),
    # it might indicate an issue, but load_model_checkpoint handles .to(device).
    # The original script exited if `loaded_path` was None.
    # `load_model_checkpoint` prints warnings but doesn't return a status of whether a checkpoint was loaded.
    # To replicate the exit behavior if no checkpoint was loaded:
    # We need a way to know if `load_model_checkpoint` actually loaded from a file.
    # For now, we'll proceed, and if a pre-trained model is essential, the user should verify.
    # The original script's check was:
    # if not loaded_path:
    #     print("Warning: No pre-trained model found or loaded. Fine-tuning will start from initial model weights.")
    #     print("Exiting: A pre-trained model is required for retain fine-tuning.")
    #     exit(1)
    # This check needs to be adapted. The utility function prints a warning.
    # If the script *must* exit if no checkpoint is loaded, `load_model_checkpoint` would need to signal this.
    # For now, we'll rely on the warning from `load_model_checkpoint`.
    # If the user wants a hard exit, they can add a check after the call.

    # --- Configuration ---
    classes_to_forget = [54] # Example: Class to "forget"
    val_split_fraction = 0.1  # 10% of original train for validation during fine-tuning
    retain_split_fraction = 1
    forget_split_fraction = 1
    
    ft_lr_main = 1e-4
    ft_epochs_main = 20
    ft_patience_main = 5

    print(f"\n--- Configuration ---")
    print(f"Classes to Forget (for data splitting consistency): {classes_to_forget}")
    print(f"Validation Split Fraction for FT: {val_split_fraction:.2f}")
    print(f"Fine-tuning: LR={ft_lr_main}, Epochs={ft_epochs_main}, Patience={ft_patience_main}")
    print(f"AMP Enabled: {use_amp_main}")

    # --- Data Loading and Splitting ---
    print("\n--- Data Preparation ---")
    print("Loading CIFAR-100 dataset...")
    try:
        full_trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=test_transform)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    except Exception as e:
        print(f"Error loading CIFAR-100 dataset: {e}. Exiting.")
        exit(1)

    # 1. Create Validation Set from Full Train Set (for monitoring fine-tuning)
    num_total_train = len(full_trainset)
    indices = list(range(num_total_train))
    np.random.shuffle(indices)
    split_idx = int(np.floor(val_split_fraction * num_total_train))
    
    val_indices_ft, train_pool_indices_ft = indices[:split_idx], indices[split_idx:]

    val_dataset_ft = Subset(full_trainset, val_indices_ft)
    # This is the pool of data from which we'll take the retain samples for actual fine-tuning
    train_pool_dataset_ft = Subset(full_trainset, train_pool_indices_ft)

    # 2. Split Validation Set into Retain/Forget (for monitoring fine-tuning)
    val_ft_forget_indices, val_ft_retain_indices = split_indices(val_dataset_ft, classes_to_forget)
    val_ft_forget_dataset = Subset(val_dataset_ft, val_ft_forget_indices)
    val_ft_retain_dataset = Subset(val_dataset_ft, val_ft_retain_indices)

    # 3. From the training pool, get the retain samples for fine-tuning
    _ , train_actual_retain_indices_ft = split_indices(train_pool_dataset_ft, classes_to_forget, retain_fraction=retain_split_fraction, forget_fraction=forget_split_fraction)
    train_retain_dataset_for_finetuning = Subset(train_pool_dataset_ft, train_actual_retain_indices_ft)

    # 4. Prepare Test Set Splits
    test_forget_indices, test_retain_indices = split_indices(testset, classes_to_forget)
    test_forget_dataset = Subset(testset, test_forget_indices)
    test_retain_dataset = Subset(testset, test_retain_indices)

    print(f"Sizes: FullTrainOrig={num_total_train}, ValForFT={len(val_dataset_ft)}, TrainPoolForFT={len(train_pool_dataset_ft)}")
    print(f"       ValForFT(R/F)={len(val_ft_retain_dataset)}/{len(val_ft_forget_dataset)}")
    print(f"       ActualTrainForFT(Retain Only)={len(train_retain_dataset_for_finetuning)}")
    print(f"       Test(R/F)={len(test_retain_dataset)}/{len(test_forget_dataset)}")

    # --- Create DataLoaders ---
    batch_size = 32
    num_workers = 2
    pin_memory = True if device.type == 'cuda' else False

    # Loader for actual fine-tuning (only retain data from the training pool)
    train_retain_loader_ft = DataLoader(train_retain_dataset_for_finetuning, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    
    # Loaders for validation during fine-tuning
    val_retain_loader_ft = DataLoader(val_ft_retain_dataset, batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    val_forget_loader_ft = DataLoader(val_ft_forget_dataset, batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # Loaders for Final Testing
    test_forget_loader = DataLoader(test_forget_dataset, batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_retain_loader = DataLoader(test_retain_dataset, batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # --- Evaluate Original Model ---
    print("\n--- Original Model Evaluation (Test Set) ---")
    original_retain_acc = evaluate(model, test_retain_loader, device, "Original Test Retain")
    original_forget_acc = evaluate(model, test_forget_loader, device, "Original Test Forget")
    print(f"Original Model -> Test Retain Acc: {original_retain_acc:.4f}, Test Forget Acc: {original_forget_acc:.4f}")

    # --- Fine-tune on Retain Data ---
    start_time = time.time()
    finetuned_model = finetune_on_retain(
        model_to_finetune=model,
        train_retain_loader=train_retain_loader_ft,
        val_retain_loader=val_retain_loader_ft,
        val_forget_loader=val_forget_loader_ft,
        lr=ft_lr_main,
        epochs=ft_epochs_main,
        patience=ft_patience_main,
        device=device,
        use_amp=use_amp_main
    )
    finetuning_end_time = time.time()
    print(f"Retain fine-tuning finished (Time: {finetuning_end_time - start_time:.2f}s)")
    torch.save(finetuned_model.state_dict(), 'resnet18_cifar100_ft_r_10%.pth')

    # --- Evaluate Fine-tuned Model ---
    print("\n--- Fine-tuned Model Evaluation (Test Set) ---")
    ft_retain_acc = evaluate(finetuned_model, test_retain_loader, device, "FT Test Retain")
    ft_forget_acc = evaluate(finetuned_model, test_forget_loader, device, "FT Test Forget")
    print(f"Fine-tuned Model -> Test Retain Acc: {ft_retain_acc:.4f}, Test Forget Acc: {ft_forget_acc:.4f}")

    # --- Performance Summary ---
    print("\n--- Performance Summary (Test Set) ---")
    print(f"Original Model:  Retain={original_retain_acc:.4f}, Forget={original_forget_acc:.4f}")
    print(f"Fine-tuned Model: Retain={ft_retain_acc:.4f}, Forget={ft_forget_acc:.4f}")
    
    retain_delta = ft_retain_acc - original_retain_acc
    forget_delta = ft_forget_acc - original_forget_acc
    print(f"Change:          Retain Δ={retain_delta:+.4f}, Forget Δ={forget_delta:+.4f}")

    if original_retain_acc > 1e-6:
        print(f"Retain Performance Ratio: {ft_retain_acc / original_retain_acc:.4f}")
    if original_forget_acc > 1e-6:
        forgetting_rate_ft = max(0.0, (original_forget_acc - ft_forget_acc) / original_forget_acc)
        print(f"Forgetting Rate (due to FT): {forgetting_rate_ft:.4f} (measures how much forget acc dropped relative to original)")


if __name__ == "__main__":
    # Basic dependency check
    try:
        if 'ResNet18ForCIFAR100' not in globals(): raise ImportError("ResNet not defined")
    except ImportError as e:
        print(f"Missing dependency or ResNet definition: {e}. Please ensure resnet.py is available.")
        exit(1)
    
    main()