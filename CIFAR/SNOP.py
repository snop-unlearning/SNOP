# SNOP_faster_better.py

import torch
import torch.nn as nn
import copy
import torchvision
from torch.utils.data import DataLoader, Subset, Dataset # Added Dataset
import numpy as np
from tqdm import tqdm
from resnet import ResNet18ForCIFAR100, test_transform
from utils import set_seed, split_indices, evaluate, load_model_checkpoint

import os
import time # For timing operations
# from torch.cuda.amp import GradScaler, autocast # <-- Import for Mixed Precision
from torch.amp import GradScaler, autocast # <-- Import for Mixed Precision


class SNOP:
    def __init__(self, model, alpha=1.0, beta=0.8, gamma=0.2, delta = 0.1, dampening_constant=0.9):
        """
        Initialize SNOP.

        Args:
            model: The model to unlearn.
            alpha, beta, gamma: Loss weights.
            dampening_constant: Strength of direct modification.
        """
        # Store original model state on CPU to save GPU memory
        self.original_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.direct_dampening = dampening_constant

    # --- identify_critical_circuits, compute_fisher_matrix, project_orthogonal_gradients ---
    def identify_critical_circuits(self, forget_loader, device='cuda', threshold_mode='stddev', percentile=95):
        """Identifies critical parameters using Fisher info on forget set."""
        mask = {}
        # Ensure we only create masks for trainable parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                mask[name] = torch.zeros_like(param, device='cpu') # Store mask on CPU initially

        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        # Initialize Fisher accumulator on the target device
        accumulated_fisher = {name: torch.zeros_like(param, device=device)
                              for name, param in self.model.named_parameters() if param.requires_grad}
        num_batches = 0

        print(f"Calculating Fisher for {len(forget_loader.dataset)} forget samples...")
        for inputs, labels in tqdm(forget_loader, desc="Forget Fisher Calc", leave=False):
            if inputs.numel() == 0: continue # Skip empty batches
            inputs, labels = inputs.to(device), labels.to(device)

            # Use autocast for potential speedup during Fisher calculation forward pass
            with autocast(enabled=(device.type=='cuda'), device_type='cuda'): # Check device type
                 self.model.zero_grad(set_to_none=True) # Use set_to_none=True for potential efficiency
                 outputs = self.model(inputs)
                 loss = criterion(outputs, labels)
            # No scaler needed for backward here as we only need grads for Fisher calc
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None and name in accumulated_fisher:
                    # Ensure gradients are float for pow(2)
                    accumulated_fisher[name] += param.grad.float().pow(2).detach()
            num_batches += 1
        self.model.zero_grad(set_to_none=True) # Clear grads after loop

        if num_batches == 0:
            print("Warning: Forget loader was empty during Fisher calculation.")
            return {name: m.to(device) for name, m in mask.items()} # Return zero masks on target device

        # Normalize and Threshold
        print(f"Applying threshold (Mode: {threshold_mode})...")
        for name in accumulated_fisher:
            if name not in mask: continue # Skip if param somehow wasn't included initially

            fisher_values = accumulated_fisher[name] / num_batches
            # Perform thresholding on CPU for consistency, unless dealing with huge models
            fisher_values_cpu = fisher_values.cpu().float() # Ensure float for calculations
            del fisher_values # Free GPU memory

            current_threshold = 0.0 # Default threshold
            if fisher_values_cpu.numel() == 0:
                 print(f"Warning: Fisher tensor {name} is empty.")
                 threshold_val = 0.0
            elif torch.isnan(fisher_values_cpu).any() or torch.isinf(fisher_values_cpu).any():
                 print(f"Warning: NaN or Inf found in Fisher values for {name}. Setting mask to zero.")
                 threshold_val = float('inf') # This will result in a zero mask
            else:
                 if threshold_mode == 'stddev':
                     mean_val = fisher_values_cpu.mean()
                     std_dev = fisher_values_cpu.std()
                     threshold_val = (mean_val + 1.5 * std_dev).item() # Get scalar value
                     # Fallback to percentile if mean+1.5*std is non-positive
                     if threshold_val <= 1e-10:
                         print(f"Warning: StdDev threshold for {name} is non-positive ({threshold_val:.2e}). Falling back to 95th percentile.")
                         q_val = torch.quantile(fisher_values_cpu.view(-1), 0.95)
                         threshold_val = q_val.item() if q_val.numel() > 0 else 0.0

                 elif threshold_mode == 'percentile':
                     q = max(1, min(99, percentile)) / 100.0
                     q_val = torch.quantile(fisher_values_cpu.view(-1), q)
                     threshold_val = q_val.item() if q_val.numel() > 0 else 0.0

                 else:
                     raise ValueError("Invalid threshold_mode. Choose 'stddev' or 'percentile'.")

            # Create binary mask (ensure threshold is positive)
            if threshold_val > 1e-10: # Use small epsilon
                mask[name] = (fisher_values_cpu > threshold_val).float()
            else:
                # Ensure mask remains zeros if threshold is non-positive
                mask[name].zero_()

        # Move final masks back to the target device
        mask = {name: m.to(device) for name, m in mask.items()}
        print("Critical mask generated.")
        return mask

    def compute_fisher_matrix(self, retain_loader, device='cuda'):
        """Computes Fisher Information (squared gradients) for the retain set."""
        fisher = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param, device=device) # Initialize on target device

        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        num_batches = 0

        print(f"Calculating Fisher for {len(retain_loader.dataset)} retain samples...")
        for inputs, labels in tqdm(retain_loader, desc="Retain Fisher Calc", leave=False):
            if inputs.numel() == 0: continue
            inputs, labels = inputs.to(device), labels.to(device)

            with autocast(enabled=(device.type=='cuda'), device_type='cuda'): # Use AMP
                self.model.zero_grad(set_to_none=True)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None and name in fisher:
                    fisher[name] += param.grad.float().pow(2).detach()
            num_batches += 1
        self.model.zero_grad(set_to_none=True)

        if num_batches > 0:
            for name in fisher:
                fisher[name] /= num_batches
                # Clamp potentially huge values if necessary, though normalization helps
                # fisher[name] = torch.clamp(fisher[name], max=1e6) # Optional clamping
        else:
            print("Warning: Retain loader was empty during Fisher calculation.")
            for name in fisher: fisher[name].zero_()

        print("Retain Fisher matrix computed.")
        return fisher

    def project_orthogonal_gradients(self, forget_grads, retain_fisher):
        """ Projects forget gradients orthogonal to retain Fisher directions. """
        orthogonal_grads = {}
        for name in forget_grads:
            if name in retain_fisher and forget_grads[name].numel() > 0 and retain_fisher[name].numel() > 0:
                retain_fisher_flat = retain_fisher[name].view(-1).float() # Ensure float
                grad_flat = forget_grads[name].view(-1).float()
                norm_fisher = torch.norm(retain_fisher_flat)

                if norm_fisher > 1e-8:
                    u_direction = retain_fisher_flat / norm_fisher
                    projection_scalar = torch.dot(grad_flat, u_direction)
                    proj = projection_scalar * u_direction
                    orthogonal = grad_flat - proj
                    orthogonal_grads[name] = orthogonal.view_as(forget_grads[name])
                else: # Fisher norm is zero, no projection possible/needed
                    orthogonal_grads[name] = forget_grads[name]
            elif name in forget_grads: # If no corresponding retain fisher (e.g., layer added later?)
                 orthogonal_grads[name] = forget_grads[name]
        return orthogonal_grads


    def unlearn(self,
            # --- Data Loaders ---
            train_retain_loader, train_forget_loader,
            val_retain_loader, val_forget_loader,
            # --- Control ---
            lr=0.0001, epochs=50, patience=7, device='cuda',
            # --- Initial Signal Calculation ---
            initial_forget_loader=None, initial_retain_loader=None,
            # --- Masking Strategy ---
            threshold_mode='stddev', percentile=95,
            dynamic_mask_update_freq=None,
            # --- Speed/Performance Options ---
            use_amp=True, 
            # --- Post-processing ---
            post_tune_epochs=0,
            post_tune_lr=None
            ):
        """
        Main unlearning function with options for speed and performance.
        """
        start_time = time.time()
        actual_initial_forget_loader = initial_forget_loader if initial_forget_loader is not None else train_forget_loader
        actual_initial_retain_loader = initial_retain_loader if initial_retain_loader is not None else train_retain_loader

        # --- Dataset Checks ---
        if len(actual_initial_forget_loader.dataset) == 0 or len(train_forget_loader.dataset) == 0:
             print("Warning: Forget dataset is empty. Skipping unlearning.")
             return self.model
        if len(actual_initial_retain_loader.dataset) == 0 or len(train_retain_loader.dataset) == 0:
             print("Warning: Retain dataset is empty. Skipping unlearning.")
             return self.model

        # --- Step 1 & 2 ---
        print("--- Initial Signal Calculation ---")
        critical_mask = self.identify_critical_circuits(actual_initial_forget_loader, device, threshold_mode, percentile)
        retain_fisher = self.compute_fisher_matrix(actual_initial_retain_loader, device)
        trainable_params = {name for name, param in self.model.named_parameters() if param.requires_grad}
        critical_mask = {k: v for k, v in critical_mask.items() if k in trainable_params}
        retain_fisher = {k: v for k, v in retain_fisher.items() if k in trainable_params}

        unlearned_model = copy.deepcopy(self.model).to(device)

        # --- Step 3A: Direct Parameter Modification ---
        print("\n--- Step 3A: Direct Parameter Modification ---")
        modified_params_count = 0
        with torch.no_grad():
            for name, param in unlearned_model.named_parameters():
                if name in critical_mask and name in retain_fisher and param.requires_grad:
                    if critical_mask[name].numel() == 0 or retain_fisher[name].numel() == 0: continue
                    if critical_mask[name].sum() > 0:
                        param_shape = param.shape
                        param_flat = param.view(-1).float()
                        fisher_flat = retain_fisher[name].view(-1).float()
                        norm_fisher = torch.norm(fisher_flat)
                        if norm_fisher > 1e-8:
                            u_direction = fisher_flat / norm_fisher
                            proj_scalar = torch.dot(param_flat, u_direction)
                            proj = proj_scalar * u_direction
                            param_ortho = param_flat - proj
                            mask_flat = critical_mask[name].view(-1).to(device).float()
                            dampened_ortho = param_ortho * (1.0 - self.direct_dampening * mask_flat)
                            new_param_flat = proj + dampened_ortho
                            param.copy_(new_param_flat.view(param_shape).to(param.dtype))
                            modified_params_count += 1
            print(f"Applied direct dampening to {modified_params_count} parameter tensors.")

        # --- STEP 3B: Fine-tune with optimization ---
        print("\n--- Step 3B: Optimization Loop ---")
        optimizer = torch.optim.AdamW(unlearned_model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=patience // 2, verbose=True, threshold=1e-4)
        criterion = nn.CrossEntropyLoss() # Used for both retain and direct forget loss
        amp_enabled = use_amp and (device.type == 'cuda')
        scaler = GradScaler(enabled=amp_enabled)
        print(f"Automatic Mixed Precision (AMP): {'Enabled' if amp_enabled else 'Disabled'}")

        # Initialize iterator for the forget loader if delta is active and forget data exists
        iter_train_forget_loader = None
        # Assuming self.delta is defined in your SNOP class __init__
        use_direct_forget_loss = hasattr(self, 'delta') and self.delta > 0 and len(train_forget_loader.dataset) > 0
        if use_direct_forget_loss:
            iter_train_forget_loader = iter(train_forget_loader)
            print(f"Direct forget loss maximization enabled with delta: {self.delta}")

        best_overall_forget_acc = float('inf')
        best_retain_acc_at_min_forget = -float('inf')
        best_model_state = None
        patience_counter = 0
        best_epoch = -1

        for epoch in range(epochs):
            epoch_start_time = time.time()
            print(f"\n--- Epoch {epoch+1}/{epochs} ---")

            if dynamic_mask_update_freq is not None and epoch > 0 and epoch % dynamic_mask_update_freq == 0:
                 update_start_time = time.time()
                 print(f"Updating critical mask and retain Fisher (Epoch {epoch+1})...")
                 critical_mask = self.identify_critical_circuits(train_forget_loader, device, threshold_mode, percentile)
                 retain_fisher = self.compute_fisher_matrix(train_retain_loader, device)
                 critical_mask = {k: v for k, v in critical_mask.items() if k in trainable_params}
                 retain_fisher = {k: v for k, v in retain_fisher.items() if k in trainable_params}
                 print(f"Dynamic update took {time.time() - update_start_time:.2f}s")

            forget_grads = {}
            unlearned_model.eval()
            num_forget_batches = 0
            for inputs_fg, labels_fg in train_forget_loader: 
                 if inputs_fg.numel() == 0: continue
                 inputs_fg, labels_fg = inputs_fg.to(device), labels_fg.to(device)
                 unlearned_model.zero_grad(set_to_none=True)
                 outputs_fg = unlearned_model(inputs_fg)
                 loss_fg = criterion(outputs_fg, labels_fg.long())
                 loss_fg.backward()
                 for name, param in unlearned_model.named_parameters():
                      if param.grad is not None:
                          grad_clone = param.grad.detach().clone()
                          if name not in forget_grads: forget_grads[name] = grad_clone
                          else: forget_grads[name] += grad_clone
                 num_forget_batches += 1
            unlearned_model.zero_grad(set_to_none=True)
            if num_forget_batches > 0:
                for name in forget_grads: forget_grads[name] /= num_forget_batches
            else: print(f"Warning: Epoch {epoch+1}, no batches in train_forget_loader.") ; forget_grads = {}
            orthogonal_grads = self.project_orthogonal_gradients(forget_grads, retain_fisher)

            unlearned_model.train()
            epoch_train_loss = 0.0
            num_train_batches = 0
            
            current_batch_direct_forget_loss = 0.0 # For logging

            with tqdm(train_retain_loader, desc=f"Epoch {epoch+1}/{epochs} Training", leave=False, dynamic_ncols=True) as pbar:
                for batch_idx, (inputs, labels) in enumerate(pbar):
                    if inputs.numel() == 0: continue
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad(set_to_none=True)

                    with autocast(enabled=amp_enabled, device_type='cuda'):
                        # --- Retain Loss ---
                        outputs = unlearned_model(inputs)
                        retain_loss = criterion(outputs, labels.long())

                        # --- Direct Forget Loss Term ---
                        direct_forget_loss_term = torch.tensor(0.0, device=device, dtype=torch.float32)
                        if use_direct_forget_loss:
                            try:
                                forget_inputs, forget_labels = next(iter_train_forget_loader)
                            except StopIteration:
                                iter_train_forget_loader = iter(train_forget_loader) # Reset iterator
                                forget_inputs, forget_labels = next(iter_train_forget_loader)
                            
                            if forget_inputs.numel() > 0: # Ensure batch is not empty
                                forget_inputs, forget_labels = forget_inputs.to(device), forget_labels.to(device)
                                
                                forget_outputs_direct = unlearned_model(forget_inputs)
                                direct_forget_loss_term = criterion(forget_outputs_direct, forget_labels.long())
                                current_batch_direct_forget_loss = direct_forget_loss_term.item() # For logging
                            else:
                                current_batch_direct_forget_loss = 0.0


                        # --- Regularization Terms ---
                        forget_sparsity = torch.tensor(0.0, device=device, dtype=torch.float32)
                        retain_stability = torch.tensor(0.0, device=device, dtype=torch.float32)
                        for name, param in unlearned_model.named_parameters():
                            if param.requires_grad:
                                is_critical = name in critical_mask and critical_mask[name].sum() > 0
                                if is_critical and name in orthogonal_grads:
                                     mask_term = critical_mask[name] 
                                     ortho_grad_term = orthogonal_grads[name]
                                     if mask_term.shape == ortho_grad_term.shape:
                                         forget_sparsity += (mask_term * ortho_grad_term).abs().sum().float()
                                if not is_critical:
                                    retain_stability += param.float().pow(2).sum()

                        # --- Combine Losses ---
                        total_loss = (self.alpha * retain_loss.float() +
                                      self.beta * forget_sparsity +  # Already float from calculation
                                      self.gamma * retain_stability) # Already float
                        
                        if use_direct_forget_loss:
                            # Subtract to MAXIMIZE the direct_forget_loss_term
                            total_loss = total_loss - (self.delta * direct_forget_loss_term.float())

                    scaler.scale(total_loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(unlearned_model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()

                    epoch_train_loss += total_loss.item()
                    num_train_batches += 1
                    if batch_idx % 20 == 0:
                        log_postfix = {"loss": f"{total_loss.item():.4f}", "ret_L": f"{retain_loss.item():.2f}"}
                        if use_direct_forget_loss:
                            log_postfix["fgt_L_max"] = f"{current_batch_direct_forget_loss:.2f}"
                        pbar.set_postfix(**log_postfix)

            avg_epoch_loss = epoch_train_loss / num_train_batches if num_train_batches > 0 else 0

            epoch_retain_acc = evaluate(unlearned_model, val_retain_loader, device, "Val Retain")
            epoch_forget_acc = evaluate(unlearned_model, val_forget_loader, device, "Val Forget")
            epoch_end_time = time.time()
            print(f"Epoch {epoch+1}/{epochs} Summary - TrainLoss: {avg_epoch_loss:.4f}, "
                  f"Val RetainAcc: {epoch_retain_acc:.4f}, Val ForgetAcc: {epoch_forget_acc:.6f} "
                  f"(Time: {epoch_end_time - epoch_start_time:.2f}s)")

            scheduler.step(epoch_forget_acc)

            is_better = False
            if epoch_forget_acc < best_overall_forget_acc - 1e-6:
                print(f"-> New best forget accuracy: {epoch_forget_acc:.6f} (was {best_overall_forget_acc:.6f})")
                best_overall_forget_acc = epoch_forget_acc
                best_retain_acc_at_min_forget = epoch_retain_acc 
                best_model_state = copy.deepcopy(unlearned_model.state_dict())
                best_epoch = epoch + 1
                is_better = True
                patience_counter = 0
            elif abs(epoch_forget_acc - best_overall_forget_acc) < 1e-6:
                if epoch_retain_acc > best_retain_acc_at_min_forget + 1e-5:
                    print(f"  -> Forget accuracy tied ({epoch_forget_acc:.6f}), retain accuracy improved: {epoch_retain_acc:.4f} (was {best_retain_acc_at_min_forget:.4f})")
                    best_retain_acc_at_min_forget = epoch_retain_acc
                    best_model_state = copy.deepcopy(unlearned_model.state_dict()) 
                    best_epoch = epoch + 1
                    is_better = True
                    patience_counter = 0 
                else:
                     patience_counter += 1
                     print(f"  -> No improvement according to criterion. Patience: {patience_counter}/{patience}")
            else:
                patience_counter += 1
                print(f"  -> Forget accuracy worse than best ({epoch_forget_acc:.6f} > {best_overall_forget_acc:.6f}). Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after epoch {epoch+1}.")
                break

        unlearning_end_time = time.time()
        print(f"\n--- Unlearning Optimization Finished (Time: {unlearning_end_time - start_time:.2f}s) ---")
        if best_model_state is not None:
            print(f"Loading best model state from Epoch {best_epoch} (Val Forget Acc: {best_overall_forget_acc:.6f}, Val Retain Acc: {best_retain_acc_at_min_forget:.4f})")
            unlearned_model.load_state_dict(best_model_state)
        else:
            print("Warning: No improvement detected during training. Using model from last epoch.")

        if post_tune_epochs > 0 and len(train_retain_loader.dataset) > 0:
            post_tune_start_time = time.time()
            print(f"\n--- Post-Unlearning Retain Fine-tuning ({post_tune_epochs} epochs) ---")
            post_lr = post_tune_lr if post_tune_lr is not None else lr / 10.0
            print(f"Post-tuning LR: {post_lr}")
            post_optimizer = torch.optim.Adam(unlearned_model.parameters(), lr=post_lr)
            post_scaler = GradScaler(enabled=amp_enabled)
            unlearned_model.train()
            for pt_epoch in range(post_tune_epochs):
                pt_epoch_loss = 0
                num_pt_batches = 0
                with tqdm(train_retain_loader, desc=f"Post-Tune Epoch {pt_epoch+1}/{post_tune_epochs}", leave=False, dynamic_ncols=True) as pbar:
                    for inputs, labels in pbar:
                        if inputs.numel() == 0: continue
                        inputs, labels = inputs.to(device), labels.to(device)
                        post_optimizer.zero_grad(set_to_none=True)
                        with autocast(enabled=amp_enabled, device_type='cuda'):
                            outputs = unlearned_model(inputs)
                            loss = criterion(outputs, labels.long())
                        post_scaler.scale(loss).backward()
                        post_scaler.step(post_optimizer)
                        post_scaler.update()
                        pt_epoch_loss += loss.item()
                        num_pt_batches += 1
                        if num_pt_batches % 20 == 0: pbar.set_postfix(loss=f"{loss.item():.4f}")
                avg_pt_loss = pt_epoch_loss / num_pt_batches if num_pt_batches > 0 else 0
                pt_val_retain = evaluate(unlearned_model, val_retain_loader, device, "Post-Tune Val Retain")
                pt_val_forget = evaluate(unlearned_model, val_forget_loader, device, "Post-Tune Val Forget")
                print(f"Post-Tune Epoch {pt_epoch+1} - Avg Loss: {avg_pt_loss:.4f}, Val Retain: {pt_val_retain:.4f}, Val Forget: {pt_val_forget:.6f}")
            print(f"--- Post-tuning finished (Time: {time.time() - post_tune_start_time:.2f}s) ---")

        total_time = time.time() - start_time
        print(f"\n--- SNOP Process Complete (Total Time: {total_time:.2f}s) ---")
        return unlearned_model
    
    # --- validate_unlearning ---
    def validate_unlearning(self, unlearned_model, forget_loader, retain_loader, critical_mask, device='cuda'):
         """Validates unlearning using mechanistic ablation."""
         print("\n--- Mechanistic Validation ---")
         retain_acc = evaluate(unlearned_model, retain_loader, device, "Validate Retain")
         forget_acc = evaluate(unlearned_model, forget_loader, device, "Validate Forget")
         print(f"Unlearned Model: Retain={retain_acc:.4f}, Forget={forget_acc:.4f}")

         ablated_model = copy.deepcopy(unlearned_model)
         trainable_params = {name for name, param in ablated_model.named_parameters() if param.requires_grad}
         # Use the passed critical_mask
         valid_mask = {k: v for k, v in critical_mask.items() if k in trainable_params}
         print(f"Applying ablation using mask derived from {len(valid_mask)} parameter groups.")

         params_ablated = 0
         with torch.no_grad():
             for name, param in ablated_model.named_parameters():
                 if name in valid_mask:
                      mask_tensor = valid_mask[name].to(param.device) # Ensure mask is on same device
                      if mask_tensor.shape == param.shape:
                           param.data.mul_(1.0 - mask_tensor) # Modify in-place
                           params_ablated += 1
                      else:
                           print(f"Warning: Shape mismatch during ablation - {name}: Mask {mask_tensor.shape}, Param {param.shape}")

         print(f"Ablated {params_ablated} parameter tensors.")
         print("Evaluating ablated model...")
         ablated_retain_acc = evaluate(ablated_model, retain_loader, device, "Ablated Retain")
         ablated_forget_acc = evaluate(ablated_model, forget_loader, device, "Ablated Forget")
         print(f"Ablated Model: Retain={ablated_retain_acc:.4f}, Forget={ablated_forget_acc:.4f}")


         # Calculate metrics robustly
         forgetting_effectiveness = 0.0
         if ablated_forget_acc > 1e-8:
             forgetting_effectiveness = max(0.0, min(1.0, (ablated_forget_acc - forget_acc) / ablated_forget_acc))

         retain_preservation = 0.0
         if ablated_retain_acc > 1e-8:
              retain_preservation = retain_acc / ablated_retain_acc

         print(f"Forget Effectiveness (vs Ablation): {forgetting_effectiveness:.4f}")
         print(f"Retain Preservation (vs Ablation): {retain_preservation:.4f}")

         return {
             'forget_accuracy': forget_acc, 'retain_accuracy': retain_acc,
             'ablated_forget_accuracy': ablated_forget_acc, 'ablated_retain_accuracy': ablated_retain_acc,
             'forgetting_effectiveness (vs ablation)': forgetting_effectiveness,
             'retain_preservation (vs ablation)': retain_preservation
         }


def main():
    set_seed(42) # Default seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Enable AMP only if CUDA is available
    use_amp_main = True if device.type == 'cuda' else False

    # --- Model Loading ---
    model_path_final = 'resnet18_cifar100_final.pth'
    model_path_best = 'resnet18_cifar100_best.pth'
    model = load_model_checkpoint(ResNet18ForCIFAR100, model_path_best, model_path_final, device)
    # The utility function load_model_checkpoint prints warnings if no model is loaded.

    # --- Configuration ---
    import random
    
    classes_to_forget = [54] # Randomly select 10 classes to forget
    val_split_fraction = 0.1 # Use 15% of original train for validation
    train_retain_sampling_fraction = 1 # Use 10% of available retain for opt loop
    train_forget_sampling_fraction = 1 # Use 20% of available forget for opt loop
    use_full_data_for_initial_signals = False # Recommended: Use all non-val data for initial signals
    
    # --- New Config Options ---
    dynamic_mask_update_freq_main = 2 # Update every 10 epochs, None to disable
    post_tune_epochs_main = 1 # Add 2 epochs of post-tuning
    post_tune_lr_main = 1e-5 # Small LR for post-tuning
    
    # --- Hyperparameters (tuned from previous runs) ---
    alpha_main = 20   # 50 for 10% data
    beta_main = 10.0 # Keep aggressive forgetting (5 for 10% data)
    gamma_main = 0.005 # Keep low stability weight
    delta_main = 0.2 
    dampening_main = 0.9
    lr_main = 0.0001
    epochs_main = 1 # (5 epochs for 10% data)
    patience_main = 5 # Slightly increased patience

    print(f"\n--- Configuration ---")
    print(f"Classes to Forget: {classes_to_forget}")
    print(f"Validation Split Fraction: {val_split_fraction:.2f}")
    print(f"Unlearning Optimization Data Sampling: Retain={train_retain_sampling_fraction:.2f}, Forget={train_forget_sampling_fraction:.2f}")
    print(f"Use Full Data for Initial Fisher/Mask: {use_full_data_for_initial_signals}")
    print(f"AMP Enabled: {use_amp_main}")
    print(f"Dynamic Mask Update Freq: {dynamic_mask_update_freq_main}")
    print(f"Post-Unlearning Tune Epochs: {post_tune_epochs_main} (LR: {post_tune_lr_main})")
    print(f"Hyperparameters: alpha={alpha_main}, beta={beta_main}, gamma={gamma_main}, damp={dampening_main}, lr={lr_main}")


    # --- Data Loading and Splitting ---
    print("\n--- Data Preparation ---")
    print("Loading CIFAR-100 dataset...")
    try:
        full_trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=test_transform)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    except Exception as e:
        print(f"Error loading CIFAR-100 dataset: {e}")
        print("Please check your internet connection or dataset path.")
        exit(1)

    # 1. Create Dedicated Validation Set from Full Train Set
    num_train = len(full_trainset)
    indices = list(range(num_train))
    np.random.shuffle(indices) # Shuffle before splitting
    split = int(np.floor(val_split_fraction * num_train))
    val_indices, train_subset_indices = indices[:split], indices[split:]

    val_dataset = Subset(full_trainset, val_indices)
    # This is the pool of data available for unlearning training/signals
    train_subset_for_unlearning = Subset(full_trainset, train_subset_indices)

    # 2. Split Validation Set into Retain/Forget
    val_forget_indices, val_retain_indices = split_indices(val_dataset, classes_to_forget)
    val_forget_dataset = Subset(val_dataset, val_forget_indices)
    val_retain_dataset = Subset(val_dataset, val_retain_indices)

    # 3. Split the Training Subset Pool into Full Retain/Forget
    full_train_forget_indices, full_train_retain_indices = split_indices(train_subset_for_unlearning, classes_to_forget)
    full_train_forget_dataset = Subset(train_subset_for_unlearning, full_train_forget_indices)
    full_train_retain_dataset = Subset(train_subset_for_unlearning, full_train_retain_indices)

    # 4. Subsample from the Full Train Retain/Forget for the Optimization Loop
    # Ensure indices are available before sampling
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

    # Print final dataset sizes for clarity
    print(f"Sizes: FullTrain={num_train}, Val={len(val_dataset)}, TrainSub={len(train_subset_for_unlearning)}")
    print(f"       Val(R/F)={len(val_retain_dataset)}/{len(val_forget_dataset)}")
    print(f"       FullTrainSub(R/F)={len(full_train_retain_dataset)}/{len(full_train_forget_dataset)}")
    print(f"       OptLoop(R/F)={len(train_retain_dataset_opt)}/{len(train_forget_dataset_opt)}")
    print(f"       Test(R/F)={len(test_retain_dataset)}/{len(test_forget_dataset)}")

    # --- Create DataLoaders ---
    batch_size = 32 # Keep batch size reasonable
    num_workers = 2 # Adjust based on your system
    pin_memory = True if device.type == 'cuda' else False

    # Loaders for Optimization Loop
    train_forget_loader_opt = DataLoader(train_forget_dataset_opt, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True) # drop_last might help stability
    train_retain_loader_opt = DataLoader(train_retain_dataset_opt, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)

    # Loaders for Validation (Dedicated Split)
    val_forget_loader = DataLoader(val_forget_dataset, batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=pin_memory) # Use larger batch for faster validation
    val_retain_loader = DataLoader(val_retain_dataset, batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # Loaders for Initial Signal Calculation (Optional: Full Data)
    initial_forget_loader = None
    initial_retain_loader = None
    if use_full_data_for_initial_signals:
        # Use the full split from the training subset (non-validation part)
        initial_forget_loader = DataLoader(full_train_forget_dataset, batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=pin_memory) # Larger batch for calc
        initial_retain_loader = DataLoader(full_train_retain_dataset, batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        print("Using full train subset data for initial Fisher/Mask calculation.")

    # Loaders for Final Testing
    test_forget_loader = DataLoader(test_forget_dataset, batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=pin_memory) # Larger batch for test
    test_retain_loader = DataLoader(test_retain_dataset, batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)


    # --- Evaluate Original Model ---
    print("\n--- Original Model Evaluation (Test Set) ---")
    original_retain_acc = evaluate(model, test_retain_loader, device, "Original Test Retain")
    original_forget_acc = evaluate(model, test_forget_loader, device, "Original Test Forget")
    print(f"Original Model -> Test Retain Acc: {original_retain_acc:.4f}, Test Forget Acc: {original_forget_acc:.4f}")


    # --- Initialize and Run SNOP ---
    print("\n--- SNOP Unlearning ---")
    snop = SNOP(model, alpha=alpha_main, beta=beta_main, gamma=gamma_main, delta= delta_main,
                dampening_constant=dampening_main,)

    import time
    start_time = time.time()

    unlearned_model = snop.unlearn(
        # Optimization loaders (subsampled)
        train_retain_loader=train_retain_loader_opt,
        train_forget_loader=train_forget_loader_opt,
        # Validation loaders (dedicated split)
        val_retain_loader=val_retain_loader,
        val_forget_loader=val_forget_loader,
        # Control
        lr=lr_main, epochs=epochs_main, patience=patience_main, device=device,
        # Initial signals (optional override)
        initial_forget_loader=initial_forget_loader,
        initial_retain_loader=initial_retain_loader,
        # Masking
        threshold_mode='percentile', percentile=95, # Keep percentile as it worked well
        dynamic_mask_update_freq=dynamic_mask_update_freq_main,
        # Speed/Perf
        use_amp=use_amp_main,
        # Post-tuning
        post_tune_epochs=post_tune_epochs_main,
        post_tune_lr=post_tune_lr_main
    )

    torch.save(unlearned_model.state_dict(), 'resnet18_cifar100_snop_10%.pth')

    unlearning_end_time = time.time()
    print(f"SNOP unlearning finished (Time: {unlearning_end_time - start_time:.2f}s)")    
    
    # --- Evaluate Unlearned Model ---
    print("\n--- Unlearned Model Evaluation (Test Set) ---")
    unlearned_retain_acc = evaluate(unlearned_model, test_retain_loader, device, "Unlearned Test Retain")
    unlearned_forget_acc = evaluate(unlearned_model, test_forget_loader, device, "Unlearned Test Forget")
    print(f"Unlearned Model -> Test Retain Acc: {unlearned_retain_acc:.4f}, Test Forget Acc: {unlearned_forget_acc:.4f}")

    # --- Performance Summary ---
    print("\n--- Performance Summary ---")
    print(f"Original Model: Retain={original_retain_acc:.4f}, Forget={original_forget_acc:.4f}")
    print(f"Unlearned Model: Retain={unlearned_retain_acc:.4f}, Forget={unlearned_forget_acc:.4f}")
    retain_delta = unlearned_retain_acc - original_retain_acc
    forget_delta = unlearned_forget_acc - original_forget_acc
    print(f"Change: Retain Δ={retain_delta:+.4f}, Forget Δ={forget_delta:+.4f}")
    if original_retain_acc > 1e-4: # Avoid division by zero
        print(f"Retain Performance Ratio: {unlearned_retain_acc / original_retain_acc:.4f}")
    if original_forget_acc > 1e-4:
         forgetting_rate = max(0.0, (original_forget_acc - unlearned_forget_acc) / original_forget_acc) # Ensure rate is non-negative
         print(f"Forgetting Rate: {forgetting_rate:.4f}")


    # --- Optional: Mechanistic Validation ---
    print("\n--- Mechanistic Validation (Test Set) ---")
    # Need the final critical mask for validation. Either save it during training
    # or recalculate it here (potentially on test forget data for true test validation).
    # For simplicity, let's assume we recalculate on test data here.
    try:
        validation_mask = snop.identify_critical_circuits(test_forget_loader, device, threshold_mode='percentile')
        validation_results = snop.validate_unlearning(
            unlearned_model=unlearned_model,
            forget_loader=test_forget_loader,
            retain_loader=test_retain_loader,
            critical_mask=validation_mask,
            device=device
        )
        print("\nValidation Results:")
        for key, value in validation_results.items():
            print(f"{key}: {value:.4f}")
    except Exception as e:
        print(f"Could not perform mechanistic validation: {e}")


if __name__ == "__main__":
    # Basic dependency check
    try:
        if 'ResNet18ForCIFAR100' not in globals(): raise ImportError("ResNet not defined")
    except ImportError as e:
        print(f"Missing dependency or ResNet definition: {e}. Please install required libraries and ensure resnet.py is available.")
        exit(1)

    # Check for model file before running main logic
    model_path_final = 'resnet18_cifar100_final.pth'
    model_path_best = 'resnet18_cifar100_best.pth'
    if not os.path.exists(model_path_final) and not os.path.exists(model_path_best):
         print(f"Warning: Model weights not found at '{model_path_final}' or '{model_path_best}'. Ensure model is present or script might fail/use random weights.")
         # Decide whether to exit or proceed

    main()