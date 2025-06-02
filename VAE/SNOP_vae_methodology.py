import torch
from torch.cuda.amp import GradScaler
from vae_utils import vae_loss_function, evaluate_vae
import tqdm
import time
import copy

class SNOP_VAE:
    def __init__(self, model, alpha=1.0, beta=0.8, gamma=0.2, dampening_constant=0.9, delta = 1, kl_weight=1.0):
        """
        Initialize SNOP for VAE.

        Args:
            model: The VAE model to unlearn.
            alpha, beta, gamma: Loss weights.
            dampening_constant: Strength of direct modification.
            kl_weight: Weight for the KL divergence term in the VAE loss (used for retain set).
        """
        self.original_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.direct_dampening = dampening_constant
        self.kl_weight = kl_weight # Store KL weight

    def identify_critical_circuits(self, forget_loader, device='cuda:1', threshold_mode='stddev', percentile=95):
        """Identifies critical parameters using RECONSTRUCTION loss gradients on forget set."""
        mask = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                mask[name] = torch.zeros_like(param, device='cpu')

        self.model.eval()
        accumulated_fisher = {name: torch.zeros_like(param, device=device)
                              for name, param in self.model.named_parameters() if param.requires_grad}
        num_batches = 0

        print(f"Calculating Fisher (Recon Loss) for {len(forget_loader.dataset)} forget samples...")
        # No labels needed for VAE reconstruction loss calculation
        for inputs, _ in tqdm(forget_loader, desc="Forget Fisher Calc (Recon Loss)", leave=False):
            if inputs.numel() == 0: continue
            inputs = inputs.to(device)

            self.model.zero_grad(set_to_none=True)
            recon_batch, mu, logvar = self.model(inputs)
            # Calculate loss using ONLY the reconstruction component
            recon_loss, _ = vae_loss_function(recon_batch, inputs, mu, logvar)
            # Normalize loss by batch size for consistent gradient magnitudes? Optional but can help.
            loss = recon_loss / inputs.size(0)

            # No scaler needed for backward here
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None and name in accumulated_fisher:
                    accumulated_fisher[name] += param.grad.float().pow(2).detach()
            num_batches += 1
        self.model.zero_grad(set_to_none=True)

        if num_batches == 0:
            print("Warning: Forget loader was empty during Fisher calculation.")
            return {name: m.to(device) for name, m in mask.items()}

        # --- Normalization and Thresholding (same as before) ---
        print(f"Applying threshold (Mode: {threshold_mode})...")
        for name in accumulated_fisher:
             if name not in mask: continue

             fisher_values = accumulated_fisher[name] / num_batches
             fisher_values_cpu = fisher_values.cpu().float()
             del fisher_values

             if fisher_values_cpu.numel() == 0:
                  print(f"Warning: Fisher tensor {name} is empty.")
                  threshold_val = 0.0
             elif torch.isnan(fisher_values_cpu).any() or torch.isinf(fisher_values_cpu).any():
                  print(f"Warning: NaN or Inf found in Fisher values for {name}. Setting mask to zero.")
                  threshold_val = float('inf')
             else:
                  if threshold_mode == 'stddev':
                      mean_val = fisher_values_cpu.mean()
                      std_dev = fisher_values_cpu.std()
                      threshold_val = (mean_val + 1.5 * std_dev).item()
                      if threshold_val <= 1e-10:
                          print(f"Warning: StdDev threshold for {name} non-positive ({threshold_val:.2e}). Falling back to 95th percentile.")
                          q_val = torch.quantile(fisher_values_cpu.view(-1), 0.95)
                          threshold_val = q_val.item() if q_val.numel() > 0 else 0.0
                  elif threshold_mode == 'percentile':
                      q = max(1, min(99, percentile)) / 100.0
                      q_val = torch.quantile(fisher_values_cpu.view(-1), q)
                      threshold_val = q_val.item() if q_val.numel() > 0 else 0.0
                  else: raise ValueError("Invalid threshold_mode.")

             if threshold_val > 1e-10:
                 mask[name] = (fisher_values_cpu > threshold_val).float()
             else: mask[name].zero_()
        # --- End Thresholding ---

        mask = {name: m.to(device) for name, m in mask.items()}
        print("Critical mask generated based on forget reconstruction loss.")
        
        total_params = 0
        masked_params = 0
        for name in mask:
            total_params += mask[name].numel()
            masked_params += mask[name].sum().item()
            
        if total_params > 0:
            print(f"Critical mask density: {masked_params/total_params*100:.2f}% ({masked_params}/{total_params})")
            
        return mask

    def compute_fisher_matrix(self, retain_loader, device='cuda'):
        """Computes Fisher Information using FULL VAE loss for the retain set."""
        fisher = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param, device=device)

        self.model.eval()
        num_batches = 0

        print(f"Calculating Fisher (Full VAE Loss) for {len(retain_loader.dataset)} retain samples...")
        # No labels needed for VAE loss
        for inputs, _ in tqdm(retain_loader, desc="Retain Fisher Calc (Full Loss)", leave=False):
            if inputs.numel() == 0: continue
            inputs = inputs.to(device)

            self.model.zero_grad(set_to_none=True)
            recon_batch, mu, logvar = self.model(inputs)
            # Calculate the FULL VAE loss (weighted)
            recon_loss, kld_loss = vae_loss_function(recon_batch, inputs, mu, logvar)
            total_loss = recon_loss + self.kl_weight * kld_loss
            # Normalize loss by batch size
            loss = total_loss / inputs.size(0)

            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None and name in fisher:
                    fisher[name] += param.grad.float().pow(2).detach()
            num_batches += 1
        self.model.zero_grad(set_to_none=True)

        if num_batches > 0:
            for name in fisher: fisher[name] /= num_batches
        else:
            print("Warning: Retain loader was empty during Fisher calculation.")
            for name in fisher: fisher[name].zero_()

        print("Retain Fisher matrix (Full VAE Loss) computed.")
        return fisher

    def project_orthogonal_gradients(self, forget_grads, retain_fisher):
        """ Projects forget gradients orthogonal to retain Fisher directions. (No change needed)"""
        orthogonal_grads = {}
        for name in forget_grads:
            if name in retain_fisher and forget_grads[name].numel() > 0 and retain_fisher[name].numel() > 0:
                retain_fisher_flat = retain_fisher[name].view(-1).float()
                grad_flat = forget_grads[name].view(-1).float()
                norm_fisher = torch.norm(retain_fisher_flat)

                if norm_fisher > 1e-8:
                    u_direction = retain_fisher_flat / norm_fisher
                    projection_scalar = torch.dot(grad_flat, u_direction)
                    proj = projection_scalar * u_direction
                    orthogonal = grad_flat - proj
                    orthogonal_grads[name] = orthogonal.view_as(forget_grads[name])
                else:
                    orthogonal_grads[name] = forget_grads[name]
            elif name in forget_grads:
                 orthogonal_grads[name] = forget_grads[name]
        return orthogonal_grads

    def unlearn(self,
                # --- Data Loaders ---
                train_retain_loader, train_forget_loader,
                val_retain_loader, val_forget_loader,
                # --- Control ---
                lr=0.0001, epochs=50, patience=7, device='cuda',
                kl_weight=1.0, # Pass KL weight for loss calcs
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

        start_time = time.time()
        # Use provided initial loaders or default to training loaders
        actual_initial_forget_loader = initial_forget_loader if initial_forget_loader is not None else train_forget_loader
        actual_initial_retain_loader = initial_retain_loader if initial_retain_loader is not None else train_retain_loader
        iter_train_forget_loader = iter(train_forget_loader) # train_forget_loader_opt

        # Dataset Checks
        if len(actual_initial_forget_loader.dataset) == 0 or len(train_forget_loader.dataset) == 0:
             print("Warning: Forget dataset is empty. Skipping unlearning.")
             return self.model
        if len(actual_initial_retain_loader.dataset) == 0 or len(train_retain_loader.dataset) == 0:
             print("Warning: Retain dataset is empty. Skipping unlearning.")
             return self.model

        # --- Step 1 & 2: Initial Signals ---
        print("--- Initial Signal Calculation (VAE Adaptation) ---")
        # Critical mask based on forget RECONSTRUCTION loss
        critical_mask = self.identify_critical_circuits(actual_initial_forget_loader, device, threshold_mode, percentile)
        # Retain Fisher based on FULL VAE loss
        retain_fisher = self.compute_fisher_matrix(actual_initial_retain_loader, device)
        trainable_params = {name for name, param in self.model.named_parameters() if param.requires_grad}
        critical_mask = {k: v for k, v in critical_mask.items() if k in trainable_params}
        retain_fisher = {k: v for k, v in retain_fisher.items() if k in trainable_params}

        unlearned_model = copy.deepcopy(self.model).to(device)

        # --- Step 3A: Direct Parameter Modification (using retain Fisher based on full loss) ---
        print("\n--- Step 3A: Direct Parameter Modification ---")
        modified_params_count = 0
        with torch.no_grad():
            for name, param in unlearned_model.named_parameters():
                if name in critical_mask and name in retain_fisher and param.requires_grad:
                    if critical_mask[name].numel() == 0 or retain_fisher[name].numel() == 0: continue
                    if critical_mask[name].sum() > 0:
                        param_shape = param.shape
                        param_flat = param.view(-1).float()
                        fisher_flat = retain_fisher[name].view(-1).float() # Fisher based on full VAE loss
                        norm_fisher = torch.norm(fisher_flat)
                        if norm_fisher > 1e-8:
                            u_direction = fisher_flat / norm_fisher
                            proj_scalar = torch.dot(param_flat, u_direction)
                            proj = proj_scalar * u_direction
                            param_ortho = param_flat - proj
                            mask_flat = critical_mask[name].view(-1).to(device).float() # Mask based on forget recon loss
                            dampened_ortho = param_ortho * (1.0 - self.direct_dampening * mask_flat)
                            new_param_flat = proj + dampened_ortho
                            param.copy_(new_param_flat.view(param_shape).to(param.dtype))
                            modified_params_count += 1
            print(f"Applied direct dampening to {modified_params_count} parameter tensors.")


        # --- STEP 3B: Fine-tune with optimization ---
        print("\n--- Step 3B: Optimization Loop (VAE Adaptation) ---")
        optimizer = torch.optim.AdamW(unlearned_model.parameters(), lr=lr, weight_decay=1e-5) # Lower weight decay maybe
        # Scheduler monitors validation FORGET RECONSTRUCTION LOSS (want it to increase or stay high)
        # OR monitor validation RETAIN FULL LOSS (want it to stay low)
        # try monitoring RETAIN full loss (mode='min')
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=patience // 2, verbose=True, threshold=1e-4)

        amp_enabled = False
        # amp_enabled = use_amp and (device.type == 'cuda')
        scaler = GradScaler(enabled=amp_enabled)
        print(f"Automatic Mixed Precision (AMP): {'Enabled' if amp_enabled else 'Disabled'}")

        # Early Stopping Initialization (minimize validation retain loss)
        best_val_retain_loss = float('inf')
        best_forget_recon_at_best_retain = float('inf') # Track forget recon when retain is best
        best_model_state = None
        patience_counter = 0
        best_epoch = -1

        for epoch in range(epochs):
            epoch_start_time = time.time()
            print(f"\n--- Epoch {epoch+1}/{epochs} ---")

            # Dynamic Mask/Fisher Update (Optional - uses same logic as before)
            if dynamic_mask_update_freq is not None and epoch > 0 and epoch % dynamic_mask_update_freq == 0:
                update_start_time = time.time()
                print(f"Updating critical mask (Recon Loss) and retain Fisher (Full Loss) (Epoch {epoch+1})...")
                critical_mask = self.identify_critical_circuits(train_forget_loader, device, threshold_mode, percentile)
                retain_fisher = self.compute_fisher_matrix(train_retain_loader, device)
                critical_mask = {k: v for k, v in critical_mask.items() if k in trainable_params}
                retain_fisher = {k: v for k, v in retain_fisher.items() if k in trainable_params}
                print(f"Dynamic update took {time.time() - update_start_time:.2f}s")

            # --- Calculate Forget Gradients (based on RECONSTRUCTION loss) ---
            forget_grads = {}
            unlearned_model.eval()
            num_forget_batches = 0
            # No labels needed
            for inputs, _ in train_forget_loader:
                if inputs.numel() == 0: continue
                inputs = inputs.to(device)
                unlearned_model.zero_grad(set_to_none=True)
                recon_batch, mu, logvar = unlearned_model(inputs)
                # Use RECONSTRUCTION loss ONLY for forget gradients
                forget_recon_loss, _ = vae_loss_function(recon_batch, inputs, mu, logvar)
                loss = forget_recon_loss / inputs.size(0) # Normalize
                loss.backward() # Get gradients w.r.t. reconstruction error on forget data

                for name, param in unlearned_model.named_parameters():
                    if param.grad is not None:
                        grad_clone = param.grad.detach().clone()
                        if name not in forget_grads: forget_grads[name] = grad_clone
                        else: forget_grads[name] += grad_clone
                num_forget_batches += 1
            unlearned_model.zero_grad(set_to_none=True)

            if num_forget_batches > 0:
                for name in forget_grads: forget_grads[name] /= num_forget_batches
            else: print(f"Warning: Epoch {epoch+1}, no batches in train_forget_loader."); forget_grads = {}

            # Project forget gradients (based on recon loss) orthogonal to retain Fisher (based on full loss)
            orthogonal_grads = self.project_orthogonal_gradients(forget_grads, retain_fisher)

            # --- Training Loop on Retain Set ---
            unlearned_model.train()
            epoch_train_loss = 0.0
            num_train_batches = 0
            # No labels needed
            with tqdm(train_retain_loader, desc=f"Epoch {epoch+1}/{epochs} Training", leave=False, dynamic_ncols=True) as pbar:
                for batch_idx, (inputs, _) in enumerate(pbar):
                    if inputs.numel() == 0: continue
                    inputs = inputs.to(device)
                    optimizer.zero_grad(set_to_none=True)

                    recon_batch, mu, logvar = unlearned_model(inputs)
                    # Calculate FULL VAE loss for the retain objective
                    retain_recon_loss, retain_kld_loss = vae_loss_function(recon_batch, inputs, mu, logvar, kl_weight=self.kl_weight)
                    # Normalize the loss components per sample
                    avg_retain_recon = retain_recon_loss / inputs.size(0)
                    avg_retain_kld = retain_kld_loss / inputs.size(0)
                    retain_loss_term = avg_retain_recon + self.kl_weight * avg_retain_kld # Weighted average loss
                    direct_forget_loss_val = 0.0
                    
                    #####
                    if self.delta > 0 and len(train_forget_loader.dataset) > 0:
                        try:
                            forget_inputs, _ = next(iter_train_forget_loader)
                        except StopIteration:
                            iter_train_forget_loader = iter(train_forget_loader) # Reset iterator
                            forget_inputs, _ = next(iter_train_forget_loader)
                        forget_inputs = forget_inputs.to(device)

                        forget_recon_batch, forget_mu, forget_logvar = unlearned_model(forget_inputs)
                        forget_recon_component, _ = vae_loss_function(forget_recon_batch, forget_inputs, forget_mu, forget_logvar)
                        avg_forget_recon_component = forget_recon_component / forget_inputs.size(0)
                        direct_forget_loss_val = avg_forget_recon_component
                    #####
                    
                    # Calculate regularization terms
                    forget_sparsity = 0.0
                    retain_stability = 0.0
                    for name, param in unlearned_model.named_parameters():
                        if param.requires_grad:
                            is_critical = name in critical_mask and critical_mask[name].sum() > 0

                            # Forget Sparsity (L1 on ortho grads of critical params)
                            # Ortho grads derived from forget RECONSTRUCTION loss
                            if is_critical and name in orthogonal_grads:
                                    mask_term = critical_mask[name]
                                    ortho_grad_term = orthogonal_grads[name]
                                    if mask_term.shape == ortho_grad_term.shape:
                                        forget_sparsity += (mask_term * ortho_grad_term).abs().sum().float()
                                    # else: print(f"Warning: Shape mismatch L1 {name}") # Debug if needed

                            # Retain Stability (L2 on non-critical params)
                            if not is_critical:
                                retain_stability += param.float().pow(2).sum()

                    # Combine losses
                    total_loss = (self.alpha * retain_loss_term
                                - self.delta * direct_forget_loss_val 
                                + self.beta * forget_sparsity
                                + self.gamma * retain_stability)

                    scaler.scale(total_loss).backward()
                    # Optional: Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(unlearned_model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()

                    epoch_train_loss += total_loss.item()
                    num_train_batches += 1
                    if batch_idx % 50 == 0: # Update less frequently
                         pbar.set_postfix(loss=f"{total_loss.item():.4f}",
                                          ret_rec=f"{avg_retain_recon.item():.2f}",
                                          ret_kld=f"{avg_retain_kld.item():.2f}")


            avg_epoch_loss = epoch_train_loss / num_train_batches if num_train_batches > 0 else 0

            # --- Validation Step ---
            print(f"Validating on dedicated validation set ({len(val_retain_loader.dataset)} R / {len(val_forget_loader.dataset)} F)...")
            # Evaluate using the new VAE evaluation function
            val_retain_metrics = evaluate_vae(unlearned_model, val_retain_loader, device, self.kl_weight, "Val Retain")
            val_forget_metrics = evaluate_vae(unlearned_model, val_forget_loader, device, self.kl_weight, "Val Forget")

            epoch_end_time = time.time()
            print(f"Epoch {epoch+1}/{epochs} Summary - Train Loss: {avg_epoch_loss:.4f}")
            print(f"  Val Retain -> Recon: {val_retain_metrics['recon_loss']:.2f}, KLD: {val_retain_metrics['kld']:.2f}, Total: {val_retain_metrics['total_loss']:.4f}")
            print(f"  Val Forget -> Recon: {val_forget_metrics['recon_loss']:.2f}, KLD: {val_forget_metrics['kld']:.2f}, Total: {val_forget_metrics['total_loss']:.4f}")
            print(f"  Epoch Time: {epoch_end_time - epoch_start_time:.2f}s")

            # Scheduler Step - step based on validation retain total loss
            scheduler.step(val_retain_metrics['total_loss'])

            # --- Early Stopping Check (Based on minimizing validation retain total loss) ---
            current_val_retain_loss = val_retain_metrics['total_loss']
            current_val_forget_recon = val_forget_metrics['recon_loss'] # Track this too

            if current_val_retain_loss < best_val_retain_loss - 1e-5: # Tolerance
                print(f"-> New best validation retain loss: {current_val_retain_loss:.4f} (was {best_val_retain_loss:.4f})")
                best_val_retain_loss = current_val_retain_loss
                best_forget_recon_at_best_retain = current_val_forget_recon
                best_model_state = copy.deepcopy(unlearned_model.state_dict())
                best_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"  -> No improvement in validation retain loss. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after epoch {epoch+1}.")
                break

        # --- End of Training Loop ---
        unlearning_end_time = time.time()
        print(f"\n--- Unlearning Optimization Finished (Time: {unlearning_end_time - start_time:.2f}s) ---")
        if best_model_state is not None:
            print(f"Loading best model state from Epoch {best_epoch} (Val Retain Loss: {best_val_retain_loss:.4f}, Val Forget Recon: {best_forget_recon_at_best_retain:.2f})")
            unlearned_model.load_state_dict(best_model_state)
        else:
            print("Warning: No improvement detected during training or patience=0. Using model from last epoch.")

        # --- Optional Post-Unlearning Retain Fine-tuning ---
        if post_tune_epochs > 0 and len(train_retain_loader.dataset) > 0:
            post_tune_start_time = time.time()
            print(f"\n--- Post-Unlearning Retain Fine-tuning ({post_tune_epochs} epochs) ---")
            post_lr = post_tune_lr if post_tune_lr is not None else lr / 10.0
            print(f"Post-tuning LR: {post_lr}")
            post_optimizer = torch.optim.Adam(unlearned_model.parameters(), lr=post_lr)
            post_scaler = GradScaler(enabled=amp_enabled)

            unlearned_model.train()
            for pt_epoch in range(post_tune_epochs):
                pt_epoch_loss = 0; num_pt_batches = 0
                # No labels needed
                with tqdm(train_retain_loader, desc=f"Post-Tune Epoch {pt_epoch+1}/{post_tune_epochs}", leave=False, dynamic_ncols=True) as pbar:
                    for inputs, _ in pbar:
                        if inputs.numel() == 0: continue
                        inputs = inputs.to(device)
                        post_optimizer.zero_grad(set_to_none=True)
                        recon_batch, mu, logvar = unlearned_model(inputs)
                        # Fine-tune using the FULL VAE loss
                        recon_loss, kld_loss = vae_loss_function(recon_batch, inputs, mu, logvar)
                        total_loss = recon_loss + self.kl_weight * kld_loss
                        loss = total_loss / inputs.size(0) # Average loss

                        post_scaler.scale(loss).backward()
                        post_scaler.step(post_optimizer)
                        post_scaler.update()
                        pt_epoch_loss += loss.item()
                        num_pt_batches += 1
                        if num_pt_batches % 50 == 0: pbar.set_postfix(loss=f"{loss.item():.4f}")

                avg_pt_loss = pt_epoch_loss / num_pt_batches if num_pt_batches > 0 else 0
                pt_val_retain = evaluate_vae(unlearned_model, val_retain_loader, device, self.kl_weight, "Post-Tune Val Retain")
                pt_val_forget = evaluate_vae(unlearned_model, val_forget_loader, device, self.kl_weight, "Post-Tune Val Forget")
                print(f"Post-Tune Epoch {pt_epoch+1} - Avg Train Loss: {avg_pt_loss:.4f}")
                print(f"  Val Retain -> Recon: {pt_val_retain['recon_loss']:.2f}, KLD: {pt_val_retain['kld']:.2f}")
                print(f"  Val Forget -> Recon: {pt_val_forget['recon_loss']:.2f}, KLD: {pt_val_forget['kld']:.2f}")
            print(f"--- Post-tuning finished (Time: {time.time() - post_tune_start_time:.2f}s) ---")

        total_time = time.time() - start_time
        print(f"\n--- SNOP VAE Process Complete (Total Time: {total_time:.2f}s) ---")
        return unlearned_model

    def validate_unlearning(self, unlearned_model, forget_loader, retain_loader, critical_mask, device='cuda'):
         """Validates unlearning using mechanistic ablation (adapted for VAE)."""
         print("\n--- Mechanistic Validation (VAE) ---")
         # Evaluate unlearned model
         unlearned_retain_metrics = evaluate_vae(unlearned_model, retain_loader, device, self.kl_weight, "Validate Unlearned Retain")
         unlearned_forget_metrics = evaluate_vae(unlearned_model, forget_loader, device, self.kl_weight, "Validate Unlearned Forget")
         print("Unlearned Model Metrics:")
         print(f"  Retain -> Recon: {unlearned_retain_metrics['recon_loss']:.2f}, KLD: {unlearned_retain_metrics['kld']:.2f}")
         print(f"  Forget -> Recon: {unlearned_forget_metrics['recon_loss']:.2f}, KLD: {unlearned_forget_metrics['kld']:.2f}")

         ablated_model = copy.deepcopy(unlearned_model)
         trainable_params = {name for name, param in ablated_model.named_parameters() if param.requires_grad}
         valid_mask = {k: v for k, v in critical_mask.items() if k in trainable_params} # Mask based on forget recon loss
         print(f"Applying ablation using mask derived from {len(valid_mask)} parameter groups.")

         params_ablated = 0
         with torch.no_grad():
             for name, param in ablated_model.named_parameters():
                 if name in valid_mask:
                      mask_tensor = valid_mask[name].to(param.device)
                      if mask_tensor.shape == param.shape:
                           param.data.mul_(1.0 - mask_tensor)
                           params_ablated += 1
                      # else: print(f"Warning: Shape mismatch ablation {name}")

         print(f"Ablated {params_ablated} parameter tensors.")
         print("Evaluating ablated model...")
         ablated_retain_metrics = evaluate_vae(ablated_model, retain_loader, device, self.kl_weight, "Validate Ablated Retain")
         ablated_forget_metrics = evaluate_vae(ablated_model, forget_loader, device, self.kl_weight, "Validate Ablated Forget")
         print("Ablated Model Metrics:")
         print(f"  Retain -> Recon: {ablated_retain_metrics['recon_loss']:.2f}, KLD: {ablated_retain_metrics['kld']:.2f}")
         print(f"  Forget -> Recon: {ablated_forget_metrics['recon_loss']:.2f}, KLD: {ablated_forget_metrics['kld']:.2f}")

         # Metrics focused on Reconstruction Loss changes
         # Forget Effectiveness: How much did forget recon loss increase compared to ablation?
         # Higher is better (closer to ablation effect = good forgetting)
         forget_recon_increase_unlearned = unlearned_forget_metrics['recon_loss'] - unlearned_retain_metrics['recon_loss'] # Base increase due to domain
         forget_recon_increase_ablated = ablated_forget_metrics['recon_loss'] - ablated_retain_metrics['recon_loss']
         forgetting_effectiveness = 0.0
         if forget_recon_increase_ablated > 1e-4: # Avoid division by zero/small numbers
             forgetting_effectiveness = forget_recon_increase_unlearned / forget_recon_increase_ablated

         # Retain Preservation: How close is unlearned retain recon loss to ablated retain recon loss?
         # Closer to 1 is better (unlearning didn't degrade retain much beyond ablation baseline)
         # This interpretation is tricky - ablation *should* hurt retain less than forget.
         # Alternative: Compare unlearned retain loss to ORIGINAL retain loss (before unlearning).
         # Let's compare to ablated for consistency with original SNOP validation paper's idea.
         retain_preservation = 0.0
         if ablated_retain_metrics['recon_loss'] > 1e-8:
            # Ratio of losses (lower is better, so invert?) -> higher is better if we do ablated/unlearned
            retain_preservation = ablated_retain_metrics['recon_loss'] / unlearned_retain_metrics['recon_loss']

         print(f"Forget Recon Effectiveness (vs Ablation): {forgetting_effectiveness:.4f} (Higher is better)")
         print(f"Retain Recon Preservation (vs Ablation): {retain_preservation:.4f} (Closer to 1+ is okay)")

         return {
             'unlearned_forget_recon_loss': unlearned_forget_metrics['recon_loss'],
             'unlearned_retain_recon_loss': unlearned_retain_metrics['recon_loss'],
             'ablated_forget_recon_loss': ablated_forget_metrics['recon_loss'],
             'ablated_retain_recon_loss': ablated_retain_metrics['recon_loss'],
             'forgetting_effectiveness_recon': forgetting_effectiveness,
             'retain_preservation_recon': retain_preservation
         }
