import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset as TorchDataset # Renamed to avoid conflict
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import copy
import os
import math
from tqdm import tqdm
from typing import Dict, List, Tuple

from resnet import ResNet18ForCIFAR100, test_transform
from utils import set_seed, evaluate, load_model_checkpoint

###############################################
# ParameterPerturber Class (from your provided code)
###############################################
class ParameterPerturber:
    def __init__(
        self,
        model,
        opt, # Optimizer
        device="cuda" if torch.cuda.is_available() else "cpu",
        parameters=None, # Dictionary of parameters
    ):
        self.model = model
        self.opt = opt
        self.device = device
        # self.alpha = None # These seem to be placeholders from an earlier version in your class
        # self.xmin = None

        # print("ParameterPerturber parameters:", parameters) # For debugging
        if parameters is None:
            # Provide default values if none are given, ensure these match your intended defaults
            parameters = {
                "lower_bound": 1.0, # Default to match paper's cap for beta
                "exponent": 1.0,    # Default to match paper's formula (no extra exponent)
                "magnitude_diff": 0, # Unused
                "min_layer": -1,     # Unused in provided core logic
                "max_layer": -1,     # Unused in provided core logic
                "forget_threshold": 0.1, # Unused, selection_weighting is used
                "dampening_constant": 1.0, # Corresponds to lambda
                "selection_weighting": 10.0 # Corresponds to alpha (from paper's perspective)
            }
            print("ParameterPerturber using default parameters as none were provided.")


        self.lower_bound = parameters["lower_bound"]
        self.exponent = parameters["exponent"]
        self.magnitude_diff = parameters["magnitude_diff"]
        self.min_layer = parameters["min_layer"]
        self.max_layer = parameters["max_layer"]
        self.forget_threshold = parameters["forget_threshold"]
        self.dampening_constant = parameters["dampening_constant"] # This is lambda
        self.selection_weighting = parameters["selection_weighting"] # This is alpha_ssd

    def get_layer_num(self, layer_name: str) -> int:
        layer_id = layer_name.split(".")[1]
        if layer_id.isnumeric():
            return int(layer_id)
        else:
            return -1

    def zerolike_params_dict(self, model: nn.Module) -> Dict[str, torch.Tensor]: # nn.Module for type hint
        return dict(
            [
                (k, torch.zeros_like(p, device=p.device))
                for k, p in model.named_parameters()
                if p.requires_grad
            ]
        )

    def calc_importance(self, dataloader: DataLoader) -> Dict[str, torch.Tensor]:
        criterion = nn.CrossEntropyLoss()
        importances = self.zerolike_params_dict(self.model)
        self.model.eval() # Set model to evaluation mode

        if len(dataloader.dataset) == 0:
            print("Warning: calc_importance called with an empty dataset. Returning zero importances.")
            return importances

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Calculating Importances")):
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)

            self.opt.zero_grad() # Use the optimizer passed during initialization
            out = self.model(x)
            loss = criterion(out, y)
            loss.backward()

            for (k, p) in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    if k not in importances: # Should not happen if zerolike_params_dict is correct
                        importances[k] = torch.zeros_like(p, device=p.device)
                    importances[k].data += p.grad.data.clone().pow(2)

        # average over mini batch count (number of batches) - THIS IS A DEVIATION FROM STANDARD FIM
        if len(dataloader) > 0 :
            for _, imp in importances.items():
                imp.data /= float(len(dataloader))
        else: # Handle empty dataloader case for division
            print("Warning: Dataloader for calc_importance was empty. Importances will be zero.")

        return importances

    def modify_weight(
        self,
        # Type hints changed to single dicts based on usage
        original_importance: Dict[str, torch.Tensor],
        forget_importance: Dict[str, torch.Tensor],
    ) -> None:
        with torch.no_grad(): # IMPORTANT: ensure no gradient tracking during direct weight modification
            for (n, p), (oimp_n, oimp), (fimp_n, fimp) in zip(
                self.model.named_parameters(),
                original_importance.items(),
                forget_importance.items(),
            ):
                if not p.requires_grad:
                    continue

                # Ensure importances are on the same device as the parameter
                oimp = oimp.to(p.device)
                fimp = fimp.to(p.device)

                # Synapse Selection with parameter alpha_ssd (selection_weighting)
                # oimp_norm is I_D * alpha_ssd
                oimp_norm = oimp.mul(self.selection_weighting)
                # locations where I_Df > I_D * alpha_ssd
                locations = torch.where(fimp > oimp_norm)

                if locations[0].numel() == 0: # No parameters selected for this layer/parameter tensor
                    continue

                # Synapse Dampening with parameter lambda (dampening_constant)
                # weight = ( (I_D * lambda) / I_Df ) ^ exponent
                # Add epsilon to prevent division by zero if fimp is zero where locations are true
                epsilon = 1e-12
                div_fimp = fimp + epsilon

                weight_numerator = oimp.mul(self.dampening_constant)
                weight_ratio = weight_numerator.div(div_fimp) # (I_D * lambda) / I_Df
                # Handle potential NaNs or Infs from division if epsilon wasn't enough or values are extreme
                weight_ratio = torch.nan_to_num(weight_ratio, nan=1.0, posinf=1.0, neginf=0.0)

                dampening_factor_base = weight_ratio.pow(self.exponent)

                update = dampening_factor_base[locations]

                # Bound by self.lower_bound. This acts as the 'min(..., value)' part.
                # If self.lower_bound is 1.0, it matches paper's min(..., 1).
                # The logic `min_locs = torch.where(update > self.lower_bound)` and `update[min_locs] = self.lower_bound`
                # effectively means update = min(update, self.lower_bound)
                update = torch.minimum(update, torch.tensor(self.lower_bound, device=update.device, dtype=update.dtype))

                # Apply the update
                p_data_selected = p.data[locations]
                p.data[locations] = p_data_selected.mul(update)
###############################################

def evaluate_unlearning_performance(original_model, unlearned_model,
                                   train_retain_loader_eval, train_forget_loader_eval, # Dataloaders for eval
                                   test_retain_loader_eval, test_forget_loader_eval,
                                   device):
    print("\n--- Evaluating Original Model ---")
    # original_train_retain_acc = evaluate(original_model, train_retain_loader_eval, device, "Original on Train Retain")
    # original_train_forget_acc = evaluate(original_model, train_forget_loader_eval, device, "Original on Train Forget")
    original_test_retain_acc  = evaluate(original_model, test_retain_loader_eval,  device, "Original on Test Retain")
    original_test_forget_acc  = evaluate(original_model, test_forget_loader_eval,  device, "Original on Test Forget")

    print("\n--- Evaluating Unlearned Model ---")
    # unlearned_train_retain_acc = evaluate(unlearned_model, train_retain_loader_eval, device, "Unlearned on Train Retain")
    # unlearned_train_forget_acc = evaluate(unlearned_model, train_forget_loader_eval, device, "Unlearned on Train Forget")
    unlearned_test_retain_acc  = evaluate(unlearned_model, test_retain_loader_eval,  device, "Unlearned on Test Retain")
    unlearned_test_forget_acc  = evaluate(unlearned_model, test_forget_loader_eval,  device, "Unlearned on Test Forget")

    print("\n--- Performance Summary ---")
    print(f"Original Model:")
    # print(f"  Train Retain Accuracy: {original_train_retain_acc:.2f}%")
    # print(f"  Train Forget Accuracy: {original_train_forget_acc:.2f}%")
    print(f"  Test Retain Accuracy:  {original_test_retain_acc:.2f}%")
    print(f"  Test Forget Accuracy:  {original_test_forget_acc:.2f}%")

    print(f"\nUnlearned Model:")
    # print(f"  Train Retain Accuracy: {unlearned_train_retain_acc:.2f}%")
    # print(f"  Train Forget Accuracy: {unlearned_train_forget_acc:.2f}%")
    print(f"  Test Retain Accuracy:  {unlearned_test_retain_acc:.2f}%")
    print(f"  Test Forget Accuracy:  {unlearned_test_forget_acc:.2f}%")

    if original_test_forget_acc > 0:
        forgetting_efficacy = (original_test_forget_acc - unlearned_test_forget_acc) / original_test_forget_acc
        print(f"\nForgetting Efficacy (Test Forget Acc Drop Ratio): {forgetting_efficacy:.4f}")
    else:
        print("\nForgetting Efficacy (Test Forget Acc Drop Ratio): N/A (Original forget acc is 0)")

    if original_test_retain_acc > 0:
        retain_performance_ratio = unlearned_test_retain_acc / original_test_retain_acc
        print(f"Retain Performance Ratio (Test Retain Acc Ratio): {retain_performance_ratio:.4f}")
    else:
        print(f"Retain Performance Ratio (Test Retain Acc Ratio): N/A (Original retain acc is 0)")


def main():
    set_seed(0) # Consistent with original script's seed
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_classes = 100
    
    # --- Model Loading ---
    model_path_final = 'resnet18_cifar100_final.pth'
    # SSD script only uses one path, so model_path_best is None
    model = load_model_checkpoint(ResNet18ForCIFAR100, None, model_path_final, device)
    # The utility function load_model_checkpoint prints warnings if no model is loaded.

    # Create a dummy optimizer, required by ParameterPerturber
    # Its state won't be used beyond zero_grad for FIM calculation
    dummy_optimizer = optim.SGD(model.parameters(), lr=0.001)

    print("Loading CIFAR-100 dataset...")
    full_trainset_orig = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=test_transform)
    testset_orig = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)

    class_to_forget = 54

    print(f"Targeting class to forget: {class_to_forget}")

    # Indices based on original datasets
    # Parallelize the dataset splitting using numpy and concurrent processing
    import concurrent.futures

    def extract_indices(dataset, target_class, condition="equal"):
        """Extract indices based on label condition (equal or not equal to target class)"""
        labels = np.array([label for _, label in dataset])
        if condition == "equal":
            return np.where(labels == target_class)[0]
        else:  # not equal
            return np.where(labels != target_class)[0]

    # Use ThreadPoolExecutor to parallelize the work
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit tasks for train dataset
        train_forget_future = executor.submit(extract_indices, full_trainset_orig, class_to_forget, "equal")
        train_retain_future = executor.submit(extract_indices, full_trainset_orig, class_to_forget, "not_equal")
        
        # Submit tasks for test dataset
        test_forget_future = executor.submit(extract_indices, testset_orig, class_to_forget, "equal")
        test_retain_future = executor.submit(extract_indices, testset_orig, class_to_forget, "not_equal")
        
        # Get results
        train_forget_indices = train_forget_future.result()
        train_retain_indices = train_retain_future.result()
        test_forget_indices = test_forget_future.result()
        test_retain_indices = test_retain_future.result()

    fraction_to_forget = 0.1
    fraction_to_retain = 0.1
    
    train_forget_indices = np.random.choice(train_forget_indices, int(len(train_forget_indices) * fraction_to_forget), replace=False)
    train_retain_indices = np.random.choice(train_retain_indices, int(len(train_retain_indices) * fraction_to_retain), replace=False)

    # test_forget_indices  = np.random.choice(test_forget_indices, int(len(test_forget_indices) * fraction_to_forget), replace=False)
    # test_retain_indices  = np.random.choice(test_retain_indices, int(len(test_retain_indices) * fraction_to_retain), replace=False)

    # Subsets for FIM calculation (using wrapped dataset)
    train_forget_subset_fim = Subset(full_trainset_orig, train_forget_indices)
    full_trainset_orig = Subset(full_trainset_orig, np.concatenate((train_retain_indices, train_forget_indices)))
    # testset_orig = Subset(testset_orig, np.concatenate((test_retain_indices, test_forget_indices)))

    # Subsets for evaluation (using original datasets - eval function handles both (x,y) and (x,_,y) )
    train_retain_subset_eval = Subset(full_trainset_orig, train_retain_indices)
    train_forget_subset_eval = Subset(full_trainset_orig, train_forget_indices) # For evaluating on train forget samples
    test_forget_subset_eval  = Subset(testset_orig, test_forget_indices)
    test_retain_subset_eval  = Subset(testset_orig, test_retain_indices)


    batch_size = 64
    # Loader for I_D (full original training data, wrapped)
    full_original_train_loader_fim = DataLoader(full_trainset_orig, batch_size=batch_size, shuffle=False)
    # Loader for I_Df (forget portion of original training data, wrapped)
    train_forget_loader_fim = DataLoader(train_forget_subset_fim, batch_size=batch_size, shuffle=False)

    # Loaders for evaluation (can use original datasets as evaluate handles it)
    train_retain_loader_eval = DataLoader(train_retain_subset_eval, batch_size=batch_size, shuffle=False)
    train_forget_loader_eval = DataLoader(train_forget_subset_eval, batch_size=batch_size, shuffle=False)
    test_forget_loader_eval  = DataLoader(test_forget_subset_eval,  batch_size=batch_size, shuffle=False)
    test_retain_loader_eval  = DataLoader(test_retain_subset_eval,  batch_size=batch_size, shuffle=False)


    print(f"Full original train set size (for I_D): {len(full_trainset_orig)}")
    print(f"Train forget set size (for I_Df): {len(train_forget_subset_fim)}")

    # --- SSD Unlearning using ParameterPerturber ---
    # These parameters need to be set based on your "original code's" intended usage
    ssd_params = {
        "lower_bound": 1.0,        # If paper's min(..., 1) is desired
        "exponent": 1.0,           # If paper's no extra exponent is desired
        "magnitude_diff": 0,       # Unused
        "min_layer": -1,           # Unused
        "max_layer": -1,           # Unused
        "forget_threshold": 0.1,   # Unused by core logic shown
        "dampening_constant": 1.0, # This is "lambda" from paper (e.g., 1.0)
        "selection_weighting": 10.0# This is "alpha" from paper (e.g., 10.0 for CIFAR)
    }
    print(f"\nUsing ParameterPerturber with parameters: {ssd_params}")

    # Create a copy of the model for unlearning
    model_to_unlearn = copy.deepcopy(model)
    import time
    perturber = ParameterPerturber(model_to_unlearn, dummy_optimizer, device, parameters=ssd_params)

    starting_time = time.time()
    
    print("Calculating I_D (original importance)...")
    importance_D = perturber.calc_importance(full_original_train_loader_fim)
    print("Calculating I_Df (forget importance)...")
    importance_Df = perturber.calc_importance(train_forget_loader_fim)

    print("Modifying weights using ParameterPerturber...")
    perturber.modify_weight(importance_D, importance_Df)
    unlearned_model = model_to_unlearn # model_to_unlearn has been modified in-place by perturber
    elapsed_time = time.time() - starting_time
    print(f"Unlearning took {elapsed_time:.2f} seconds.")
    # --- Evaluation ---
    evaluate_unlearning_performance(model, unlearned_model,
                                   train_retain_loader_eval, train_forget_loader_eval,
                                   test_retain_loader_eval, test_forget_loader_eval,
                                   device)

    unlearned_model_path = 'resnet18_cifar100_parameter_perturber_unlearned.pth'
    torch.save(unlearned_model.state_dict(), f'resnet18_cifar100_ssd.pth')

    print(f"\nUnlearned model saved as '{unlearned_model_path}'")

if __name__ == "__main__":
    main()