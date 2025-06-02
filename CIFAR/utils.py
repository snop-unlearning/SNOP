import torch
import numpy as np
import random
import os
import json
from torch.utils.data import Subset, Dataset, DataLoader
from tqdm import tqdm

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

# --- Dataset Helpers ---
def get_dataset_targets(dataset_obj: Dataset) -> list:
    """
    Gets targets from a PyTorch Dataset or Subset object.
    Handles nested Subsets by traversing to the base dataset.
    """
    if isinstance(dataset_obj, Subset):
        if not dataset_obj.indices:
            return []
        
        current_dataset = dataset_obj.dataset
        effective_indices = list(dataset_obj.indices) 

        while isinstance(current_dataset, Subset):
            if not current_dataset.indices:
                return []
            
            parent_indices_list = list(current_dataset.indices)
            new_effective_indices = []
            for idx_in_current_subset in effective_indices:
                if 0 <= idx_in_current_subset < len(parent_indices_list):
                    new_effective_indices.append(parent_indices_list[idx_in_current_subset])
            
            effective_indices = new_effective_indices
            if not effective_indices:
                return []
            
            current_dataset = current_dataset.dataset

        if hasattr(current_dataset, 'targets'):
            targets_attr = current_dataset.targets
            valid_base_indices = [i for i in effective_indices if 0 <= i < len(targets_attr)]
            if not valid_base_indices:
                return []

            if isinstance(targets_attr, torch.Tensor):
                return targets_attr[torch.tensor(valid_base_indices, dtype=torch.long)].tolist()
            elif isinstance(targets_attr, np.ndarray):
                return targets_attr[np.array(valid_base_indices, dtype=np.int64)].tolist()
            elif isinstance(targets_attr, list):
                return [targets_attr[i] for i in valid_base_indices]
        
        try:
            valid_base_indices_for_iteration = [i for i in effective_indices if 0 <= i < len(current_dataset)]
            if not valid_base_indices_for_iteration: return []
            return [current_dataset[i][1] for i in valid_base_indices_for_iteration]
        except Exception:
            return []

    elif hasattr(dataset_obj, 'targets'):
        targets_attr = dataset_obj.targets
        if isinstance(targets_attr, torch.Tensor):
            return targets_attr.tolist()
        elif isinstance(targets_attr, np.ndarray):
            return targets_attr.tolist()
        elif isinstance(targets_attr, list):
            return list(targets_attr)

    try:
        if len(dataset_obj) == 0: return []
        return [label for _, label in dataset_obj]
    except Exception:
        return []


def split_indices(dataset, classes_to_forget, retain_fraction=1.0, forget_fraction=1.0, get_targets_func=get_dataset_targets):
    """
    Splits dataset indices into forget and retain sets based on specified classes.
    Allows sampling fractions for both sets.
    """
    labels = np.array(get_targets_func(dataset))
    if labels.ndim == 0 or len(labels) == 0:
        return [], []

    forget_mask = np.isin(labels, list(classes_to_forget))
    retain_mask = ~forget_mask

    forget_indices_all = np.where(forget_mask)[0].tolist()
    retain_indices_all = np.where(retain_mask)[0].tolist()

    sample_forget_size = int(forget_fraction * len(forget_indices_all))
    sample_retain_size = int(retain_fraction * len(retain_indices_all))
    
    sample_forget_size = max(0, min(sample_forget_size, len(forget_indices_all)))
    sample_retain_size = max(0, min(sample_retain_size, len(retain_indices_all)))

    forget_indices_sampled = []
    if sample_forget_size > 0 and len(forget_indices_all) > 0: # Ensure there are indices to sample from
        forget_indices_sampled = np.random.choice(forget_indices_all, size=sample_forget_size, replace=False).tolist()
    
    retain_indices_sampled = []
    if sample_retain_size > 0 and len(retain_indices_all) > 0: # Ensure there are indices to sample from
        retain_indices_sampled = np.random.choice(retain_indices_all, size=sample_retain_size, replace=False).tolist()
        
    return forget_indices_sampled, retain_indices_sampled

# --- Evaluation ---
def evaluate(model, dataloader, device, desc="Evaluating") -> float:
    """ Evaluate model accuracy. Returns accuracy as a float (0.0 to 1.0). """
    model.eval()
    correct = 0
    total = 0
    
    if not hasattr(dataloader, 'dataset') or len(dataloader.dataset) == 0:
        return 0.0
        
    with torch.no_grad():
        with tqdm(dataloader, desc=desc, leave=False, dynamic_ncols=True) as pbar:
            for inputs, labels in pbar:
                if inputs.numel() == 0: continue
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.long()).sum().item()
    accuracy = correct / total if total > 0 else 0.0
    return accuracy

# --- Model Loading ---
def load_model_checkpoint(model_constructor, model_path_best, model_path_final, device, *args, **kwargs):
    """
    Loads a model checkpoint.
    model_constructor: A function or class that returns a new model instance (e.g., ResNet18ForCIFAR100).
    *args, **kwargs: Arguments to pass to the model_constructor.
    """
    model = model_constructor(*args, **kwargs)
    
    loaded_successfully = False

    def _load_state(checkpoint_data, model_to_load):
        if isinstance(checkpoint_data, dict) and 'model_state_dict' in checkpoint_data:
            model_to_load.load_state_dict(checkpoint_data['model_state_dict'])
        elif isinstance(checkpoint_data, dict) and 'state_dict' in checkpoint_data:
            model_to_load.load_state_dict(checkpoint_data['state_dict'])
        else:
            model_to_load.load_state_dict(checkpoint_data)

    if model_path_best and os.path.exists(model_path_best) and os.path.getsize(model_path_best) > 0:
        print(f"Attempting to load best checkpoint model from {model_path_best}")
        try:
             checkpoint = torch.load(model_path_best, map_location=device)
             _load_state(checkpoint, model)
             print(f"Successfully loaded model from {model_path_best}")
             loaded_successfully = True
        except Exception as e:
             print(f"Error loading {model_path_best}: {e}.")
    
    if not loaded_successfully and model_path_final and os.path.exists(model_path_final) and os.path.getsize(model_path_final) > 0:
         print(f"Attempting to load final model from {model_path_final}")
         try:
              checkpoint = torch.load(model_path_final, map_location=device)
              _load_state(checkpoint, model)
              print(f"Successfully loaded model from {model_path_final}")
              loaded_successfully = True
         except Exception as e:
              print(f"Error loading {model_path_final}: {e}.")

    if not loaded_successfully:
        print("Warning: No model checkpoint found or loaded successfully. Using initial model weights.")
    
    model.to(device)
    return model

# --- CIFAR-100 Class Names ---
def load_cifar100_classes(json_path="CIFAR/cifar_100_class.json"):
    """Loads CIFAR-100 class names from a JSON file."""
    # Adjust path if utils.py is not in the project root.
    # Assuming execution from project root or json_path is absolute.
    if not os.path.isabs(json_path) and not json_path.startswith("CIFAR/"):
        # Heuristic: if it's a simple filename, assume it's in CIFAR dir
        if '/' not in json_path and '\\' not in json_path:
             potential_path = os.path.join("CIFAR", json_path)
             if os.path.exists(potential_path):
                 json_path = potential_path

    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Class file not found at {json_path}")
        return [f"class_{i}" for i in range(100)] 
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}")
        return [f"class_{i}" for i in range(100)]