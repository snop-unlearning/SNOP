import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
import os
import argparse
import copy

# Assuming resnet.py is in the same directory
from resnet import ResNet18ForCIFAR100, test_transform
from utils import set_seed, get_dataset_targets, load_model_checkpoint

# --- Model Constants ---
NUM_CLASSES = 100

# --- Helper functions for Data ---
def split_indices_by_class(dataset, classes_to_select, is_forget_set=True):
    """Splits dataset indices into selected (forget/specific class) and remaining (retain/other classes)."""
    labels = np.array(get_dataset_targets(dataset))
    if labels.ndim == 0 or len(labels) == 0: return [], []

    mask = np.isin(labels, list(classes_to_select))
    selected_indices = np.where(mask)[0].tolist()
    remaining_indices = np.where(~mask)[0].tolist()

    if is_forget_set: # Typical use for forget/retain
        return selected_indices, remaining_indices # forget_indices, retain_indices
    else: # Use for selecting specific classes vs all others
        return selected_indices, remaining_indices


# --- MIA Helper Functions ---
def get_classification_losses(model, dataloader, device, desc="Calculating Losses"):
    model.eval()
    all_losses = []
    criterion = nn.CrossEntropyLoss(reduction='none') 

    if not hasattr(dataloader, 'dataset') or len(dataloader.dataset) == 0:
        # print(f"Warning: Dataloader for '{desc}' is empty or invalid. Returning no losses.")
        return np.array([])

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc=desc, leave=False, dynamic_ncols=True):
            if inputs.numel() == 0: continue
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs) 

            instance_losses = criterion(outputs, targets.long()) 
            all_losses.extend(instance_losses.cpu().numpy())
    return np.array(all_losses)

def run_mia_attack(member_losses, non_member_losses, attack_test_size=0.3):
    if len(member_losses) == 0 or len(non_member_losses) == 0:
        # print("Warning: Empty member or non-member losses for MIA. Attack cannot proceed.")
        return 0.0, 0.0

    X_members = member_losses.reshape(-1, 1)
    y_members = np.ones(len(member_losses))
    X_non_members = non_member_losses.reshape(-1, 1)
    y_non_members = np.zeros(len(non_member_losses))

    X_attack = np.concatenate((X_members, X_non_members))
    y_attack = np.concatenate((y_members, y_non_members))

    if len(np.unique(y_attack)) < 2:
        # print("Warning: MIA attack dataset has only one class. Attack cannot proceed effectively.")
        return accuracy_score(y_attack, y_attack), 0.5

    # SEED will be set by set_seed() from utils
    X_train_mia, X_test_mia, y_train_mia, y_test_mia = train_test_split(
        X_attack, y_attack, test_size=attack_test_size, stratify=y_attack, random_state=42 # Use the default seed value
    )

    if len(X_train_mia) == 0 or len(X_test_mia) == 0:
        # print("Warning: MIA train or test split resulted in empty set. Attack cannot proceed.")
        return 0.0, 0.0

    # SEED will be set by set_seed() from utils
    mia_classifier = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced') # Use the default seed value
    mia_classifier.fit(X_train_mia, y_train_mia)

    y_pred_mia = mia_classifier.predict(X_test_mia)
    y_proba_mia = mia_classifier.predict_proba(X_test_mia)[:, 1]

    accuracy = accuracy_score(y_test_mia, y_pred_mia)
    try:
        auc = roc_auc_score(y_test_mia, y_proba_mia)
    except ValueError:
        # print("Warning: AUC calculation failed (likely due to single class in MIA test labels). Setting AUC to 0.5.")
        auc = 0.5
    return accuracy, auc

# --- Main MIA Execution Logic ---
def main():
    parser = argparse.ArgumentParser(description="Membership Inference Attack on CIFAR ResNet models.")
    parser.add_argument("--original_model_path", type=str, default="resnet18_cifar100_best.pth", help="Path to the original ResNet model (.pth).")
    parser.add_argument("--unlearned_model_path", type=str, default="resnet18_cifar100_snop.pth", help="Path to the unlearned ResNet model (.pth).")
    parser.add_argument("--classes_to_forget", type=str, default="54", help="Comma-separated list of classes to forget (e.g., '54,20'). CIFAR-100 has classes 0-99.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for data loaders.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for data loaders.")
    args = parser.parse_args()

    set_seed(42) # Default seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    classes_to_forget = [int(c) for c in args.classes_to_forget.split(',')]
    print(f"Classes designated as 'forgotten': {classes_to_forget}")

    print("\n--- Data Preparation (CIFAR-100) ---")
    try:
        full_trainset_cifar = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=test_transform)
        full_testset_cifar = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    except Exception as e:
        print(f"Error loading CIFAR-100 dataset: {e}"); exit(1)

    train_forget_indices, train_retain_indices = split_indices_by_class(full_trainset_cifar, classes_to_forget, is_forget_set=True)
    train_forget_subset = Subset(full_trainset_cifar, train_forget_indices)
    train_retain_subset = Subset(full_trainset_cifar, train_retain_indices)

    test_forget_indices, test_retain_indices = split_indices_by_class(full_testset_cifar, classes_to_forget, is_forget_set=True)
    test_forget_subset = Subset(full_testset_cifar, test_forget_indices)
    test_retain_subset = Subset(full_testset_cifar, test_retain_indices)

    print(f"Original Training Set: Total={len(full_trainset_cifar)}, Forget_Class_Samples={len(train_forget_subset)}, Retain_Class_Samples={len(train_retain_subset)}")
    print(f"Original Test Set: Total={len(full_testset_cifar)}, Forget_Class_Samples={len(test_forget_subset)}, Retain_Class_Samples={len(test_retain_subset)}")

    pin_memory = True if device.type == 'cuda' else False
    
    loader_original_train_all = DataLoader(full_trainset_cifar, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
    loader_original_test_all = DataLoader(full_testset_cifar, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)

    loader_train_forget = DataLoader(train_forget_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
    loader_train_retain = DataLoader(train_retain_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
    loader_test_forget = DataLoader(test_forget_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
    loader_test_retain = DataLoader(test_retain_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)

    print("\n--- Loading ResNet Models ---")
    try:
        original_model = load_model_checkpoint(ResNet18ForCIFAR100, args.original_model_path, None, device)
        # load_model_checkpoint will print success/failure. We assume ResNet18ForCIFAR100 is the constructor.
        # No second path (like _final.pth) is provided for MIA script's direct loading.
        print(f"Original model loading attempted from: {args.original_model_path}")

        unlearned_model = load_model_checkpoint(ResNet18ForCIFAR100, args.unlearned_model_path, None, device)
        print(f"Unlearned model loading attempted from: {args.unlearned_model_path}")
    except FileNotFoundError as e:
        print(e)
        exit(1)
    except Exception as e:
        print(f"An error occurred during model loading: {e}")
        exit(1)

    print("\n--- Calculating Classification Losses ---")
    
    
    losses_orig_model_on_train_all = get_classification_losses(original_model, loader_original_train_all, device, "OrigModel on AllTrain")
    losses_orig_model_on_test_all = get_classification_losses(original_model, loader_original_test_all, device, "OrigModel on AllTest")

    losses_unlearn_model_on_train_forget = get_classification_losses(unlearned_model, loader_train_forget, device, "UnlearnModel on TrainForget")
    losses_unlearn_model_on_train_retain = get_classification_losses(unlearned_model, loader_train_retain, device, "UnlearnModel on TrainRetain")
    losses_unlearn_model_on_test_forget = get_classification_losses(unlearned_model, loader_test_forget, device, "UnlearnModel on TestForget")
    losses_unlearn_model_on_test_retain = get_classification_losses(unlearned_model, loader_test_retain, device, "UnlearnModel on TestRetain")

    print("\n\n--- Membership Inference Attack Results ---")

    print("\nScenario 1: Original Model (All Train Data vs. All Test Data)")
    if len(losses_orig_model_on_train_all) > 0 and len(losses_orig_model_on_test_all) > 0 :
        acc1, auc1 = run_mia_attack(losses_orig_model_on_train_all, losses_orig_model_on_test_all)
        print(f"  MIA Accuracy: {acc1:.4f}, MIA AUC: {auc1:.4f}")
    else:
        print("  Skipping MIA for Scenario 1 due to empty loss arrays.")

    print("\nScenario 2: Unlearned Model (FORGOTTEN Train Data vs. FORGOTTEN Test Data)")
    if len(losses_unlearn_model_on_train_forget) > 0 and len(losses_unlearn_model_on_test_forget) > 0:
        acc2, auc2 = run_mia_attack(losses_unlearn_model_on_train_forget, losses_unlearn_model_on_test_forget)
        print(f"  MIA Accuracy: {acc2:.4f}, MIA AUC: {auc2:.4f}")
        print("  (Lower values are better, indicating successful forgetting)")
    else:
        print("  Skipping MIA for Scenario 2 due to empty loss arrays.")

    print("\nScenario 3: Unlearned Model (RETAINED Train Data vs. RETAINED Test Data)")
    if len(losses_unlearn_model_on_train_retain) > 0 and len(losses_unlearn_model_on_test_retain) > 0:
        acc3, auc3 = run_mia_attack(losses_unlearn_model_on_train_retain, losses_unlearn_model_on_test_retain)
        print(f"  MIA Accuracy: {acc3:.4f}, MIA AUC: {auc3:.4f}")
        print("  (Higher values are better, indicating utility preservation on retained data)")
    else:
        print("  Skipping MIA for Scenario 3 due to empty loss arrays.")

    print("\nScenario 4: Unlearned Model (All Original Train Data vs. All Original Test Data)")
    
    # Initialize as empty numpy arrays to handle cases where one part might be empty
    losses_unlearn_model_on_train_all_list = []
    if len(losses_unlearn_model_on_train_forget) > 0:
        losses_unlearn_model_on_train_all_list.append(losses_unlearn_model_on_train_forget)
    if len(losses_unlearn_model_on_train_retain) > 0:
        losses_unlearn_model_on_train_all_list.append(losses_unlearn_model_on_train_retain)
    
    losses_unlearn_model_on_test_all_list = []
    if len(losses_unlearn_model_on_test_forget) > 0:
        losses_unlearn_model_on_test_all_list.append(losses_unlearn_model_on_test_forget)
    if len(losses_unlearn_model_on_test_retain) > 0:
        losses_unlearn_model_on_test_all_list.append(losses_unlearn_model_on_test_retain)

    # Concatenate only if the list of arrays to concatenate is not empty
    losses_unlearn_model_on_train_all = np.concatenate(losses_unlearn_model_on_train_all_list) if losses_unlearn_model_on_train_all_list else np.array([])
    losses_unlearn_model_on_test_all = np.concatenate(losses_unlearn_model_on_test_all_list) if losses_unlearn_model_on_test_all_list else np.array([])


    if losses_unlearn_model_on_train_all.size > 0 and losses_unlearn_model_on_test_all.size > 0:
         acc4, auc4 = run_mia_attack(losses_unlearn_model_on_train_all, losses_unlearn_model_on_test_all)
         print(f"  MIA Accuracy: {acc4:.4f}, MIA AUC: {auc4:.4f}")
         print("  (Compare with Scenario 1; ideally lower if unlearning D_f had an effect)")
    else:
        print("  Skipping MIA for Scenario 4 due to empty combined loss arrays.")

    print("\n--- MIA Summary ---")
    print(f"A perfect MIA has Accuracy/AUC = 1.0. Random guessing is Accuracy/AUC = 0.5.")
    print(f"Effective unlearning of forgotten data (D_f) should result in Scenario 2 Acc/AUC close to 0.5.")
    print(f"Good utility on retained data (D_r) should result in Scenario 3 Acc/AUC significantly above 0.5.")

if __name__ == "__main__":
    main()