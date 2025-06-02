import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import copy
from tqdm import tqdm
from resnet import ResNet18ForCIFAR100, train_transform, test_transform
from utils import set_seed, evaluate

# Set device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")


# Load CIFAR-100 dataset
trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=train_transform)
testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=test_transform)

# Create data loaders

# take 10% of the dataset for training
# trainset.data = trainset.data[:(len(trainset.data) // 10)]

trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
testloader = DataLoader(testset, batch_size=128, shuffle=False)

# Create and load the model
model = ResNet18ForCIFAR100().to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()

# Uncomment the next line if you want to ignore a specific class (e.g., class index 54)
# criterion = nn.CrossEntropyLoss(ignore_index = 54)

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)

# Learning rate scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Track metrics
    train_losses = []
    train_accs = []
    test_accs = []
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        
        for inputs, labels in tqdm(trainloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data).item()
            total += labels.size(0)

        scheduler.step()
        
        epoch_loss = running_loss / len(trainloader.dataset)
        epoch_acc = running_corrects / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # Evaluation phase
        model.eval()
        # The 'best_epoch' variable previously declared here (at old line 103) was local
        # to the epoch's evaluation phase and did not affect the 'best_epoch'
        # variable that tracks the best performing epoch for the overall training.
        # The relevant 'best_epoch' is updated at line 124 (original numbering).
        test_acc = evaluate(model, testloader, device, desc=f"Epoch {epoch+1}/{num_epochs} Test Eval")
        test_accs.append(test_acc)
        
        print(f'Test Acc: {test_acc:.4f}')
        
        # Save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'acc': test_acc,
            }, 'resnet18_cifar100_best_54_10%.pth')
            # }, 'resnet18_cifar100_best_54.pth')
    
    print(f'Best test Acc: {best_acc:.4f} on epoch {best_epoch+1}')
    
    # Load the best model weights
    model.load_state_dict(best_model_wts)
    
    # Save final metrics
    return model, train_losses, train_accs, test_accs

if __name__ == '__main__':
    set_seed(42) # Setting default seed
    # Train the model
    print("Starting training...")
    model, train_losses, train_accs, test_accs = train_model(
        model, criterion, optimizer, scheduler, num_epochs=50
    )

    # Save the final model
    # torch.save(model.state_dict(), 'resnet18_cifar100_final_54.pth')
    torch.save(model.state_dict(), 'resnet18_cifar100_54_10%.pth')

    # Plot the training curves
    # plt.figure(figsize=(12, 4))

    # plt.subplot(1, 2, 1)
    # plt.plot(train_losses)
    # plt.title('Training Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')

    # plt.subplot(1, 2, 2)
    # plt.plot(train_accs, label='Train')
    # plt.plot(test_accs, label='Test')
    # plt.title('Accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend()

    # plt.tight_layout()
    # plt.savefig('training_curves.png')
    # plt.show()

# Evaluate on test set
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    
    class_correct = [0] * 100
    class_total = [0] * 100
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1
    
    # Print per-class accuracy
    for i in range(100):
        print(f'Accuracy of classes {i} : {100 * class_correct[i] / class_total[i]:.2f}%')
    
    print(f'Overall Accuracy: {100 * correct / total:.2f}%')

    print("\nEvaluating model on test set:")
    evaluate_model(model, testloader)
    # torch.save(model.state_dict(), 'resnet18_cifar100_final.pth')
    print("Model saved to resnet18_cifar100_final.pth")