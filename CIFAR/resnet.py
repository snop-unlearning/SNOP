import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

class ResNet18ForCIFAR100(nn.Module):
    def __init__(self):
        super(ResNet18ForCIFAR100, self).__init__()
        # Load pre-trained ResNet18
        self.resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Get the number of features in the last layer
        num_ftrs = self.resnet.fc.in_features
        
        # Replace the original fc layer with a new sequence:
        # Original output (1000) -> New output (100)
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 100)
        )
    
    def forward(self, x):
        return self.resnet(x)

class ResNet18ForCIFAR10(nn.Module):
    def __init__(self):
        super(ResNet18ForCIFAR10, self).__init__()
        # Load pre-trained ResNet18
        self.resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Get the number of features in the last layer
        num_ftrs = self.resnet.fc.in_features
        
        # Replace the original fc layer with a new sequence:
        # Original output (1000) -> New output (10)
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 10)
        )
    
    def forward(self, x):
        return self.resnet(x)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(224),  # ResNet requires 224x224 input
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_transform = transforms.Compose([
    transforms.Resize(224),  # ResNet requires 224x224 input
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])