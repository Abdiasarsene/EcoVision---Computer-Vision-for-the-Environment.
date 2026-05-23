from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from utils.config import settings

def get_dataloaders(data_dir=settings.data, batch_size=32):
    # Data Augmentation
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])

    # Dataset
    waste_image = datasets.ImageFolder(data_dir, transform=transform_train)

    # Split
    train_size = int(0.7 * len(waste_image))
    test_size = len(waste_image) - train_size
    train_data, test_data = random_split(waste_image, [train_size, test_size])

    # Transform
    train_data.dataset.transform = transform_train
    test_data.dataset.transform = transform_test

    # Data Loader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    
    classes=waste_image.classes

    return train_loader, test_loader, classes