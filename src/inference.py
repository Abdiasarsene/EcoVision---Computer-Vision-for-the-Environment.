import torch
from torchvision import transforms
from PIL import Image
from model import build_resnet18

def predict(img_path, weights_path="../models_savior/resnet18_waste.pth", classes=None):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])

    img = Image.open(img_path)
    img = transform(img).unsqueeze(0)

    model = build_resnet18(num_classes=len(classes))
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, dim=1).item()

    return classes[pred]