import torch
import torch.nn as nn
import torch.optim as optim
from src.data_loader import get_dataloaders
from src.model import build_resnet18
from src.evaluate import evaluate
from utils.config import settings

def train_model(data_dir=settings.data, epochs=5, lr=1e-4, save_path=settings.resnet_model):
    train_loader, test_loader, classes = get_dataloaders(data_dir)
    model = build_resnet18(num_classes=len(classes))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # Évaluation
    evaluate(model, test_loader, classes)

    # Sauvegarde des poids
    torch.save(model.state_dict(), save_path)

    # Export ONNX
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, 
        dummy_input, 
        settings.resnet_onnx_model,
        input_names=["input"], 
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11
    )
    print("Done")
