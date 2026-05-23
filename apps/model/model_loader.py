# app/model/model_loader.py
import onnxruntime as ort
from utils.config import settings

classes=settings.classes
def load_model(weights_path=settings.resnet18_waste):
    session = ort.InferenceSession(weights_path, providers=["CPUExecutionProvider"])
    return session, classes