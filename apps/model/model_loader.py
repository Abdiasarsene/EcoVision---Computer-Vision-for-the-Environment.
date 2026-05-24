# app/model/model_loader.py
import onnxruntime as ort
from utils.config import settings

classes=['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
def load_model(weights_path=settings.resnet_onnx_model):
    session = ort.InferenceSession(weights_path, providers=["CPUExecutionProvider"])
    return session, classes