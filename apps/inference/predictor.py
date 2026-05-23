# app/inference/predictor.py
from torchvision import transforms
from PIL import Image
import numpy as np

def predict(session, img_file, classes):
    # Transform
    transform = transforms.Compose([
        transforms.Resize(224 ,224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])
    
    # Upload images
    img = Image.open(img_file).convert("RGB")
    img = transform(img).unsqueeze(0).np()
    
    # ONNX Runtime
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    outputs =session.run([output_name], {input_name: img})
    pred = np.argmax(outputs[0], axis=1)[0 ]
    return classes[pred]