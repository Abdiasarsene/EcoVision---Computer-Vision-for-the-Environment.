from flask import Flask, render_template, request
from apps.model.model_loader import load_model
from apps.inference.predictor import predict
from apps.inference.explainability import gradcam, overlay_gradcam
from PIL import Image
import torch
from torchvision import transforms
from utils.config import settings

app=Flask(__name__)
session, classes = load_model()
model = settings.resnet_waste_pth

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    gradcam_img = None
    if request.method == "POST":
        file = request.files["images"]
        if file:
            # Prediction
            prediction = predict(session, file, classes)
            
            # Grad-CAM
            transform = transforms.Compose([
                transforms.Resize(224,224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
            ])
            img = Image.open(file.strean).convert("RGB")
            img_tensor =transform(img).unsqueeze(0)
            
            pred_class = classes.index(prediction)
            gradcam_map = gradcam(model, img_tensor, model.layer4[1].conv2, classes, pred_class)
            gradcam_img = overlay_gradcam(file.stream, gradcam_map)
    return render_template("index.html", prediction=prediction, gradcam_img=gradcam_img)

if __name__ == "__main__":
    app.run(debug=True)