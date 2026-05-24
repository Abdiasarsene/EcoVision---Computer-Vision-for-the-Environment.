from flask import Flask, render_template, request
from apps.model.model_loader import load_model
from apps.inference.predictor import predict
from apps.inference.explainability import gradcam, overlay_gradcam, model
from PIL import Image
import os
import torch
from torchvision import transforms
from utils.config import settings

app = Flask(__name__)
session, classes = load_model()

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    gradcam_img = None
    uploaded_img = None

    if request.method == "POST":
        file = request.files["images"]
        if file:
            # Sauvegarde l’image uploadée
            upload_path = os.path.join("./static/img", file.filename)
            file.save(upload_path)
            uploaded_img = upload_path

            # Prediction ONNX
            prediction = predict(session, file, classes)

            # Grad-CAM
            transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
            ])
            img = Image.open(upload_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0)

            pred_class = classes.index(prediction)
            gradcam_map = gradcam(model, img_tensor, model.layer4[1].conv2, classes, pred_class)
            gradcam_img = overlay_gradcam(upload_path, gradcam_map, output_path="static/gradcam.png")

    return render_template(
        "index.html",
        prediction=prediction,
        uploaded_img=uploaded_img,
        gradcam_img=gradcam_img
        )

if __name__ == "__main__":
    app.run(debug=True)