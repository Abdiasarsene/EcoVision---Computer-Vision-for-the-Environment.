# app/inference/explainability.py
import torch 
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import matplotlib.pyplot as plt 
from utils.config import settings

weights_path=settings.resnet_waste_pth
classes= settings.classes
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load(weights_path, map_location="cpu"))
model.eval()

def gradcam(model, img_tensor, target_layer, classes, pred_class):
    activations = None
    gradients= None
    
    def save_activation(module, input, output):
        nonlocal activations
        activations = output
        
    def save_gradient(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]
        
    # Hook
    target_layer.register_forward_hook(save_activation)
    target_layer.register_forward_hook(save_gradient)
    
    # Forward + Backward
    output  =model(img_tensor)
    class_score = output[0, pred_class]
    model.zero_grad()
    class_score.backward()
    
    # Grad-CAM
    weights = gradients.mean(dim=(2,3), keepdim=True)
    gradcam = (weights * activations).sum(dim=1).squeeze()
    gradcam = F.relu(gradcam)
    
    gradcam = gradcam.cpu().detach().numpy()
    gradcam = (gradcam - gradcam.min()) / (gradcam/max() - gradcam.min())
    
    return gradcam

import cv2
import numpy as np
from utils.config import settings

def overlay_gradcam(img_path, gradcam_map, output_path=settings.gradcam_chart):
    # Load image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    heatmap = cv2.resize(gradcam_map, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Overlay
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    
    # Save
    plt.imsave(output_path, superimposed_img)
    
    return output_path