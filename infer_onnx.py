import onnxruntime as ort
import cv2
import numpy as np
import os
import torch
import csv

# --- Load class names from your training checkpoint ---
checkpoint_path = "outputs/best_model.pth"
ckpt = torch.load(checkpoint_path, map_location="cpu")
class_names = ckpt.get("class_names", ["class0"])  # fallback if not found

# --- Load ONNX model ---
onnx_model_path = "outputs/model.onnx"
session = ort.InferenceSession(onnx_model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# --- Folder with images ---
image_folder = "verification_images"

# --- Gather all image files and sort alphabetically ---
image_files = sorted(os.listdir(image_folder))

# --- CSV output ---
csv_file = "predictions.csv"
with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "predicted_class", "confidence"])

    # --- Loop through images ---
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        
        # Load and preprocess image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        
        # Run inference
        outputs = session.run([output_name], {input_name: img})
        outputs = outputs[0][0]  # get first batch
        
        # Apply softmax to get confidence
        exp_scores = np.exp(outputs - np.max(outputs))
        probs = exp_scores / exp_scores.sum()
        pred_class_idx = np.argmax(probs)
        pred_class = class_names[pred_class_idx]
        confidence = probs[pred_class_idx]
        
        # Write to CSV
        writer.writerow([img_file, pred_class, confidence])
        print(f"{img_file} → {pred_class}, {confidence:.3f}")

print(f"\nPredictions saved to {csv_file}")
