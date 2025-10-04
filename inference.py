import csv
# inference.py
import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from torchvision import models

def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    class_names = ckpt.get("class_names")
    model_state = ckpt["model_state"]
    # Build model slightly generically (assumes resnet50 unless stated)
    model = models.resnet50(pretrained=False)
    num_f = model.fc.in_features
    model.fc = torch.nn.Linear(num_f, len(class_names))
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()
    return model, class_names

def predict_folder(model, class_names, folder, device):
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    out = []
    for fname in sorted(os.listdir(folder)):
        p = os.path.join(folder, fname)
        if not fname.lower().endswith((".jpg",".png",".jpeg")):
            continue
        img = Image.open(p).convert("RGB")
        x = tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            pr = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred = int(pr.argmax())
            out.append((fname, class_names[pred], float(pr.max())))
    return out

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--input_folder", required=True)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_model(args.model_path, device)
    preds = predict_folder(model, class_names, args.input_folder, device)
    results = []
    for fn, pred_class, confidence in preds:
        results.append({
            'filename': os.path.basename(fn),
            'predicted_class': pred_class,
            'confidence': confidence
        })

    # Save predictions to predictions.csv
    with open('predictions.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['filename', 'predicted_class', 'confidence'])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print('Saved predictions to predictions.csv')
    for fn, cls, conf in preds:
        print(fn, cls, f"{conf:.3f}")
