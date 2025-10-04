import argparse
import numpy as np
import cv2
import os
import shutil
import onnx
import onnxruntime as rt
import csv
from collections import defaultdict


# --- Preprocessing (cv2 based, deterministic) ---
def preprocess_cv2(img_bgr, image_size):
    # img_bgr: numpy array as read by cv2.imread
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, image_size)
    img = img.astype(np.float32) / 255.0
    # normalize with ImageNet stats
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    # HWC -> CHW
    img = np.transpose(img, (2, 0, 1))
    return img


def safe_softmax(logits):
    e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def load_classes(path):
    with open(path) as f:
        classes = [c.strip() for c in f.readlines() if c.strip()]
    return classes


def compute_metrics_if_labels(true_labels, pred_labels, class_names):
    try:
        from sklearn.metrics import classification_report, confusion_matrix
    except Exception:
        print("sklearn not available; skipping detailed metrics. Install scikit-learn to enable it.")
        return None

    report = classification_report(true_labels, pred_labels, target_names=class_names, digits=4, zero_division=0)
    cm = confusion_matrix(true_labels, pred_labels)
    return report, cm


def main(model_path, classes_path, image_size, image_folder, batch_size, confidence_threshold, save_images, labels_file=None):
    # Load ONNX model
    sess = rt.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    classes = load_classes(classes_path)

    os.makedirs("./runs/org", exist_ok=True)
    os.makedirs("./runs/pred", exist_ok=True)

    # load optional labels mapping (csv with filename,label)
    label_map = {}
    if labels_file and os.path.exists(labels_file):
        with open(labels_file) as lf:
            for line in lf:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    label_map[parts[0]] = parts[1]

    # gather image files
    img_files = [f for f in sorted(os.listdir(image_folder)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(img_files) == 0:
        print(f"No images found in {image_folder}")
        return

    csv_file = "predictions_validation.csv"
    results = []

    # batching
    for i in range(0, len(img_files), batch_size):
        batch_files = img_files[i:i+batch_size]
        batch_imgs = []
        valid_files = []
        for img_file in batch_files:
            img_path = os.path.join(image_folder, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: {img_file} could not be read; skipping")
                continue
            # copy original
            shutil.copy(img_path, os.path.join("./runs/org", img_file))
            p = preprocess_cv2(img, image_size)
            batch_imgs.append(p)
            valid_files.append(img_file)

        if len(batch_imgs) == 0:
            continue

        inp = np.stack(batch_imgs, axis=0).astype(np.float32)
        outs = sess.run([output_name], {input_name: inp})[0]
        probs = safe_softmax(outs)

        for idx, fname in enumerate(valid_files):
            p = probs[idx]
            pred_idx = int(p.argmax())
            pred_class = classes[pred_idx] if pred_idx < len(classes) else str(pred_idx)
            confidence = float(p[pred_idx])

            # Save annotated image optionally
            if save_images:
                org_img_path = os.path.join(image_folder, fname)
                org_img = cv2.imread(org_img_path)
                text = f"{pred_class} {confidence:.3f}"
                cv2.putText(org_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                save_path = os.path.join("./runs/pred", f"pred_{fname}")
                cv2.imwrite(save_path, org_img)
            else:
                save_path = ""

            true_label = label_map.get(fname)
            correct = None
            if true_label is not None:
                # if labels are class names, map to index for comparison
                try:
                    correct = (true_label == pred_class)
                except Exception:
                    correct = None

            results.append({
                'filename': fname,
                'predicted_class': pred_class,
                'confidence': round(confidence, 4),
                'true_label': true_label if true_label is not None else '',
                'correct': correct,
                'saved_image': save_path
            })

            print(f"{fname} -> {pred_class} (conf={confidence:.3f})")

    # write CSV
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename','predicted_class','confidence','true_label','correct','saved_image'])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\nPredictions written to {csv_file}")

    # compute metrics if labels provided
    true_labels = [r['true_label'] for r in results if r['true_label']]
    pred_labels = [r['predicted_class'] for r in results if r['true_label']]
    if len(true_labels) > 0:
        metrics = compute_metrics_if_labels(true_labels, pred_labels, classes)
        if metrics is not None:
            report, cm = metrics
            with open('classification_report.txt','w') as rf:
                rf.write(report)
            np.save('confusion_matrix.npy', cm)
            print('\nClassification report saved to classification_report.txt')
            print('Confusion matrix saved to confusion_matrix.npy')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify ONNX model on validation images")
    parser.add_argument('-m', '--model_path', required=True, help='Path to ONNX model')
    parser.add_argument('-c', '--classes_path', required=True, help='Path to classes.txt')
    parser.add_argument('-s', '--image_size', default=224, type=int, help='Image size (square)')
    parser.add_argument('-i', '--image_folder', default='validation_images', help='Folder with images to predict')
    parser.add_argument('-b', '--batch_size', default=16, type=int, help='Batch size for inference')
    parser.add_argument('-t', '--confidence_threshold', default=0.0, type=float, help='Confidence threshold to flag low-confidence predictions')
    parser.add_argument('--save_images', action='store_true', help='Save annotated images to ./runs/pred')
    parser.add_argument('--labels_file', default='', help='Optional CSV mapping filename,label to compute metrics')

    args = parser.parse_args()

    main(
        model_path=args.model_path,
        classes_path=args.classes_path,
        image_size=(args.image_size, args.image_size),
        image_folder=args.image_folder,
        batch_size=args.batch_size,
        confidence_threshold=args.confidence_threshold,
        save_images=args.save_images,
        labels_file=args.labels_file if args.labels_file else None
    )
