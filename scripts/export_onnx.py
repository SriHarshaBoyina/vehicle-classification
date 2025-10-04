import torch
import torch.onnx
import argparse
from torchvision import models

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model .pth file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the ONNX model')
    parser.add_argument('--image_size', type=int, default=224, help='Input image size (default: 224)')
    args = parser.parse_args()

    dummy_input = torch.randn(1, 3, args.image_size, args.image_size)

    # Load checkpoint
    ckpt = torch.load(args.model_path, map_location=torch.device('cpu'))
    class_names = ckpt.get('class_names')
    model_state = ckpt['model_state']
    # Build model (assume resnet50, adjust if needed)
    model = models.resnet50(pretrained=False)
    num_f = model.fc.in_features
    model.fc = torch.nn.Linear(num_f, len(class_names))
    model.load_state_dict(model_state)
    model.eval()

    torch.onnx.export(
        model,
        dummy_input,
        args.output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=11,
        verbose=True
    )
    print(f"Exported ONNX model to {args.output_path}")

if __name__ == "__main__":
    main()
