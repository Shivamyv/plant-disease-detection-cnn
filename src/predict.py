import argparse
from pathlib import Path
import torch
from torch import nn
from PIL import Image
from torchvision import transforms, models

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="models/best_model.pt")
    args = parser.parse_args()
    if not Path(args.model).exists():
        raise FileNotFoundError("Model not found. Run src/train.py first.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device)
    classes = checkpoint["classes"]
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    tf = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image = Image.open(args.image).convert("RGB")
    with torch.no_grad():
        probs = model(tf(image).unsqueeze(0).to(device)).softmax(1)[0]
    values, indices = probs.topk(min(3, len(classes)))
    for score, idx in zip(values.cpu().tolist(), indices.cpu().tolist()):
        print(f"{classes[idx]}: {score:.4f}")
    print("Model predictions are informational and should not replace expert diagnosis.")

if __name__ == "__main__":
    main()
