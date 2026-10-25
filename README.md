# Plant Disease Detection using CNN

PyTorch image-classification baseline for identifying plant leaf categories.

## Dataset layout
Download a public plant-leaf dataset (for example, PlantVillage) and check its license/terms. Organize it as:
```text
data/plantvillage/
  train/
    class_a/*.jpg
    class_b/*.jpg
  val/
    class_a/*.jpg
    class_b/*.jpg
  test/
    class_a/*.jpg
    class_b/*.jpg
```
Each class must be a folder. Keep the test set separate from training and model selection.

## Setup
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python src/train.py --data data/plantvillage --epochs 10
python src/predict.py --image path/to/leaf.jpg --model models/best_model.pt
```

This baseline uses transfer learning with a pretrained ResNet18. The first run may download pretrained weights. If no GPU is available, training may take longer. Report actual test metrics after running the script.
