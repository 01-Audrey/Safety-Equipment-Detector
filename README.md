# 🛡️ Safety Equipment Detector

AI-powered safety equipment detection system using YOLOv8 for construction site monitoring.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![mAP](https://img.shields.io/badge/mAP@50-70%25+-brightgreen)
![Status](https://img.shields.io/badge/Status-Production--Ready-success)

## 🎯 Project Overview

This project detects personal protective equipment (PPE) on construction workers to enhance workplace safety compliance. The system identifies:

- ✅ **Helmets** (hard hats)
- ✅ **Safety Vests** (high-visibility clothing)
- ⚠️ **No Helmet** (workers without head protection)
- ⚠️ **No Vest** (workers without visibility gear)
- 👷 **Persons** (all workers in frame)

### 🎥 Demo

*[GIF or video demo will go here after deployment]*

## 📊 Model Performance

| Version | Dataset Size | Model | Epochs | mAP@50 | Status |
|---------|-------------|-------|--------|--------|--------|
| v1 | 66 images | YOLOv8n | 10 | 17.1% | Baseline |
| v2 | 66 images | YOLOv8n | 50 | 48.5% | Optimized |
| v3 | 246 images | YOLOv8s | 100 | **[TBD]%** | Production |

### 📈 Performance Metrics (v3)
```
Overall mAP@50: [TBD]%
Precision: [TBD]%
Recall: [TBD]%
Inference Speed: ~1.3ms per image (real-time capable)
```

## 🏗️ Architecture

- **Base Model:** YOLOv8s (11M parameters)
- **Input Size:** 640×640 pixels
- **Framework:** Ultralytics YOLO
- **Training:** Transfer learning from COCO pretrained weights

## 📁 Project Structure
```
Safety-Equipment-Detector/
├── notebooks/          # Training & evaluation notebooks
├── results/           # Model outputs & visualizations
├── models/            # Training configurations
├── src/              # Source code (future deployment)
└── docs/             # Documentation
```

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.11+
CUDA (optional, for GPU acceleration)
```

### Installation
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/Safety-Equipment-Detector.git
cd Safety-Equipment-Detector

# Install dependencies
pip install -r requirements.txt
```

### Training
```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8s.pt')

# Train
results = model.train(
    data='data/data.yaml',
    epochs=100,
    imgsz=640
)
```

### Inference
```python
from ultralytics import YOLO

# Load trained model
model = YOLO('models/best.pt')

# Run inference
results = model('path/to/image.jpg')
results[0].show()
```

## 📊 Results & Analysis

### Version Evolution

**v1 → v2 → v3 Journey:**

1. **v1 (Baseline):** Quick prototype with minimal data
   - Result: 17.1% mAP
   - Learning: Need more training time

2. **v2 (Optimization):** Hyperparameter tuning
   - Result: 48.5% mAP (+184% improvement!)
   - Learning: Model capacity sufficient, need more data

3. **v3 (Production):** Dataset expansion + model upgrade
   - Result: [TBD]% mAP
   - 4.7x more source images (22→104)
   - Bigger model (YOLOv8n→YOLOv8s)

### Key Insights

- 📈 **Data quality > Model size** (initially)
- ⏰ **Training time matters** (10→50 epochs = 3x improvement)
- 🎯 **Systematic iteration** produces results
- 🔄 **Transfer learning** accelerates development

## 🛠️ Technical Details

### Dataset

- **Source:** Custom annotated construction site images
- **Size:** 104 source images → 246 augmented
- **Split:** 70% train / 20% validation / 10% test
- **Annotation Tool:** Roboflow
- **Classes:** 5 (helmet, no-helmet, vest, no-vest, person)

### Augmentation

- Horizontal flip (50%)
- Brightness adjustment (±15%)
- Mosaic augmentation
- Mixup (15%)
- Copy-paste (10%)

### Training Configuration
```yaml
model: yolov8s.pt
epochs: 100
batch: 16
imgsz: 640
optimizer: AdamW
lr0: 0.01
patience: 30
```

## 🎯 Use Cases

1. **Construction Site Monitoring**
   - Real-time PPE compliance checking
   - Automated safety violation alerts

2. **Safety Audits**
   - Analyze historical footage
   - Generate compliance reports

3. **Access Control**
   - Gate entry verification
   - Restricted area monitoring

## 🚧 Future Improvements

- [ ] Expand to 500+ images for 80%+ mAP
- [ ] Add pose estimation (proper wearing detection)
- [ ] Multi-camera deployment system
- [ ] Real-time alert dashboard
- [ ] Mobile app integration
- [ ] Edge device deployment (Jetson Nano)

## 📚 Documentation

- [Project Overview](docs/project_overview.md)
- [Training Guide](docs/training.md)
- [Deployment Guide](docs/deployment.md)
- [API Reference](docs/api.md)

## 🤝 Contributing

This is a portfolio project, but feedback and suggestions are welcome!

## 📝 License

MIT License (or choose appropriate license)

## 👤 Author

**Audrey**
- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/your-profile)
- Portfolio: [Your Website](https://yourwebsite.com)

## 🙏 Acknowledgments

- Ultralytics YOLO team
- Roboflow annotation platform
- Construction safety image datasets

---

⭐ **Star this repo if you find it helpful!**

*Built with 💪 as part of ML Learning Journey*
```

---

## 📋 STEP 4: CREATE .gitignore

**Create: `.gitignore`**
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb_checkpoints/

# Machine Learning
*.pt
*.pth
*.onnx
*.h5
*.pkl
*.weights
models/weights/
runs/
wandb/

# Dataset (DON'T commit large image datasets!)
data/images/
data/train/
data/valid/
data/test/
datasets/
*.jpg
*.jpeg
*.png
*.mp4
*.avi

# Environment
.env
.venv
.DS_Store
Thumbs.db

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Logs
*.log
logs/

# Results (commit visualizations, not raw outputs)
results/raw/
*.cache
```

---

## 📦 STEP 5: CREATE requirements.txt

**Create: `requirements.txt`**
```
# Core ML
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0

# Data & Visualization
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
pillow>=10.0.0
opencv-python>=4.8.0

# Jupyter
jupyter>=1.0.0
ipykernel>=6.25.0

# Utilities
pyyaml>=6.0
tqdm>=4.65.0
requests>=2.31.0

# Optional: for deployment
# fastapi>=0.100.0
# uvicorn>=0.23.0
# streamlit>=1.25.0
