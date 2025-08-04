# 🌾 Agricultural Crop Identification

A comprehensive deep learning solution for classifying 30 different agricultural crops using advanced computer vision techniques with NASNet architecture.

## 🎯 Project Overview

This project implements state-of-the-art deep learning models for agricultural crop classification, featuring:

- **30 Crop Classes**: Rice, wheat, maize, cotton, sugarcane, and 25 more agricultural crops
- **NASNet Architecture**: EfficientNet-B7 based model with 88% validation accuracy
- **Interactive Web App**: Streamlit-based interface for real-time crop prediction
- **Comprehensive Notebooks**: Detailed Jupyter notebooks for training and analysis

## 🏆 Model Performance

- **🎯 Validation Accuracy: 88.0%**
- **📈 Training Epochs: 68**
- **🏗️ Architecture: NASNet-equivalent (EfficientNet-B7)**
- **📐 Input Size: 331x331 pixels**
- **💾 Model Size: ~755 MB**

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/shanthan1999/Agricultural-Crop-Identification.git
cd Agricultural-Crop-Identification
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Extract dataset** (if you have the dataset zip)
```bash
# Extract to agricultural_data/Agricultural-crops/
```

4. **Run the Streamlit app**
```bash
streamlit run app.py
```

## 📁 Project Structure

```
Agricultural-Crop-Identification/
├── app.py                                          # Main Streamlit application
├── best_nasnet_crop_model.pth                      # Pre-trained NASNet model (88% accuracy)
├── agricultural_crop_classification_nasnet_revised.ipynb  # Enhanced training notebook
├── requirements.txt                                # Python dependencies
├── agricultural_data/                              # Dataset directory
│   └── Agricultural-crops/                         # 30 crop class folders
└── README.md                                       # This file
```

## 🌟 Features

### 🖥️ **Streamlit Web Application**
- **📊 Dataset Overview**: Comprehensive statistics and visualizations
- **🏗️ Model Training**: Interactive training interface
- **📈 Model Evaluation**: Detailed performance metrics
- **🔮 Crop Prediction**: Real-time image classification
- **🎯 Batch Testing**: Test on multiple images

### 📓 **Jupyter Notebook**
- **Advanced Training Pipeline**: Complete training workflow
- **Data Augmentation**: Agricultural-specific transformations
- **Early Stopping**: Prevents overfitting with patience mechanism
- **Comprehensive Evaluation**: Confusion matrix, classification reports
- **Visualization Tools**: Training history and performance plots

## 🔧 Technical Details

### Model Architecture
- **Base Model**: EfficientNet-B7 (NASNet-equivalent)
- **Input Size**: 331×331 pixels
- **Output Classes**: 30 agricultural crops
- **Regularization**: Dropout, batch normalization, weight decay
- **Training Strategy**: Transfer learning with layer freezing

### Data Processing
- **Augmentation**: Rotation, flipping, color jitter, perspective transforms
- **Normalization**: ImageNet statistics
- **Class Balancing**: Weighted sampling and loss functions
- **Stratified Splitting**: Maintains class distribution

### Performance Optimizations
- **Early Stopping**: Patience-based training termination
- **Learning Rate Scheduling**: Cosine annealing with warm restarts
- **Gradient Clipping**: Prevents exploding gradients
- **Mixed Precision**: Memory-efficient training (when available)

## 📊 Supported Crop Classes

The model can identify 30 different agricultural crops:

1. Almond
2. Banana
3. Cardamom
4. Cherry
5. Chilli
6. Clove
7. Coconut
8. Coffee Plant
9. Cotton
10. Cucumber
11. Fox Nut (Makhana)
12. Gram
13. Jowar
14. Jute
15. Lemon
16. Maize
17. Mustard Oil
18. Olive Tree
19. Papaya
20. Pearl Millet (Bajra)
21. Pineapple
22. Rice
23. Soybean
24. Sugarcane
25. Sunflower
26. Tea
27. Tobacco Plant
28. Tomato
29. Vigna Radiata (Mung)
30. Wheat

## 🎮 Usage Examples

### Web Application
```bash
# Start the Streamlit app
streamlit run app.py

# Navigate to http://localhost:8501
# Upload crop images for instant classification
```

### Jupyter Notebook
```bash
# Launch Jupyter
jupyter notebook agricultural_crop_classification_nasnet_revised.ipynb

# Follow the notebook cells for:
# - Data exploration
# - Model training
# - Performance evaluation
```

### Python API
```python
import torch
from PIL import Image
from torchvision import transforms

# Load the model
model = torch.load('best_nasnet_crop_model.pth', map_location='cpu')

# Preprocess image
transform = transforms.Compose([
    transforms.Resize(int(331 * 1.15)),
    transforms.CenterCrop(331),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Make prediction
image = Image.open('crop_image.jpg')
input_tensor = transform(image).unsqueeze(0)
with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1)
```

## 🔬 Research & Development

### Training Process
1. **Data Preparation**: Stratified splitting with class balancing
2. **Augmentation**: Agricultural-specific transformations
3. **Model Selection**: NASNet-equivalent architecture
4. **Training**: 68 epochs with early stopping
5. **Validation**: 88% accuracy achieved
6. **Testing**: Comprehensive evaluation metrics

### Key Innovations
- **Agricultural-Specific Augmentation**: Tailored for crop images
- **Multi-Scale Architecture**: Captures both global and local features
- **Attention Mechanisms**: Focus on discriminative crop features
- **Robust Evaluation**: Cross-validation and detailed metrics

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| Validation Accuracy | 88.0% |
| Training Epochs | 68 |
| Model Parameters | ~66M |
| Inference Time | ~50ms (GPU) |
| Memory Usage | ~2GB (training) |

## 🛠️ Development

### Requirements
- PyTorch 2.6+
- Streamlit 1.28+
- scikit-learn 1.3+
- Pillow 10.0+
- NumPy 1.24+

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: Agricultural crop image dataset
- **Architecture**: EfficientNet-B7 (NASNet-equivalent)
- **Framework**: PyTorch and Streamlit
- **Inspiration**: Precision agriculture and computer vision research

## 📞 Contact

- **Author**: Shanthan
- **GitHub**: [@shanthan1999](https://github.com/shanthan1999)
- **Repository**: [Agricultural-Crop-Identification](https://github.com/shanthan1999/Agricultural-Crop-Identification)

## 🚀 Future Enhancements

- [ ] Mobile app deployment
- [ ] Real-time video classification
- [ ] Crop disease detection
- [ ] Multi-language support
- [ ] API endpoint deployment
- [ ] Edge device optimization

---

**⭐ If you find this project helpful, please give it a star!**

Made with ❤️ for sustainable agriculture and AI research.