import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import os
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import time

# Set page config
st.set_page_config(
    page_title="Agricultural Crop Classification",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2E8B57;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🌾 Agricultural Crop Classification System</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", [
    "📊 Dataset Overview", 
    "🏗️ Model Training", 
    "📈 Model Evaluation", 
    "🔮 Crop Prediction"
])

# Load crop classes and image counts
@st.cache_data
def get_crop_classes():
    data_path = "agricultural_data/Agricultural-crops"
    if os.path.exists(data_path):
        return sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    return []

@st.cache_data
def get_image_counts(crop_classes):
    """Get image counts for all crop classes"""
    image_counts = {}
    total_images = 0
    for class_name in crop_classes:
        class_dir = os.path.join("agricultural_data/Agricultural-crops", class_name)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            image_counts[class_name] = count
            total_images += count
        else:
            image_counts[class_name] = 0
    return image_counts, total_images

crop_classes = get_crop_classes()
image_counts, total_images = get_image_counts(crop_classes)

# Custom Dataset Class
class CropDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(crop_classes)}
        
        for class_name in crop_classes:
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ResNet-based Model Architecture (Compatible with saved model)
class CropClassifier(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(CropClassifier, self).__init__()
        
        # Use ResNet50 backbone (matches the saved model structure)
        self.backbone = models.resnet50(pretrained=True)
        
        # Replace the final fully connected layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

# EfficientNet-based Model Architecture
class CropClassifierEfficientNet(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(CropClassifierEfficientNet, self).__init__()
        
        # Use EfficientNet-B3 for better fine-grained classification
        self.backbone = models.efficientnet_b3(pretrained=True)
        
        # Freeze early layers
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False
        
        # Custom classifier head
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

# NASNet-equivalent Model Architecture (EfficientNet-B7)
class NASNetCropClassifier(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(NASNetCropClassifier, self).__init__()
        
        # Use EfficientNet-B7 as NASNet equivalent
        self.backbone = models.efficientnet_b7(pretrained=True)
        
        # Freeze early layers
        for param in list(self.backbone.parameters())[:-30]:
            param.requires_grad = False
        
        # Custom classifier head with more capacity
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate * 0.4),
            nn.Linear(num_features, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

# Alternative ResNet-based architecture for compatibility
class CropClassifierAlt(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(CropClassifierAlt, self).__init__()
        
        # Use ResNet101 for better feature extraction
        self.model = models.resnet101(pretrained=True)
        
        # Freeze early layers
        for param in list(self.model.parameters())[:-30]:
            param.requires_grad = False
        
        # Enhanced classifier with attention mechanism
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)

# Specialized Agricultural CNN with Multi-Scale Features
class AgriCropNet(nn.Module):
    def __init__(self, num_classes):
        super(AgriCropNet, self).__init__()
        
        # Multi-scale feature extraction
        self.backbone = models.resnet50(pretrained=True)
        
        # Remove the final layers to get feature maps
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Multi-scale pooling for different crop parts
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.local_pool = nn.AdaptiveMaxPool2d(1)
        
        # Attention mechanism for crop-specific features
        self.attention = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.ReLU(),
            nn.Conv2d(512, 1, 1),
            nn.Sigmoid()
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048 * 2, 1024),  # *2 for global + local features
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # Extract features
        features = self.features(x)
        
        # Apply attention
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        
        # Multi-scale pooling
        global_features = self.global_pool(attended_features).flatten(1)
        local_features = self.local_pool(attended_features).flatten(1)
        
        # Combine features
        combined_features = torch.cat([global_features, local_features], dim=1)
        
        # Classify
        return self.classifier(combined_features)

# Dataset Overview Page
if page == "📊 Dataset Overview":
    st.header("Dataset Overview")
    
    if not crop_classes:
        st.error("Dataset not found! Please ensure the Agricultural-crops folder is in the agricultural_data directory.")
        st.stop()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Crop Classes", len(crop_classes))
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Use global image counts (already calculated)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Images", total_images)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        avg_images = total_images / len(crop_classes) if crop_classes else 0
        st.metric("Avg Images per Class", f"{avg_images:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Distribution chart
    st.subheader("Class Distribution")
    df = pd.DataFrame(list(image_counts.items()), columns=['Crop', 'Count'])
    fig = px.bar(df, x='Crop', y='Count', title="Number of Images per Crop Class")
    fig.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Sample images
    st.subheader("Sample Images")
    selected_crop = st.selectbox("Select a crop to view samples:", crop_classes)
    
    if selected_crop:
        crop_dir = os.path.join("agricultural_data/Agricultural-crops", selected_crop)
        if os.path.exists(crop_dir):
            image_files = [f for f in os.listdir(crop_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:6]
            
            cols = st.columns(3)
            for i, img_file in enumerate(image_files):
                with cols[i % 3]:
                    img_path = os.path.join(crop_dir, img_file)
                    image = Image.open(img_path)
                    st.image(image, caption=f"{selected_crop} - {img_file}", use_column_width=True)

# Model Training Page
elif page == "🏗️ Model Training":
    st.header("Model Training")
    
    if not crop_classes:
        st.error("Dataset not found!")
        st.stop()
    
    if not image_counts:
        st.error("Unable to load dataset information!")
        st.stop()
    
    # Debug information
    with st.expander("🔍 Debug Information"):
        st.write(f"Crop classes found: {len(crop_classes)}")
        st.write(f"Image counts available: {len(image_counts)}")
        st.write(f"Total images: {sum(image_counts.values())}")
        if crop_classes:
            st.write(f"Sample classes: {crop_classes[:5]}")
        if image_counts:
            sample_counts = dict(list(image_counts.items())[:5])
            st.write(f"Sample counts: {sample_counts}")
    
    # Training parameters
    st.subheader("Training Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        epochs = st.slider("Number of Epochs", 1, 50, 10)
        batch_size = st.selectbox("Batch Size", [8, 16, 32, 64], index=1)
        learning_rate = st.selectbox("Learning Rate", [0.001, 0.01, 0.1], index=0)
    
    with col2:
        train_split = st.slider("Training Split (%)", 60, 90, 80)
        model_type = st.selectbox("Model Architecture", [
            "CropClassifier (ResNet50 - Compatible)",
            "CropClassifierEfficientNet (EfficientNet-B3)",
            "NASNetCropClassifier (EfficientNet-B7)",
            "CropClassifierAlt (ResNet101)", 
            "AgriCropNet (Multi-scale)"
        ])
        data_augmentation = st.checkbox("Enable Data Augmentation", value=True)
        use_scheduler = st.checkbox("Use Learning Rate Scheduler", value=True)
    
    if st.button("Start Training", type="primary"):
        # Enhanced data transforms for agricultural crops
        if data_augmentation:
            train_transform = transforms.Compose([
                # Preserve aspect ratio with padding
                transforms.Resize(256),
                transforms.CenterCrop(224),
                
                # Agricultural-specific augmentations
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),  # Crops can be viewed from different angles
                transforms.RandomRotation(degrees=30),  # More rotation for natural crop variations
                
                # Color augmentations for different lighting/seasons
                transforms.ColorJitter(
                    brightness=0.4,  # Different lighting conditions
                    contrast=0.4,    # Weather variations
                    saturation=0.3,  # Seasonal changes
                    hue=0.1         # Slight color variations
                ),
                
                # Geometric augmentations
                transforms.RandomAffine(
                    degrees=15,
                    translate=(0.1, 0.1),
                    scale=(0.8, 1.2),
                    shear=10
                ),
                
                # Random perspective for field variations
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                
                # Random erasing to simulate occlusion
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
                
                # Normalization
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Create dataset
        full_dataset = CropDataset("agricultural_data/Agricultural-crops", train_transform)
        
        # Split dataset
        train_size = int(train_split / 100 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        
        # Update validation dataset transform
        val_dataset.dataset.transform = val_transform
        
        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Enhanced model selection
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if "ResNet50" in model_type:
            model = CropClassifier(len(crop_classes)).to(device)
            st.info("🔧 Using ResNet50 architecture (compatible with existing models)")
        elif "EfficientNet-B3" in model_type:
            model = CropClassifierEfficientNet(len(crop_classes)).to(device)
            st.info("🔧 Using EfficientNet-B3 architecture")
        elif "EfficientNet-B7" in model_type:
            model = NASNetCropClassifier(len(crop_classes)).to(device)
            st.info("🔧 Using NASNet-equivalent (EfficientNet-B7) architecture")
        elif "ResNet101" in model_type:
            model = CropClassifierAlt(len(crop_classes)).to(device)
            st.info("🔧 Using ResNet101 architecture")
        else:  # AgriCropNet
            model = AgriCropNet(len(crop_classes)).to(device)
            st.info("🔧 Using custom AgriCropNet architecture")
        
        st.info(f"Using {model_type} with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
        
        # Enhanced loss function with class weighting
        # Calculate class weights for imbalanced dataset
        try:
            class_counts = [image_counts.get(crop, 0) for crop in crop_classes]
            total_samples = sum(class_counts)
            
            if total_samples == 0:
                st.error("No images found in dataset!")
                st.stop()
            
            class_weights = [total_samples / (len(crop_classes) * count) if count > 0 else 1.0 for count in class_counts]
            class_weights = torch.FloatTensor(class_weights).to(device)
            
            st.info(f"📊 Dataset loaded: {total_samples} total images across {len(crop_classes)} classes")
            
        except Exception as e:
            st.error(f"Error calculating class weights: {str(e)}")
            st.info("Using equal weights for all classes")
            class_weights = torch.ones(len(crop_classes)).to(device)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Enhanced optimizer with weight decay
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        # Learning rate scheduler
        if use_scheduler:
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        
        # Training loop
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
            
            # Calculate metrics
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            # Update learning rate
            if use_scheduler:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = learning_rate
            
            # Update progress
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            status_text.text(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, LR: {current_lr:.6f}")
        
        # Save model
        torch.save(model.state_dict(), 'final_crop_model.pth')
        
        st.success("Training completed! Model saved as 'final_crop_model.pth'")
        
        # Plot training history
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(train_losses, label='Train Loss')
        ax1.plot(val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        ax2.plot(train_accuracies, label='Train Accuracy')
        ax2.plot(val_accuracies, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        
        st.pyplot(fig)

# Model Evaluation Page
elif page == "📈 Model Evaluation":
    st.header("Model Evaluation")
    
    if not (os.path.exists('final_crop_model.pth') or os.path.exists('best_nasnet_crop_model.pth')):
        st.warning("No trained model found! Please train a model first.")
        st.stop()
    
    # Enhanced model loading with architecture detection
    @st.cache_resource
    def load_model():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # Load the saved model (prioritize NASNet model)
            if os.path.exists('best_nasnet_crop_model.pth'):
                checkpoint = torch.load('best_nasnet_crop_model.pth', map_location=device, weights_only=False)
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                st.info("🔧 Loading NASNet model from best_nasnet_crop_model.pth")
                
                # Display model info if available
                if 'epoch' in checkpoint:
                    st.info(f"📈 Training epoch: {checkpoint['epoch']}")
                if 'val_acc' in checkpoint:
                    st.info(f"🎯 Validation accuracy: {checkpoint['val_acc']:.4f}")
                    
            elif os.path.exists('final_crop_model.pth'):
                state_dict = torch.load('final_crop_model.pth', map_location=device, weights_only=False)
                st.info("🔧 Loading model from final_crop_model.pth")
            else:
                st.error("No trained model found!")
                st.info("Please train a model first using the training page.")
                return None, device
            
            # Detect model architecture based on state dict keys
            sample_keys = list(state_dict.keys())
            
            if any('backbone.features' in key for key in sample_keys):
                # EfficientNet-based model (NASNet equivalent)
                model = NASNetCropClassifier(len(crop_classes)).to(device)
                st.info("🔧 Detected NASNet-equivalent (EfficientNet-B7) model")
            elif any('efficientnet' in key.lower() for key in sample_keys):
                if any('classifier.1.weight' in key for key in sample_keys):
                    # EfficientNet-B7 (NASNet equivalent)
                    model = NASNetCropClassifier(len(crop_classes)).to(device)
                    st.info("🔧 Detected NASNet-equivalent (EfficientNet-B7) model")
                else:
                    # EfficientNet-B3
                    model = CropClassifierEfficientNet(len(crop_classes)).to(device)
                    st.info("🔧 Detected EfficientNet-B3 model")
            elif any(key.startswith('backbone.conv1') for key in sample_keys):
                # ResNet-based model
                model = CropClassifier(len(crop_classes)).to(device)
                st.info("🔧 Detected ResNet-based model")
            elif any(key.startswith('model.') for key in sample_keys):
                # Alternative architecture
                model = CropClassifierAlt(len(crop_classes)).to(device)
                st.info("🔧 Detected alternative model architecture")
            else:
                # Default to NASNet for best_nasnet_crop_model.pth
                model = NASNetCropClassifier(len(crop_classes)).to(device)
                st.info("🔧 Using NASNet architecture as default")
            
            # Try to load the state dict
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            
            st.success("✅ Model loaded successfully!")
            return model, device
            
        except Exception as e:
            st.error(f"❌ Model loading failed: {str(e)}")
            st.info("💡 Troubleshooting tips:")
            st.info("1. Make sure you have trained a model first")
            st.info("2. Check if the model file is corrupted")
            st.info("3. Try retraining the model with the current architecture")
            
            # Offer to create a new model for training
            if st.button("🔄 Reset and Create New Model"):
                if os.path.exists('final_crop_model.pth'):
                    os.remove('final_crop_model.pth')
                if os.path.exists('best_nasnet_crop_model.pth'):
                    os.remove('best_nasnet_crop_model.pth')
                st.success("Model files cleared. Please retrain using the training page.")
                st.experimental_rerun()
            
            return None, device
    
    model, device = load_model()
    
    if model is None:
        st.stop()
    
    # Evaluation on test set
    if st.button("Evaluate Model", type="primary"):
        # Create test dataset with appropriate input size
        # Use 331x331 for NASNet, 224x224 for others
        input_size = 331 if 'NASNet' in str(type(model).__name__) else 224
        
        test_transform = transforms.Compose([
            transforms.Resize(int(input_size * 1.15)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        test_dataset = CropDataset("agricultural_data/Agricultural-crops", test_transform)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Evaluate
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
        
        # Calculate accuracy
        accuracy = 100 * sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_labels)
        
        st.success(f"Overall Accuracy: {accuracy:.2f}%")
        
        # Classification report
        st.subheader("Classification Report")
        report = classification_report(all_labels, all_predictions, 
                                     target_names=crop_classes, output_dict=True)
        
        # Convert to DataFrame for better display
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(3))
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(all_labels, all_predictions)
        
        fig, ax = plt.subplots(figsize=(15, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=crop_classes, yticklabels=crop_classes, ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        st.pyplot(fig)
        
        # Per-class accuracy
        st.subheader("Per-Class Accuracy")
        class_accuracies = []
        for i, class_name in enumerate(crop_classes):
            class_correct = sum((p == i and l == i) for p, l in zip(all_predictions, all_labels))
            class_total = sum(l == i for l in all_labels)
            class_acc = 100 * class_correct / class_total if class_total > 0 else 0
            class_accuracies.append(class_acc)
        
        acc_df = pd.DataFrame({
            'Crop': crop_classes,
            'Accuracy (%)': class_accuracies
        }).sort_values('Accuracy (%)', ascending=False)
        
        fig = px.bar(acc_df, x='Crop', y='Accuracy (%)', 
                    title="Per-Class Accuracy")
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

# Crop Prediction Page
elif page == "🔮 Crop Prediction":
    st.header("Crop Prediction")
    
    if not (os.path.exists('final_crop_model.pth') or os.path.exists('best_nasnet_crop_model.pth')):
        st.warning("No trained model found! Please train a model first.")
        st.stop()
    
    # Enhanced prediction model loading
    @st.cache_resource
    def load_prediction_model():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # Load the saved model (prioritize NASNet model)
            if os.path.exists('best_nasnet_crop_model.pth'):
                checkpoint = torch.load('best_nasnet_crop_model.pth', map_location=device, weights_only=False)
                state_dict = checkpoint.get('model_state_dict', checkpoint)
            elif os.path.exists('final_crop_model.pth'):
                state_dict = torch.load('final_crop_model.pth', map_location=device, weights_only=False)
            else:
                st.error("No trained model found!")
                st.info("Please train a model first using the training page.")
                return None, device
            
            # Detect model architecture based on state dict keys
            sample_keys = list(state_dict.keys())
            
            if any('backbone.features' in key for key in sample_keys):
                # EfficientNet-based model (NASNet equivalent)
                model = NASNetCropClassifier(len(crop_classes)).to(device)
            elif any('efficientnet' in key.lower() for key in sample_keys):
                if any('classifier.1.weight' in key for key in sample_keys):
                    # EfficientNet-B7 (NASNet equivalent)
                    model = NASNetCropClassifier(len(crop_classes)).to(device)
                else:
                    # EfficientNet-B3
                    model = CropClassifierEfficientNet(len(crop_classes)).to(device)
            elif any(key.startswith('backbone.conv1') for key in sample_keys):
                # ResNet-based model
                model = CropClassifier(len(crop_classes)).to(device)
            elif any(key.startswith('model.') for key in sample_keys):
                # Alternative architecture
                model = CropClassifierAlt(len(crop_classes)).to(device)
            else:
                # Default to NASNet for best_nasnet_crop_model.pth
                model = NASNetCropClassifier(len(crop_classes)).to(device)
            
            # Try to load the state dict
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            
            return model, device
            
        except Exception as e:
            st.error(f"❌ Prediction model loading failed: {str(e)}")
            st.info("💡 Please ensure you have a trained model available.")
            return None, device
    
    model, device = load_prediction_model()
    
    if model is None:
        st.stop()
    
    # Image upload
    uploaded_file = st.file_uploader("Choose a crop image...", 
                                   type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            # Preprocess image with appropriate input size
            # Use 331x331 for NASNet, 224x224 for others
            input_size = 331 if 'NASNet' in str(type(model).__name__) else 224
            
            transform = transforms.Compose([
                transforms.Resize(int(input_size * 1.15)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            # Make prediction
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                
            # Get top 5 predictions
            top5_prob, top5_indices = torch.topk(probabilities, 5)
            
            st.subheader("Prediction Results")
            
            for i in range(5):
                crop_name = crop_classes[top5_indices[i]]
                confidence = top5_prob[i].item() * 100
                
                st.write(f"**{i+1}. {crop_name}**: {confidence:.2f}%")
                st.progress(confidence / 100)
            
            # Prediction confidence chart
            st.subheader("Top 5 Predictions")
            pred_data = {
                'Crop': [crop_classes[idx] for idx in top5_indices],
                'Confidence (%)': [prob.item() * 100 for prob in top5_prob]
            }
            
            fig = px.bar(pred_data, x='Confidence (%)', y='Crop', 
                        orientation='h', title="Prediction Confidence")
            st.plotly_chart(fig, use_container_width=True)
    
    # Batch prediction
    st.subheader("Batch Prediction")
    if st.button("Test on Random Sample Images"):
        # Get random images from dataset
        sample_images = []
        sample_labels = []
        
        for class_idx, class_name in enumerate(crop_classes[:6]):  # Show 6 classes
            class_dir = os.path.join("agricultural_data/Agricultural-crops", class_name)
            if os.path.exists(class_dir):
                image_files = [f for f in os.listdir(class_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if image_files:
                    img_file = np.random.choice(image_files)
                    img_path = os.path.join(class_dir, img_file)
                    sample_images.append(img_path)
                    sample_labels.append(class_name)
        
        # Display predictions
        cols = st.columns(3)
        for i, (img_path, true_label) in enumerate(zip(sample_images, sample_labels)):
            with cols[i % 3]:
                image = Image.open(img_path).convert('RGB')
                st.image(image, caption=f"True: {true_label}", use_column_width=True)
                
                # Predict with appropriate input size
                # Use 331x331 for NASNet, 224x224 for others
                input_size = 331 if 'NASNet' in str(type(model).__name__) else 224
                
                transform = transforms.Compose([
                    transforms.Resize(int(input_size * 1.15)),
                    transforms.CenterCrop(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                
                input_tensor = transform(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    predicted_idx = torch.argmax(probabilities).item()
                    confidence = probabilities[predicted_idx].item() * 100
                
                predicted_crop = crop_classes[predicted_idx]
                color = "green" if predicted_crop == true_label else "red"
                
                st.markdown(f"**Predicted**: <span style='color:{color}'>{predicted_crop}</span>", 
                           unsafe_allow_html=True)
                st.write(f"**Confidence**: {confidence:.1f}%")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and PyTorch")