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
st.markdown("<h1 class=\"main-header\">🌾 Agricultural Crop Classification System</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", [
    "📊 Dataset Overview", 
    "🏗️ Model Training", 
    "📈 Model Evaluation", 
    "🔮 Crop Prediction"
])

# Load crop classes
@st.cache_data
def get_crop_classes():
    data_path = "agricultural_data/Agricultural-crops"
    if os.path.exists(data_path):
        return sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    return []

crop_classes = get_crop_classes()

# Count images per class (moved to a higher scope)
@st.cache_data
def count_images():
    image_counts = {}
    total_images = 0
    for class_name in crop_classes:
        class_dir = os.path.join("agricultural_data/Agricultural-crops", class_name)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir) 
                       if f.lower().endswith((".png", ".jpg", ".jpeg"))])
            image_counts[class_name] = count
            total_images += count
    return image_counts, total_images

image_counts, total_images = count_images()

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
                    if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
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
        st.markdown("<div class=\"metric-card\">", unsafe_allow_html=True)
        st.metric("Total Crop Classes", len(crop_classes))
        st.markdown("</div>", unsafe_allow_html=True)
    
    # image_counts and total_images are now defined globally
    
    with col2:
        st.markdown("<div class=\"metric-card\">", unsafe_allow_html=True)
        st.metric("Total Images", total_images)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class=\"metric-card\">", unsafe_allow_html=True)
        avg_images = total_images / len(crop_classes) if crop_classes else 0
        st.metric("Avg Images per Class", f"{avg_images:.1f}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Distribution chart
    st.subheader("Class Distribution")
    df = pd.DataFrame(list(image_counts.items()), columns=["Crop", "Count"])
    fig = px.bar(df, x="Crop", y="Count", title="Number of Images per Crop Class")
    fig.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Sample images
    st.subheader("Sample Images")
    selected_crop = st.selectbox("Select a crop to view samples:", crop_classes)
    
    if selected_crop:
        crop_dir = os.path.join("agricultural_data/Agricultural-crops", selected_crop)
        if os.path.exists(crop_dir):
            image_files = [f for f in os.listdir(crop_dir) 
                          if f.lower().endswith((".png", ".jpg", ".jpeg"))][:6]
            
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
            st.info("🚀 Using EfficientNet-B3 architecture")
        elif "EfficientNet-B7" in model_type:
            model = NASNetCropClassifier(len(crop_classes)).to(device)
            st.info("🌌 Using NASNet-equivalent (EfficientNet-B7) architecture")
        elif "ResNet101" in model_type:
            model = CropClassifierAlt(len(crop_classes)).to(device)
            st.info("✨ Using ResNet101 architecture")
        elif "AgriCropNet" in model_type:
            model = AgriCropNet(len(crop_classes)).to(device)
            st.info("🌿 Using AgriCropNet (Multi-scale) architecture")
        else:
            st.error("Invalid model type selected.")
            st.stop()
        
        st.info(f"Using {model_type} with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
        
        # Enhanced loss function with class weighting
        # Calculate class weights for imbalanced dataset
        class_counts = [image_counts.get(crop, 0) for crop in crop_classes]
        total_samples = sum(class_counts)
        class_weights = [total_samples / (len(crop_classes) * count) if count > 0 else 1.0 for count in class_counts]
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        if use_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        
        # Training loop
        st.subheader("Training Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()
        chart_data = pd.DataFrame(columns=["Epoch", "Loss", "Accuracy", "Type"])
        loss_chart = st.line_chart(chart_data)
        
        best_val_accuracy = 0.0
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                
                progress_bar.progress((i + 1) / len(train_loader))
                status_text.text(f"Epoch {epoch+1}/{epochs} - Batch {i+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
            
            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = correct_train / total_train
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
            
            val_epoch_loss = val_loss / len(val_loader)
            val_epoch_accuracy = correct_val / total_val
            
            # Update learning rate scheduler
            if use_scheduler:
                scheduler.step(val_epoch_loss)
            
            # Update charts
            new_train_row = pd.DataFrame([{"Epoch": epoch + 1, "Loss": epoch_loss, "Accuracy": epoch_accuracy, "Type": "Training"}])
            new_val_row = pd.DataFrame([{"Epoch": epoch + 1, "Loss": val_epoch_loss, "Accuracy": val_epoch_accuracy, "Type": "Validation"}])
            chart_data = pd.concat([chart_data, new_train_row, new_val_row], ignore_index=True)
            loss_chart.line_chart(chart_data.set_index("Epoch")[["Loss", "Accuracy"]])
            
            status_text.text(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f} | Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_accuracy:.4f}")
            
            # Save best model
            if val_epoch_accuracy > best_val_accuracy:
                best_val_accuracy = val_epoch_accuracy
                torch.save(model.state_dict(), "best_crop_classifier.pth")
                st.success(f"✅ Saved best model with Validation Accuracy: {best_val_accuracy:.4f}")
        
        st.success("Training Complete!")

elif page == "📈 Model Evaluation":
    st.header("Model Evaluation")
    
    if not crop_classes:
        st.error("Dataset not found!")
        st.stop()

    st.subheader("Load Model and Evaluate")
    model_path = st.text_input("Path to saved model (.pth)", "best_crop_classifier.pth")
    
    if st.button("Evaluate Model", type="primary"):
        if not os.path.exists(model_path):
            st.error(f"Model file not found at {model_path}")
            st.stop()
        
        # Data transforms for evaluation
        eval_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Create dataset for evaluation (using full dataset for comprehensive evaluation)
        full_dataset = CropDataset("agricultural_data/Agricultural-crops", eval_transform)
        eval_loader = DataLoader(full_dataset, batch_size=32, shuffle=False)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model (assuming ResNet50 for evaluation, adjust if needed)
        model = CropClassifier(len(crop_classes)).to(device) # Or load based on saved model type
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for inputs, labels in eval_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        
        st.subheader("Evaluation Results")
        
        # Classification Report
        report = classification_report(all_labels, all_predictions, target_names=crop_classes, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        st.dataframe(df_report)
        
        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_predictions)
        fig_cm = px.imshow(cm, 
                           labels=dict(x="Predicted", y="True", color="Count"),
                           x=crop_classes,
                           y=crop_classes,
                           color_continuous_scale="Viridis",
                           title="Confusion Matrix")
        st.plotly_chart(fig_cm, use_container_width=True)
        
        st.success("Model Evaluation Complete!")

elif page == "🔮 Crop Prediction":
    st.header("Crop Prediction")
    
    if not crop_classes:
        st.error("Dataset not found!")
        st.stop()

    st.subheader("Upload Image for Prediction")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    model_path = st.text_input("Path to saved model (.pth)", "best_crop_classifier.pth")
    
    if uploaded_file is not None and st.button("Predict Crop", type="primary"):
        if not os.path.exists(model_path):
            st.error(f"Model file not found at {model_path}")
            st.stop()
            
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Preprocess image
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model (assuming ResNet50 for prediction, adjust if needed)
        model = CropClassifier(len(crop_classes)).to(device) # Or load based on saved model type
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        with torch.no_grad():
            output = model(input_batch.to(device))
            probabilities = F.softmax(output, dim=1)
            
        # Get top 5 predictions
        top5_prob, top5_idx = torch.topk(probabilities, 5)
        
        st.subheader("Prediction Results")
        
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()
        predicted_class_name = crop_classes[predicted_class_idx]
        confidence = probabilities[0][predicted_class_idx].item() * 100
        
        st.success(f"Predicted Crop: **{predicted_class_name}** with **{confidence:.2f}%** confidence.")
        
        st.write("Top 5 Predictions:")
        for i in range(top5_prob.size(1)):
            class_name = crop_classes[top5_idx[0][i].item()]
            prob = top5_prob[0][i].item() * 100
            st.write(f"- {class_name}: {prob:.2f}%")



