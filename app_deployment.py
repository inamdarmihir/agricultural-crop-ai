import streamlit as st
import torch
import torch.nn as nn
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
import gc
import warnings
warnings.filterwarnings('ignore')

# Set page config for deployment
st.set_page_config(
    page_title="Agricultural Crop Classification",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for deployment
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
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
    .stAlert {
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #2E8B57;
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

# Memory management function
def clear_memory():
    """Clear memory and garbage collect"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Load crop classes and image counts with caching
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_crop_classes():
    data_path = "agricultural_data/Agricultural-crops"
    if os.path.exists(data_path):
        return sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    return []

@st.cache_data(ttl=3600)  # Cache for 1 hour
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

# Get crop classes
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
        try:
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            st.error(f"Error loading image {img_path}: {str(e)}")
            # Return a placeholder image
            placeholder = Image.new('RGB', (224, 224), color='gray')
            if self.transform:
                placeholder = self.transform(placeholder)
            return placeholder, label

# Model architectures
class CropClassifier(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(CropClassifier, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class NASNetCropClassifier(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(NASNetCropClassifier, self).__init__()
        self.backbone = models.efficientnet_b7(pretrained=True)
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
                    try:
                        img_path = os.path.join(crop_dir, img_file)
                        image = Image.open(img_path)
                        st.image(image, caption=f"{selected_crop} - {img_file}", use_column_width=True)
                    except Exception as e:
                        st.error(f"Error loading image: {str(e)}")

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
        epochs = st.slider("Number of Epochs", 1, 20, 5)  # Reduced for deployment
        batch_size = st.selectbox("Batch Size", [8, 16, 32], index=1)
        learning_rate = st.selectbox("Learning Rate", [0.001, 0.01], index=0)
    
    with col2:
        train_split = st.slider("Training Split (%)", 70, 90, 80)
        model_type = st.selectbox("Model Architecture", [
            "CropClassifier (ResNet50)",
            "NASNetCropClassifier (EfficientNet-B7)"
        ])
        data_augmentation = st.checkbox("Enable Data Augmentation", value=True)
    
    if st.button("Start Training", type="primary"):
        try:
            with st.spinner("Setting up training..."):
                # Data transforms
                if data_augmentation:
                    train_transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomRotation(degrees=15),
                        transforms.ColorJitter(brightness=0.2, contrast=0.2),
                        transforms.ToTensor(),
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
            
            # Model setup
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            st.info(f"Using device: {device}")
            
            if "ResNet50" in model_type:
                model = CropClassifier(len(crop_classes)).to(device)
            else:
                model = NASNetCropClassifier(len(crop_classes)).to(device)
            
            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            
            # Training loop
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            train_losses = []
            val_losses = []
            
            for epoch in range(epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        loss = criterion(output, target)
                        val_loss += loss.item()
                
                # Calculate metrics
                train_loss /= len(train_loader)
                val_loss /= len(val_loader)
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                # Update progress
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                status_text.text(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save model
            torch.save(model.state_dict(), 'crop_model_deployment.pth')
            
            st.success("Training completed! Model saved as 'crop_model_deployment.pth'")
            
            # Plot training history
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(train_losses, label='Train Loss')
            ax.plot(val_losses, label='Validation Loss')
            ax.set_title('Training and Validation Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            st.pyplot(fig)
            
            # Clear memory
            clear_memory()
            
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            st.info("Please check your dataset and try again.")

# Model Evaluation Page
elif page == "📈 Model Evaluation":
    st.header("Model Evaluation")
    
    model_files = ['crop_model_deployment.pth', 'final_crop_model.pth', 'best_nasnet_crop_model.pth']
    model_file = None
    
    for file in model_files:
        if os.path.exists(file):
            model_file = file
            break
    
    if not model_file:
        st.warning("No trained model found! Please train a model first.")
        st.stop()
    
    st.info(f"Using model: {model_file}")
    
    # Load model
    @st.cache_resource
    def load_model():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            state_dict = torch.load(model_file, map_location=device)
            
            # Detect model architecture
            if 'efficientnet' in str(state_dict).lower():
                model = NASNetCropClassifier(len(crop_classes)).to(device)
            else:
                model = CropClassifier(len(crop_classes)).to(device)
            
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            return model, device
            
        except Exception as e:
            st.error(f"Model loading failed: {str(e)}")
            return None, device
    
    model, device = load_model()
    
    if model is None:
        st.stop()
    
    # Evaluation
    if st.button("Evaluate Model", type="primary"):
        try:
            with st.spinner("Evaluating model..."):
                # Create test dataset
                test_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
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
                
                # Confusion Matrix
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(all_labels, all_predictions)
                
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=crop_classes, yticklabels=crop_classes, ax=ax)
                ax.set_title('Confusion Matrix')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                st.pyplot(fig)
                
                # Clear memory
                clear_memory()
                
        except Exception as e:
            st.error(f"Evaluation failed: {str(e)}")

# Crop Prediction Page
elif page == "🔮 Crop Prediction":
    st.header("Crop Prediction")
    
    model_files = ['crop_model_deployment.pth', 'final_crop_model.pth', 'best_nasnet_crop_model.pth']
    model_file = None
    
    for file in model_files:
        if os.path.exists(file):
            model_file = file
            break
    
    if not model_file:
        st.warning("No trained model found! Please train a model first.")
        st.stop()
    
    # Load model
    @st.cache_resource
    def load_prediction_model():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            state_dict = torch.load(model_file, map_location=device)
            
            # Detect model architecture
            if 'efficientnet' in str(state_dict).lower():
                model = NASNetCropClassifier(len(crop_classes)).to(device)
            else:
                model = CropClassifier(len(crop_classes)).to(device)
            
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            return model, device
            
        except Exception as e:
            st.error(f"Model loading failed: {str(e)}")
            return None, device
    
    model, device = load_prediction_model()
    
    if model is None:
        st.stop()
    
    # Image upload
    uploaded_file = st.file_uploader("Choose a crop image...", 
                                   type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        try:
            # Display image
            image = Image.open(uploaded_file).convert('RGB')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                # Preprocess image
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
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
                
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
    
    # Clear memory
    clear_memory()

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and PyTorch")

# Add deployment info
st.sidebar.markdown("---")
st.sidebar.markdown("### Deployment Info")
st.sidebar.info("✅ Ready for Streamlit deployment\n\n🌐 Optimized for production\n\n💾 Memory efficient")
