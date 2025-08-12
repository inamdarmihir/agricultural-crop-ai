#!/usr/bin/env python3
"""
Crop AI Streamlit App Launcher
This script launches the Crop AI application with proper configuration for deployment.
"""

import os
import sys
import subprocess
import argparse

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import streamlit
        import torch
        import torchvision
        import pandas
        import numpy
        import PIL
        import matplotlib
        import seaborn
        import plotly
        import sklearn
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def check_dataset():
    """Check if the dataset is available"""
    dataset_path = "agricultural_data/Agricultural-crops"
    if os.path.exists(dataset_path):
        crop_classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        print(f"✅ Dataset found with {len(crop_classes)} crop classes")
        return True
    else:
        print("❌ Dataset not found at agricultural_data/Agricultural-crops/")
        print("Please ensure the dataset is properly extracted")
        return False

def check_models():
    """Check if any trained models are available"""
    model_files = ['crop_model_deployment.pth', 'final_crop_model.pth', 'best_nasnet_crop_model.pth']
    available_models = [f for f in model_files if os.path.exists(f)]
    
    if available_models:
        print(f"✅ Found models: {', '.join(available_models)}")
        return True
    else:
        print("⚠️  No trained models found")
        print("You can train a model using the app's training page")
        return True  # Don't stop the app, just warn

def launch_app(port=8501, host="localhost", headless=False):
    """Launch the Streamlit app"""
    app_file = "app_deployment.py"
    
    if not os.path.exists(app_file):
        print(f"❌ App file {app_file} not found")
        return False
    
    cmd = [
        sys.executable, "-m", "streamlit", "run", app_file,
        "--server.port", str(port),
        "--server.address", host
    ]
    
    if headless:
        cmd.extend(["--server.headless", "true"])
    
    print(f"🚀 Launching Crop AI app on {host}:{port}")
    print(f"📱 App will open in your browser automatically")
    print(f"🛑 Press Ctrl+C to stop the app")
    print("-" * 50)
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n🛑 App stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch app: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Crop AI Streamlit App Launcher")
    parser.add_argument("--port", type=int, default=8501, help="Port to run the app on (default: 8501)")
    parser.add_argument("--host", default="localhost", help="Host to bind to (default: localhost)")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--check-only", action="store_true", help="Only check dependencies and exit")
    
    args = parser.parse_args()
    
    print("🌾 Crop AI - Streamlit App Launcher")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check dataset
    if not check_dataset():
        print("⚠️  App may not work properly without the dataset")
    
    # Check models
    check_models()
    
    if args.check_only:
        print("\n✅ All checks completed")
        return
    
    print("\n🚀 Starting Crop AI application...")
    
    # Launch the app
    if not launch_app(args.port, args.host, args.headless):
        sys.exit(1)

if __name__ == "__main__":
    main()
