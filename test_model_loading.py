#!/usr/bin/env python3
"""
Test script to verify model loading compatibility with PyTorch 2.6+
"""

import torch
import os
import sys

def test_model_loading():
    """Test loading models with PyTorch 2.6+ compatibility fixes"""
    
    print("🔧 Testing model loading compatibility...")
    
    # Check available model files
    model_files = ['crop_model_deployment.pth', 'final_crop_model.pth', 'best_nasnet_crop_model.pth']
    available_models = [f for f in model_files if os.path.exists(f)]
    
    if not available_models:
        print("❌ No model files found!")
        print("Available files in current directory:")
        for file in os.listdir('.'):
            if file.endswith('.pth'):
                print(f"  - {file}")
        return False
    
    print(f"✅ Found model files: {available_models}")
    
    # Test loading each available model
    for model_file in available_models:
        print(f"\n🔍 Testing {model_file}...")
        
        try:
            # Add safe globals for numpy arrays to handle PyTorch 2.6 compatibility
            import torch.serialization
            torch.serialization.add_safe_globals(['numpy.core.multiarray.scalar'])
            
            # Load model with compatibility fixes
            state_dict = torch.load(model_file, map_location='cpu', weights_only=False)
            
            print(f"✅ Successfully loaded {model_file}")
            print(f"   - State dict keys: {len(state_dict.keys())}")
            print(f"   - Sample keys: {list(state_dict.keys())[:5]}")
            
        except Exception as e:
            print(f"❌ Failed to load {model_file}: {str(e)}")
            return False
    
    print("\n🎉 All model loading tests passed!")
    return True

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)
