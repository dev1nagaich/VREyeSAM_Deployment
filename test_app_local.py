#!/usr/bin/env python3
"""
Local Testing Script for VREyeSAM Streamlit App

Run this script to test the app locally before deploying to Hugging Face Spaces.
Usage: python test_app_local.py
"""

import subprocess
import sys
import os
import time

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'streamlit',
        'torch',
        'torchvision',
        'opencv-python',
        'numpy',
        'PIL'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install them with: pip install -r requirements_deploy.txt")
        return False
    
    print("✅ All dependencies installed\n")
    return True

def check_model_files():
    """Check if model files exist"""
    print("🔍 Checking model files...")
    
    files_to_check = [
        "segment-anything-2/checkpoints/sam2_hiera_small.pt",
        "segment-anything-2/checkpoints/VREyeSAM_uncertainity_best.torch"
    ]
    
    all_exist = True
    for file_path in files_to_check:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"   ✅ {file_path} ({size_mb:.1f} MB)")
        else:
            print(f"   ❌ {file_path} - NOT FOUND")
            all_exist = False
    
    if not all_exist:
        print("\n⚠️  Some model files are missing!")
        print("Please run the setup instructions from README.md")
        return False
    
    print("✅ All model files present\n")
    return True

def check_sam2_installation():
    """Check if SAM2 is properly installed"""
    print("🔍 Checking SAM2 installation...")
    
    try:
        sys.path.insert(0, "segment-anything-2")
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        print("   ✅ SAM2 modules can be imported")
        print("✅ SAM2 properly installed\n")
        return True
    except ImportError as e:
        print(f"   ❌ SAM2 import failed: {e}")
        print("\n⚠️  SAM2 not properly installed!")
        print("Install with:")
        print("  git clone https://github.com/facebookresearch/segment-anything-2")
        print("  cd segment-anything-2")
        print("  pip install -e .")
        return False

def test_app_syntax():
    """Check if app.py has syntax errors"""
    print("🔍 Checking app.py syntax...")
    
    try:
        with open('app.py', 'r', encoding='utf-8') as f:
            code = f.read()
            compile(code, 'app.py', 'exec')
        print("   ✅ No syntax errors")
        print("✅ app.py syntax valid\n")
        return True
    except SyntaxError as e:
        print(f"   ❌ Syntax error in app.py: {e}")
        return False
    except UnicodeDecodeError as e:
        print(f"   ⚠️  Unicode encoding issue: {e}")
        print("   Trying with different encoding...")
        try:
            with open('app.py', 'r', encoding='latin-1') as f:
                code = f.read()
                compile(code, 'app.py', 'exec')
            print("   ✅ No syntax errors (latin-1 encoding)")
            print("✅ app.py syntax valid\n")
            return True
        except Exception as e2:
            print(f"   ❌ Still failed: {e2}")
            return False

def run_streamlit_app():
    """Launch the Streamlit app"""
    print("🚀 Launching Streamlit app...")
    print("=" * 60)
    print("The app will open in your browser at http://localhost:8501")
    print("Press Ctrl+C to stop the app")
    print("=" * 60)
    print()
    
    try:
        subprocess.run(['streamlit', 'run', 'app.py'], check=True)
    except KeyboardInterrupt:
        print("\n\n✅ App stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running app: {e}")
        return False
    
    return True

def create_test_image():
    """Create a simple test image if none exists"""
    print("🔍 Checking for test images...")
    
    test_dir = "test_images"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print(f"   📁 Created {test_dir} directory")
    
    # Check if there are any test images
    image_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if image_files:
        print(f"   ✅ Found {len(image_files)} test image(s)")
        print(f"   📂 Test images in: {test_dir}/")
        for img in image_files:
            print(f"      - {img}")
    else:
        print(f"   ℹ️  No test images found in {test_dir}/")
        print(f"   💡 Add some iris images to {test_dir}/ for testing")
    
    print()

def main():
    """Main testing function"""
    print("\n" + "=" * 60)
    print("VREyeSAM Local Testing Suite")
    print("=" * 60 + "\n")
    
    # Run all checks
    checks = [
        ("Dependencies", check_dependencies),
        ("Model Files", check_model_files),
        ("SAM2 Installation", check_sam2_installation),
        ("App Syntax", test_app_syntax),
    ]
    
    all_passed = True
    for name, check_func in checks:
        if not check_func():
            all_passed = False
            print(f"❌ {name} check failed\n")
    
    # Create test image directory
    create_test_image()
    
    if not all_passed:
        print("=" * 60)
        print("⚠️  Some checks failed. Please fix the issues above.")
        print("=" * 60)
        sys.exit(1)
    
    print("=" * 60)
    print("✅ All checks passed! Ready to run the app.")
    print("=" * 60)
    print()
    
    # Ask user if they want to run the app
    response = input("Do you want to launch the app now? (y/n): ").strip().lower()
    
    if response == 'y':
        run_streamlit_app()
    else:
        print("\n✅ Testing complete!")
        print("To run the app manually, execute: streamlit run app.py")
        print()

if __name__ == "__main__":
    main()