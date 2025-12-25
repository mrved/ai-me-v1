"""
Helper script to download DrivAerNet++ dataset
Provides instructions and download links
"""
import os
import subprocess
from pathlib import Path

def check_git_lfs():
    """Check if git-lfs is installed"""
    try:
        subprocess.run(['git', 'lfs', 'version'], capture_output=True, check=True)
        return True
    except:
        return False

def download_via_git():
    """Download via git (if repository is available)"""
    print("=" * 60)
    print("Downloading DrivAerNet++ via Git")
    print("=" * 60)
    
    if not check_git_lfs():
        print("⚠️  Git LFS not installed. Installing...")
        print("   macOS: brew install git-lfs")
        print("   Linux: sudo apt install git-lfs")
        print("   Then run: git lfs install")
        return False
    
    dataset_dir = Path("data/drivaernet")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    repo_url = "https://github.com/Mohamedelrefaie/DrivAerNet.git"
    
    print(f"\n📥 Cloning repository to {dataset_dir}...")
    print("   Note: This is a large dataset (several GB)")
    print("   The download may take a while...")
    
    try:
        if dataset_dir.exists() and any(dataset_dir.iterdir()):
            print(f"⚠️  Directory {dataset_dir} already exists and is not empty")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return False
        
        subprocess.run(['git', 'clone', repo_url, str(dataset_dir)], check=True)
        print("✅ Download complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        return False

def manual_download_instructions():
    """Print manual download instructions"""
    print("=" * 60)
    print("Manual Download Instructions")
    print("=" * 60)
    print("\n1. Visit: https://github.com/Mohamedelrefaie/DrivAerNet")
    print("2. Check the README for download links")
    print("3. Download the dataset files")
    print("4. Extract to: data/drivaernet/")
    print("\n5. Then run:")
    print("   python src/import_drivaernet.py --dataset-path data/drivaernet")
    print("\nAlternative sources:")
    print("- Paper: https://arxiv.org/abs/2406.09624")
    print("- Check for direct download links in the repository")

def main():
    print("DrivAerNet++ Dataset Downloader")
    print("=" * 60)
    print("\nThis script helps you download the DrivAerNet++ dataset")
    print("which contains 8,000 car designs with CFD simulations.\n")
    
    print("Options:")
    print("1. Download via Git (if repository is public)")
    print("2. Manual download instructions")
    print("3. Use existing dataset path")
    
    choice = input("\nChoose option (1/2/3): ").strip()
    
    if choice == "1":
        if download_via_git():
            print("\n✅ Dataset downloaded! Now run:")
            print("   python src/import_drivaernet.py --dataset-path data/drivaernet")
    elif choice == "2":
        manual_download_instructions()
    elif choice == "3":
        path = input("Enter dataset path: ").strip()
        if os.path.exists(path):
            print(f"\n✅ Using dataset at: {path}")
            print("Run: python src/import_drivaernet.py --dataset-path", path)
        else:
            print(f"❌ Path not found: {path}")
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()

