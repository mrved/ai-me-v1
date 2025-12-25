"""
Setup script to generate data and train model for Streamlit Cloud deployment
Run this once before deploying or it will run automatically on first dashboard load
"""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

def setup():
    print("🚀 Setting up AI Engineering Dashboard...")
    print("=" * 60)
    
    # Step 1: Generate data
    print("\n📊 Step 1: Generating car design data...")
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src" / "generate_data.py")],
        cwd=str(PROJECT_ROOT)
    )
    if result.returncode != 0:
        print("❌ Failed to generate data")
        return False
    print("✅ Data generated")
    
    # Step 2: Run ETL
    print("\n🔄 Step 2: Running ETL pipeline...")
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src" / "etl.py")],
        cwd=str(PROJECT_ROOT)
    )
    if result.returncode != 0:
        print("❌ Failed to run ETL")
        return False
    print("✅ ETL completed")
    
    # Step 2.5: Import real DrivAerNet++ data if available
    print("\n📥 Step 2.5: Importing real car design data...")
    drivaernet_csv = PROJECT_ROOT / "data" / "drivaernet" / "ParametricModels" / "DrivAerNet_ParametricData.csv"
    if drivaernet_csv.exists():
        print(f"   Found DrivAerNet++ data at {drivaernet_csv}")
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "src" / "import_drivaernet_csv.py")],
            cwd=str(PROJECT_ROOT)
        )
        if result.returncode == 0:
            print("✅ Real data imported")
        else:
            print("⚠️  Real data import failed, continuing with synthetic data")
    else:
        print("   No real data found, using synthetic data only")
    
    # Step 3: Train model
    print("\n🤖 Step 3: Training ML model...")
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src" / "train.py")],
        cwd=str(PROJECT_ROOT)
    )
    if result.returncode != 0:
        print("❌ Failed to train model")
        return False
    print("✅ Model trained")
    
    print("\n" + "=" * 60)
    print("✅ Setup complete! Dashboard is ready to use.")
    return True

if __name__ == "__main__":
    success = setup()
    sys.exit(0 if success else 1)

