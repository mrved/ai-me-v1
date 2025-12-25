# Real Data Integration Guide

## 🎯 Goal: Integrate DrivAerNet++ Dataset

DrivAerNet++ provides **8,000 real car designs** with:
- 3D meshes
- CFD simulation results
- Aerodynamic coefficients
- Pressure/velocity fields

## 📥 Step 1: Download Dataset

### Option A: Via Git (Recommended)
```bash
# Install git-lfs if needed
brew install git-lfs  # macOS
# or
sudo apt install git-lfs  # Linux

git lfs install

# Clone repository
cd data
git clone https://github.com/Mohamedelrefaie/DrivAerNet.git drivaernet
```

### Option B: Manual Download
1. Visit: https://github.com/Mohamedelrefaie/DrivAerNet
2. Follow download instructions in README
3. Extract to `data/drivaernet/`

### Option C: Use Helper Script
```bash
python src/download_drivaernet.py
```

## 📊 Step 2: Import Data

```bash
# Import the dataset
python src/import_drivaernet.py --dataset-path data/drivaernet

# Or import a subset for testing
python src/import_drivaernet.py --dataset-path data/drivaernet --limit 1000
```

### Import Options:
- `--dataset-path`: Path to DrivAerNet++ dataset
- `--replace`: Replace existing data (default: append)
- `--limit N`: Import only first N records (for testing)

## 🔄 Step 3: Retrain Model

```bash
# Retrain with real data
python src/train.py

# Evaluate new model
python src/evaluate_model.py
```

## 📈 Expected Improvements

**Before (Synthetic Data):**
- 50 designs
- 4 parameters
- R²: 0.97
- Accuracy: 84% (10% tolerance)

**After (Real Data):**
- 8,000 designs (160x more!)
- More parameters (drag coefficient, etc.)
- Expected R²: 0.98-0.99
- Expected accuracy: 90%+ (5% tolerance)

## 🔧 Custom Data Format

If your dataset has a different format, you can:

1. **Modify the parser** in `src/import_drivaernet.py`
2. **Create a custom importer** following the same pattern
3. **Use CSV/JSON** - the importer supports both

### Example: Custom CSV Format

```python
# If you have a CSV with columns: length, width, height, drag_force, max_stress
import pandas as pd
from sqlalchemy import create_engine

df = pd.read_csv('your_data.csv')
engine = create_engine("sqlite:///data/metadata.db")

# Map to our format
df_mapped = pd.DataFrame({
    'length': df['length'],
    'width': df['width'],
    'height': df['height'],
    'load': df['drag_force'],
    'max_stress': df['max_stress'],
    'max_deflection': df.get('max_deflection', df['max_stress'] * 1e-9),
    'source': 'custom'
})

df_mapped.to_sql('simulations', engine, if_exists='append', index=False)
```

## 🐛 Troubleshooting

### "No metadata files found"
- The importer will try to parse mesh files directly
- Check that mesh files (.obj, .stl, .vtk) are in the dataset

### "Dataset path not found"
- Verify the path is correct
- Use absolute path: `--dataset-path /full/path/to/drivaernet`

### "Import is slow"
- Use `--limit` to test with a subset first
- The full 8,000 designs may take 10-30 minutes to import

### "Model accuracy didn't improve"
- Check data quality: `python -c "from sqlalchemy import create_engine; import pandas as pd; engine = create_engine('sqlite:///data/metadata.db'); df = pd.read_sql('simulations', engine); print(df.describe())"`
- Verify stress values are reasonable (not all zeros)
- Check for outliers

## 📚 Dataset Information

- **Paper**: https://arxiv.org/abs/2406.09624
- **Repository**: https://github.com/Mohamedelrefaie/DrivAerNet
- **Size**: Several GB (with full CFD data)
- **Format**: 3D meshes, JSON metadata, CSV files

## ✅ Verification

After import, verify data:

```python
from sqlalchemy import create_engine
import pandas as pd

engine = create_engine("sqlite:///data/metadata.db")
df = pd.read_sql('simulations', engine)

print(f"Total records: {len(df)}")
print(f"From DrivAerNet: {(df['source'] == 'drivaernet').sum()}")
print(df.describe())
```

## 🚀 Next Steps

1. Import dataset
2. Retrain model
3. Compare accuracy with synthetic data model
4. Update dashboard to show data source
5. Add more parameters if available in dataset

