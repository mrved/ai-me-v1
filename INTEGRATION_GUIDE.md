# Quick Integration Guide for Sach Engineering

## 🎯 Quick Start - Get Running in 1 Day

### Prerequisites
- Python 3.9+ installed
- Access to simulation database
- Historical FEA/CFD results (100+ simulations)

### Step 1: Install System (30 minutes)
```bash
# Clone or copy the repository
cd /path/to/ai-me-v1

# Install dependencies
pip install -r requirements.txt

# Verify installation
python src/evaluate_model.py
```

### Step 2: Import Your Data (2-4 hours)

#### Option A: If you have a database
```python
# Create import script: src/import_sach_data.py
import pandas as pd
from sqlalchemy import create_engine

# Connect to your database
sach_db = create_engine("postgresql://user:pass@host/dbname")

# Query your simulation results
query = """
SELECT 
    design_id,
    length, width, height,
    load_force,
    max_stress,
    max_deflection
FROM simulation_results
"""

df = pd.read_sql(query, sach_db)

# Map to our format
df_mapped = pd.DataFrame({
    'length': df['length'],
    'width': df['width'],
    'height': df['height'],
    'load': df['load_force'],
    'max_stress': df['max_stress'],
    'max_deflection': df['max_deflection']
})

# Save to our database
from sqlalchemy import create_engine
engine = create_engine("sqlite:///data/metadata.db")
df_mapped.to_sql('simulations', engine, if_exists='replace', index=False)
```

#### Option B: If you have CSV/Excel files
```python
# Load your data
df = pd.read_csv('sach_simulation_results.csv')

# Map columns and save
# (similar to Option A)
```

### Step 3: Retrain Model (15 minutes)
```bash
python src/train.py
```

### Step 4: Validate Accuracy (30 minutes)
```bash
python src/evaluate_model.py
```

Check that:
- R² Score > 0.90
- Accuracy > 80% within 10% tolerance
- RMSE is reasonable for your use case

### Step 5: Launch Dashboard (5 minutes)
```bash
streamlit run src/dashboard.py
```

Access at: `http://localhost:8501`

### Step 6: Test with Real Designs (1 hour)
1. Enter a known design in the dashboard
2. Compare AI prediction vs actual simulation result
3. Validate accuracy meets requirements

---

## 🔌 Integration Options

### Option 1: Web Application (Easiest)
**Best for**: Quick deployment, team access

**Steps:**
1. Deploy on internal server
2. Share URL with team
3. Users access via browser

**Deployment:**
```bash
# On server
streamlit run src/dashboard.py --server.port 8501 --server.address 0.0.0.0
```

### Option 2: API Integration
**Best for**: Integration with existing tools

**Create API wrapper:**
```python
# src/api_server.py
from fastapi import FastAPI
from src.inference import predict

app = FastAPI()

@app.post("/predict")
def predict_stress(length: float, width: float, height: float, load: float):
    result = predict(length, width, height, load)
    return {"predicted_stress": result}
```

**Usage:**
```bash
# Start API
uvicorn src.api_server:app --host 0.0.0.0 --port 8000

# Call from other tools
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"length": 4.5, "width": 1.8, "height": 1.6, "load": 15000}'
```

### Option 3: CAD Plugin
**Best for**: Seamless design workflow

**Example SolidWorks plugin:**
```python
# Extract dimensions from SolidWorks
import win32com.client

swApp = win32com.client.Dispatch("SldWorks.Application")
model = swApp.ActiveDoc

# Get dimensions
length = model.GetDimension("D1@Sketch1").Value
width = model.GetDimension("D2@Sketch1").Value
height = model.GetDimension("D3@Sketch1").Value

# Predict stress
stress = predict(length, width, height, 15000)

# Display result
swApp.SendMsgToUser(f"Predicted Stress: {stress:,.0f} Pa")
```

---

## 📊 Data Format Requirements

### Required Fields
- `length` (meters): Design length
- `width` (meters): Design width  
- `height` (meters): Design height
- `load` (Newtons): Applied load
- `max_stress` (Pascals): Maximum stress (for training)
- `max_deflection` (meters): Maximum deflection (optional)

### Example Data
```csv
length,width,height,load,max_stress,max_deflection
4.5,1.8,1.6,15000,75000,0.00015
4.2,1.7,1.5,12000,68000,0.00012
...
```

---

## 🎓 Training Your Team

### For Engineers
1. **Dashboard basics** (30 min)
   - How to enter design parameters
   - How to interpret results
   - When to trust vs verify predictions

2. **Best practices** (15 min)
   - Use AI for initial screening
   - Verify critical designs with full simulation
   - Report discrepancies for model improvement

### For Administrators
1. **System maintenance** (1 hour)
   - How to retrain model
   - How to add new data
   - How to monitor performance

---

## 🔍 Monitoring & Maintenance

### Weekly Tasks
- Check model accuracy on new predictions
- Review any prediction errors
- Collect feedback from users

### Monthly Tasks
- Retrain model with new data
- Update dashboard features
- Review system performance

### When to Retrain
- After 50+ new simulations
- When accuracy drops below threshold
- When new design types are introduced

---

## 🆘 Troubleshooting

### Model accuracy too low
- **Solution**: Add more training data
- **Check**: Data quality and format
- **Action**: Retrain with larger dataset

### Predictions seem wrong
- **Solution**: Verify input parameters match training data range
- **Check**: Feature importance - are you using right parameters?
- **Action**: Expand training data to cover your design space

### Dashboard not loading
- **Solution**: Check if Streamlit is running
- **Check**: Database file exists
- **Action**: Restart dashboard, check logs

---

## 📞 Support

For integration help:
1. Check this guide
2. Review ROADMAP.md for detailed plan
3. Contact development team

---

**Ready to integrate? Start with Step 1 above!**

