# Car Design Parameters & Data Sources

## Current Parameters (4 parameters)

We're currently using **4 basic parameters**:

1. **Length** (3.5 - 5.5 m) - Overall car length
2. **Width** (1.6 - 2.0 m) - Car width (track width)
3. **Height** (1.4 - 1.8 m) - Overall car height
4. **Load** (5,000 - 25,000 N) - Aerodynamic drag force

### What We're Missing (Important Car Design Parameters)

#### Geometry Parameters:
- **Wheelbase** - Distance between front and rear axles
- **Front overhang** - Distance from front axle to bumper
- **Rear overhang** - Distance from rear axle to bumper
- **Ground clearance** - Height from ground to chassis
- **Roof angle** - Windshield/roof slope (affects aerodynamics)
- **Frontal area** - Cross-sectional area (affects drag)

#### Material Properties:
- **Material type** - Steel, Aluminum, Carbon Fiber, etc.
- **Young's Modulus** - Material stiffness
- **Yield strength** - Material strength limit
- **Density** - Affects weight calculations
- **Thickness** - Body panel thickness

#### Loading Conditions:
- **Speed** - Vehicle speed (affects aerodynamic load)
- **Cornering force** - Lateral loads in turns
- **Braking force** - Deceleration loads
- **Payload** - Passenger/cargo weight
- **Wind speed/direction** - Environmental factors

#### Design Features:
- **Drag coefficient (Cd)** - Aerodynamic efficiency
- **Weight** - Total vehicle mass
- **Center of gravity** - Affects handling and stress distribution
- **Suspension type** - Affects load distribution

---

## Data Sources

### ❌ Current: Synthetic Data Only

**We did NOT use publicly available data.** We generated synthetic data using:
- Physics-based calculations (beam theory approximations)
- Realistic parameter ranges based on typical car dimensions
- Simplified stress calculations

**Limitations:**
- Not based on real car designs
- Simplified physics (not full FEA/CFD)
- Limited to basic geometry

### ✅ Available Public Datasets (Not Yet Integrated)

#### 1. **DrivAerNet++** (Recommended)
- **8,000+ car designs** with high-fidelity CFD simulations
- Includes 3D meshes, aerodynamic coefficients, flow fields
- **Source**: https://arxiv.org/abs/2406.09624
- **Best for**: Aerodynamic analysis, real car geometries

#### 2. **MIT Open-Source Car Design Dataset**
- **8,000+ car designs** with aerodynamic simulations
- Developed by MIT engineers
- **Source**: https://news.mit.edu/2024/design-future-car-with-8000-design-options-1205
- **Best for**: Design optimization, eco-friendly vehicles

#### 3. **ABC Dataset** (Geometry Only)
- Large collection of 3D CAD models
- **Source**: https://deep-geometry.github.io/abc-dataset/
- **Note**: We have code to process this (`process_abc.py`) but haven't used it yet
- **Best for**: Geometry variety, but needs physics simulation

---

## Recommended Next Steps

### Phase 1: Expand Parameters (Quick Win)
Add these parameters to improve accuracy:

```python
# Additional parameters to add:
- wheelbase
- material_type (steel/aluminum/carbon_fiber)
- drag_coefficient
- vehicle_speed
- vehicle_weight
```

### Phase 2: Integrate Public Data (Better Accuracy)
1. **Download DrivAerNet++ dataset**
2. **Process and import** into our pipeline
3. **Retrain model** with real car data
4. **Compare accuracy** - should improve significantly

### Phase 3: Real Simulation Integration
1. **Connect to FEA/CFD solvers** (ANSYS, Abaqus, OpenFOAM)
2. **Run real simulations** on designs
3. **Replace synthetic physics** with actual results
4. **Build comprehensive database** of real simulation results

---

## Current Model Performance

With only 4 parameters and synthetic data:
- **R² Score**: 0.97 (excellent!)
- **Accuracy**: 84% within 10% tolerance
- **But**: Limited to simplified scenarios

**With more parameters and real data:**
- Expected R²: 0.98-0.99
- Expected accuracy: 90%+ within 5% tolerance
- Can handle real-world design scenarios

---

## How to Add More Parameters

### 1. Update Data Generation

```python
# In generate_data.py, add:
wheelbase = random.uniform(2.5, 3.2)
material_type = random.choice(['steel', 'aluminum', 'carbon_fiber'])
drag_coefficient = random.uniform(0.25, 0.35)
vehicle_speed = random.uniform(60, 120)  # km/h
```

### 2. Update Physics Calculations

```python
# Use material properties based on type
if material_type == 'steel':
    E = 200e9
    yield_strength = 250e6
elif material_type == 'aluminum':
    E = 70e9
    yield_strength = 275e6
# etc.
```

### 3. Update Model Training

```python
# In train.py, include new features:
X = df[['length', 'width', 'height', 'load', 
        'wheelbase', 'material_type_encoded', 
        'drag_coefficient', 'vehicle_speed']]
```

### 4. Update Dashboard

```python
# Add input fields for new parameters
wheelbase = st.number_input("Wheelbase (m)", 2.5, 3.2, 2.8)
material = st.selectbox("Material", ['Steel', 'Aluminum', 'Carbon Fiber'])
```

---

## Integration Plan for Public Data

### DrivAerNet++ Integration:

1. **Download dataset** (if available)
2. **Parse data format** (likely HDF5 or similar)
3. **Extract features**:
   - Geometry parameters
   - Aerodynamic coefficients
   - Stress/strain data
4. **Import to our database**:
   ```python
   python src/import_drivaernet.py --dataset-path /path/to/drivaernet
   ```
5. **Retrain model**:
   ```python
   python src/train.py
   ```

### Expected Improvements:
- **10x more training data** (8,000 vs 50 designs)
- **Real aerodynamic data** (not approximations)
- **Actual stress distributions** (from CFD)
- **Better model accuracy** and generalization

---

## Summary

**Current State:**
- ✅ 4 basic parameters (length, width, height, load)
- ✅ Synthetic data (physics-based calculations)
- ✅ Good model performance (97% R²)
- ❌ No public datasets used yet
- ❌ Limited parameters

**Recommended:**
1. **Short-term**: Add 5-10 more parameters (wheelbase, material, etc.)
2. **Medium-term**: Integrate DrivAerNet++ or MIT dataset
3. **Long-term**: Connect to real FEA/CFD simulations

**Impact:**
- More parameters = Better predictions
- Real data = More accurate models
- Public datasets = Faster development

