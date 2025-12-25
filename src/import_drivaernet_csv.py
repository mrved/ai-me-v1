"""
Import DrivAerNet++ CSV data directly
This script imports the parametric CSV file with 4,165 real car designs
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from pathlib import Path

DB_PATH = "sqlite:///data/metadata.db"
CSV_PATH = "data/drivaernet/ParametricModels/DrivAerNet_ParametricData.csv"

def import_drivaernet_csv(csv_path=None, limit=None, replace=False):
    """
    Import DrivAerNet++ parametric data from CSV
    The parameters are design variations, so we normalize them to realistic car dimensions
    """
    if csv_path is None:
        csv_path = CSV_PATH
    
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    print("=" * 60)
    print("Importing DrivAerNet++ Real Car Design Data")
    print("=" * 60)
    
    # Read CSV
    print(f"\n📂 Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if limit:
        df = df.head(limit)
        print(f"   Limiting to first {limit} records")
    
    print(f"   Found {len(df)} car designs")
    print(f"   Columns: {len(df.columns)}")
    
    # Map DrivAerNet++ columns to our format
    print("\n🔄 Processing data...")
    
    df_processed = pd.DataFrame()
    
    # DrivAerNet++ parameters are design variations/offsets, not absolute dimensions
    # Normalize them to realistic car dimensions using min-max scaling
    length_raw = df['A_Car_Length'].values
    width_raw = df['A_Car_Width'].values
    height_raw = df['A_Car_Roof_Height'].values
    
    # Normalize to 0-1 range, then scale to realistic car dimensions (meters)
    length_norm = (length_raw - length_raw.min()) / (length_raw.max() - length_raw.min() + 1e-10)
    width_norm = (width_raw - width_raw.min()) / (width_raw.max() - width_raw.min() + 1e-10)
    height_norm = (height_raw - height_raw.min()) / (height_raw.max() - height_raw.min() + 1e-10)
    
    # Scale to realistic car dimensions
    df_processed['length'] = 3.5 + length_norm * 2.0  # 3.5-5.5m
    df_processed['width'] = 1.6 + width_norm * 0.4    # 1.6-2.0m
    df_processed['height'] = 1.4 + height_norm * 0.4  # 1.4-1.8m
    
    # Drag coefficient (Cd) - this is REAL aerodynamic data from CFD!
    df_processed['drag_coefficient'] = df['Average Cd'].values
    
    # Calculate drag force from Cd (real physics)
    # Drag = 0.5 * rho * Cd * A * v^2
    air_density = 1.2  # kg/m³
    velocity = 27.8  # m/s (100 km/h)
    frontal_area = df_processed['width'] * df_processed['height'] * 0.8  # Approximate frontal area
    df_processed['load'] = 0.5 * air_density * df_processed['drag_coefficient'] * frontal_area * velocity**2
    
    # Estimate stress from drag force and geometry (simplified beam theory)
    E = 200e9  # Young's Modulus for Steel (Pa)
    I = (df_processed['width'] * df_processed['height']**3) / 12  # Moment of Inertia
    df_processed['max_stress'] = (df_processed['load'] * df_processed['length'] * (df_processed['height']/2)) / I
    df_processed['max_deflection'] = (5 * (df_processed['load'] / df_processed['length']) * df_processed['length']**4) / (384 * E * I)
    
    # Add additional parameters from the dataset
    df_processed['wheelbase'] = df_processed['length'] * 0.6  # Estimate (typical ratio)
    df_processed['roof_angle'] = df['A_Car_Green_House_Angle'].values
    df_processed['windshield_angle'] = df['D_Winscreen_Inclination'].values
    df_processed['rear_window_angle'] = df['D_Rear_Window_Inclination'].values
    
    # Filename
    df_processed['filename'] = df['Experiment'].astype(str) + '.vtk'
    df_processed['source'] = 'drivaernet_real'
    
    # Select final columns for database
    final_df = df_processed[[
        'filename', 'length', 'width', 'height', 'load', 
        'max_stress', 'max_deflection', 'source',
        'drag_coefficient', 'wheelbase', 'roof_angle'
    ]].copy()
    
    # Show statistics
    print("\n📊 Data Statistics:")
    print(f"   Length: {final_df['length'].min():.2f} - {final_df['length'].max():.2f} m")
    print(f"   Width:  {final_df['width'].min():.2f} - {final_df['width'].max():.2f} m")
    print(f"   Height: {final_df['height'].min():.2f} - {final_df['height'].max():.2f} m")
    print(f"   Drag Coeff: {final_df['drag_coefficient'].min():.3f} - {final_df['drag_coefficient'].max():.3f}")
    print(f"   Load: {final_df['load'].min():.0f} - {final_df['load'].max():.0f} N")
    print(f"   Stress: {final_df['max_stress'].min():.0f} - {final_df['max_stress'].max():.0f} Pa")
    
    # Import to database
    print("\n💾 Importing to database...")
    engine = create_engine(DB_PATH)
    
    if_exists = 'replace' if replace else 'append'
    final_df.to_sql('simulations', engine, if_exists=if_exists, index=False)
    
    # Show total records
    total = pd.read_sql('SELECT COUNT(*) as count FROM simulations', engine)['count'][0]
    drivaernet_count = pd.read_sql("SELECT COUNT(*) as count FROM simulations WHERE source = 'drivaernet_real'", engine)['count'][0]
    
    print(f"\n✅ Import complete!")
    print(f"   Imported: {len(final_df)} records")
    print(f"   Total in database: {total}")
    print(f"   From DrivAerNet++: {drivaernet_count}")
    print(f"\n🎯 Key improvement: Using REAL drag coefficients from CFD simulations!")
    
    return final_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Import DrivAerNet++ CSV data')
    parser.add_argument('--csv-path', type=str, default=None, help='Path to CSV file')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of records')
    parser.add_argument('--replace', action='store_true', help='Replace existing data')
    
    args = parser.parse_args()
    
    try:
        import_drivaernet_csv(
            csv_path=args.csv_path,
            limit=args.limit,
            replace=args.replace
        )
        print("\n✅ Success! Now run: python src/train.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
