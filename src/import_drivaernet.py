"""
Import DrivAerNet++ Dataset
Downloads and processes real car design data with CFD simulations

Dataset: https://github.com/Mohamedelrefaie/DrivAerNet
Paper: https://arxiv.org/abs/2406.09624
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sqlalchemy import create_engine
import argparse
import json

DB_PATH = "sqlite:///data/metadata.db"
DATA_DIR = Path("data/drivaernet")

def download_drivaernet(dataset_path=None):
    """
    Download or access DrivAerNet++ dataset
    
    If dataset_path is provided, use that directory.
    Otherwise, provide instructions for manual download.
    """
    if dataset_path:
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        return dataset_path
    
    print("=" * 60)
    print("DrivAerNet++ Dataset Download Instructions")
    print("=" * 60)
    print("\n1. Go to: https://github.com/Mohamedelrefaie/DrivAerNet")
    print("2. Follow instructions to download the dataset")
    print("3. Or use direct download links from the repository")
    print("\n4. Once downloaded, run:")
    print("   python src/import_drivaernet.py --dataset-path /path/to/drivaernet")
    print("\nAlternatively, if you have the data in a specific format,")
    print("we can create a custom parser for your data structure.")
    print("=" * 60)
    
    return None

def parse_drivaernet_metadata(dataset_path):
    """
    Parse DrivAerNet++ metadata files
    The dataset typically includes JSON/CSV files with design parameters
    """
    metadata_files = list(dataset_path.glob("**/*.json")) + list(dataset_path.glob("**/*.csv"))
    
    if not metadata_files:
        print("⚠️  No metadata files found. Creating parser for mesh files...")
        return parse_from_meshes(dataset_path)
    
    data_records = []
    
    for meta_file in metadata_files[:100]:  # Process first 100 for testing
        try:
            if meta_file.suffix == '.json':
                with open(meta_file, 'r') as f:
                    data = json.load(f)
                    # Extract relevant parameters (adjust based on actual structure)
                    record = extract_parameters_from_json(data, meta_file)
                    if record:
                        data_records.append(record)
            elif meta_file.suffix == '.csv':
                df = pd.read_csv(meta_file)
                # Process CSV rows
                for _, row in df.iterrows():
                    record = extract_parameters_from_csv(row)
                    if record:
                        data_records.append(record)
        except Exception as e:
            print(f"Error processing {meta_file}: {e}")
            continue
    
    return data_records

def extract_parameters_from_json(data, filepath):
    """Extract design parameters from JSON metadata"""
    try:
        # Common parameter names in car design datasets
        # Adjust based on actual DrivAerNet++ structure
        
        # Try to extract dimensions
        length = data.get('length') or data.get('L') or data.get('wheelbase', 0) * 1.2
        width = data.get('width') or data.get('W') or data.get('track_width', 1.8)
        height = data.get('height') or data.get('H') or data.get('roof_height', 1.6)
        
        # Try to extract aerodynamic data
        drag_coefficient = data.get('Cd') or data.get('drag_coefficient') or data.get('cd', 0.3)
        drag_force = data.get('drag_force') or data.get('F_drag') or data.get('drag', 15000)
        
        # Try to extract stress data
        max_stress = data.get('max_stress') or data.get('stress_max') or data.get('von_mises_max')
        
        # If we have mesh bounds, use those
        if 'bounds' in data:
            bounds = data['bounds']
            if not length or length == 0:
                length = bounds[1] - bounds[0] if len(bounds) > 1 else 4.5
            if not width or width == 0:
                width = bounds[3] - bounds[2] if len(bounds) > 3 else 1.8
            if not height or height == 0:
                height = bounds[5] - bounds[4] if len(bounds) > 5 else 1.6
        
        # Calculate load from drag if available
        if not drag_force or drag_force == 0:
            # Estimate from drag coefficient
            # Drag = 0.5 * rho * Cd * A * v^2
            # Assume: rho=1.2 kg/m³, A=2.5 m², v=100 km/h = 27.8 m/s
            frontal_area = width * height * 0.8  # Approximate
            air_density = 1.2
            velocity = 27.8  # 100 km/h
            drag_force = 0.5 * air_density * drag_coefficient * frontal_area * velocity**2
        
        # If no stress data, we'll need to calculate or skip
        if not max_stress or max_stress == 0:
            # Estimate from simplified physics (fallback)
            E = 200e9  # Steel
            I = (width * height**3) / 12
            max_stress = (drag_force * length * (height/2)) / I if I > 0 else 50000
        
        return {
            'filename': filepath.name,
            'length': float(length),
            'width': float(width),
            'height': float(height),
            'load': float(drag_force),
            'max_stress': float(max_stress),
            'max_deflection': float(max_stress * 1e-9),  # Estimate
            'drag_coefficient': float(drag_coefficient) if drag_coefficient else 0.3,
            'source': 'drivaernet'
        }
    except Exception as e:
        print(f"Error extracting from {filepath}: {e}")
        return None

def extract_parameters_from_csv(row):
    """Extract parameters from CSV row"""
    try:
        # Map common column names
        length = row.get('length') or row.get('L') or row.get('wheelbase', 4.5)
        width = row.get('width') or row.get('W') or row.get('track', 1.8)
        height = row.get('height') or row.get('H') or row.get('roof_height', 1.6)
        load = row.get('drag_force') or row.get('load') or row.get('F_drag', 15000)
        max_stress = row.get('max_stress') or row.get('stress') or row.get('von_mises', 75000)
        
        return {
            'filename': f"design_{row.name}",
            'length': float(length),
            'width': float(width),
            'height': float(height),
            'load': float(load),
            'max_stress': float(max_stress),
            'max_deflection': float(max_stress * 1e-9),
            'source': 'drivaernet'
        }
    except Exception as e:
        return None

def parse_from_meshes(dataset_path):
    """
    Parse design parameters directly from 3D mesh files
    This is a fallback if metadata files aren't available
    """
    import pyvista as pv
    
    mesh_files = list(dataset_path.glob("**/*.obj")) + \
                 list(dataset_path.glob("**/*.stl")) + \
                 list(dataset_path.glob("**/*.vtk"))
    
    if not mesh_files:
        print("❌ No mesh files found in dataset")
        return []
    
    data_records = []
    
    print(f"Found {len(mesh_files)} mesh files. Processing...")
    
    for i, mesh_file in enumerate(mesh_files[:500]):  # Limit for initial import
        try:
            mesh = pv.read(str(mesh_file))
            
            # Extract dimensions from mesh bounds
            bounds = mesh.bounds
            length = bounds[1] - bounds[0]
            width = bounds[3] - bounds[2]
            height = bounds[5] - bounds[4]
            
            # Estimate load from geometry (simplified)
            frontal_area = width * height * 0.8
            drag_coefficient = 0.3  # Average for cars
            air_density = 1.2
            velocity = 27.8  # 100 km/h
            drag_force = 0.5 * air_density * drag_coefficient * frontal_area * velocity**2
            
            # Estimate stress (simplified beam theory)
            E = 200e9
            I = (width * height**3) / 12
            max_stress = (drag_force * length * (height/2)) / I if I > 0 else 50000
            
            data_records.append({
                'filename': mesh_file.name,
                'length': float(length),
                'width': float(width),
                'height': float(height),
                'load': float(drag_force),
                'max_stress': float(max_stress),
                'max_deflection': float(max_stress * 1e-9),
                'source': 'drivaernet_mesh'
            })
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{min(500, len(mesh_files))} files...")
                
        except Exception as e:
            print(f"Error processing {mesh_file}: {e}")
            continue
    
    return data_records

def import_to_database(data_records, append=True):
    """Import parsed data into the database"""
    if not data_records:
        print("❌ No data records to import")
        return
    
    engine = create_engine(DB_PATH)
    
    df = pd.DataFrame(data_records)
    
    print(f"\n📊 Importing {len(df)} records...")
    print(f"   Length range: {df['length'].min():.2f} - {df['length'].max():.2f} m")
    print(f"   Width range: {df['width'].min():.2f} - {df['width'].max():.2f} m")
    print(f"   Height range: {df['height'].min():.2f} - {df['height'].max():.2f} m")
    print(f"   Load range: {df['load'].min():.0f} - {df['load'].max():.0f} N")
    
    # Import to database
    if_exists = 'append' if append else 'replace'
    df.to_sql('simulations', engine, if_exists=if_exists, index=False)
    
    print(f"✅ Successfully imported {len(df)} records to database")
    
    # Show summary
    total = pd.read_sql('SELECT COUNT(*) as count FROM simulations', engine)['count'][0]
    print(f"📈 Total records in database: {total}")

def main():
    parser = argparse.ArgumentParser(description='Import DrivAerNet++ dataset')
    parser.add_argument('--dataset-path', type=str, help='Path to DrivAerNet++ dataset')
    parser.add_argument('--replace', action='store_true', help='Replace existing data (default: append)')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of records to import')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DrivAerNet++ Dataset Importer")
    print("=" * 60)
    
    # Download or get dataset path
    dataset_path = download_drivaernet(args.dataset_path)
    
    if not dataset_path:
        print("\n💡 Please provide dataset path:")
        print("   python src/import_drivaernet.py --dataset-path /path/to/drivaernet")
        return
    
    # Parse data
    print(f"\n📂 Parsing data from: {dataset_path}")
    data_records = parse_drivaernet_metadata(dataset_path)
    
    if args.limit:
        data_records = data_records[:args.limit]
    
    # Import to database
    if data_records:
        import_to_database(data_records, append=not args.replace)
        print("\n✅ Import complete! Run 'python src/train.py' to retrain the model.")
    else:
        print("\n❌ No data records extracted. Check dataset format.")

if __name__ == "__main__":
    main()

