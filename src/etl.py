import os
import pyvista as pv
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, Float, String, Integer

DATA_DIR = "data/raw"
DB_PATH = "sqlite:///data/metadata.db"

def extract_metadata(filepath):
    """
    Extracts design parameters (inputs) and performance metrics (outputs) 
    from a VTK simulation file.
    """
    try:
        mesh = pv.read(filepath)
        
        # In our generator, we stored scalars in field_data
        # In real life, you might infer dimensions from bounds: mesh.bounds
        
        length = float(mesh.field_data["length"][0])
        width = float(mesh.field_data["width"][0])
        height = float(mesh.field_data["height"][0])
        load = float(mesh.field_data["load"][0])
        
        # For output, we calculate max stress from the point data array
        stress_data = mesh.point_data["von_mises_stress"]
        max_stress = float(stress_data.max())
        
        max_deflection = float(mesh.field_data["max_deflection"][0])
        
        return {
            "filename": os.path.basename(filepath),
            "length": length,
            "width": width,
            "height": height,
            "load": load,
            "max_stress": max_stress,
            "max_deflection": max_deflection
        }
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def run_etl():
    # 1. Setup Database
    engine = create_engine(DB_PATH)
    metadata = MetaData()
    
    simulations = Table('simulations', metadata,
        Column('id', Integer, primary_key=True),
        Column('filename', String),
        Column('length', Float),
        Column('width', Float),
        Column('height', Float),
        Column('load', Float),
        Column('max_stress', Float),
        Column('max_deflection', Float)
    )
    
    metadata.create_all(engine)
    
    # 2. Extract Data
    data = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".vtk"):
            filepath = os.path.join(DATA_DIR, filename)
            record = extract_metadata(filepath)
            if record:
                data.append(record)
    
    # 3. Load to DB
    df = pd.DataFrame(data)
    print(f"Extracted {len(df)} records.")
    
    df.to_sql('simulations', engine, if_exists='replace', index=False)
    print("Data loaded to SQLite.")

if __name__ == "__main__":
    run_etl()
