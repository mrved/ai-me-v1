import os
import pyvista as pv
import numpy as np
import argparse

OUTPUT_DIR = "data/raw"

def process_obj_file(filepath):
    """
    Reads an OBJ file (ABC Dataset format), calculates properties,
    assigns synthetic physics, and saves as VTK.
    """
    try:
        mesh = pv.read(filepath)
    except Exception as e:
        print(f"Failed to read {filepath}: {e}")
        return

    # 1. Calculate Geometry
    # OBJ from ABC might be surface meshes.
    if mesh.volume > 0:
        volume = mesh.volume
    else:
        # If surface mesh, estimate volume or use surface area
        volume = mesh.area * 0.01 # Mock thickness

    # 2. Assign Synthetic Physics
    # In a real scenario, you'd run a solver here.
    # We will simulate a "Load" and "Max Stress"
    # Stress ~ Load / Area
    
    load = np.random.uniform(1000, 10000)
    if mesh.area > 0:
        max_stress = load / mesh.area
    else:
        max_stress = 0.0
        
    # Generate a dummy field for visualization
    # Stress concentration at center
    centers = mesh.points
    dist = np.linalg.norm(centers - mesh.center, axis=1)
    stress_field = max_stress * np.exp(-dist)
    
    mesh.point_data["von_mises_stress"] = stress_field
    
    # Store Metadata
    mesh.field_data["length"] = np.array([mesh.bounds[1] - mesh.bounds[0]])
    mesh.field_data["width"] = np.array([mesh.bounds[3] - mesh.bounds[2]])
    mesh.field_data["height"] = np.array([mesh.bounds[5] - mesh.bounds[4]])
    mesh.field_data["load"] = np.array([load])
    mesh.field_data["max_deflection"] = np.array([max_stress * 1e-9]) # Dummy
    
    # Save as VTK for the pipeline
    base_name = os.path.basename(filepath).replace(".obj", "")
    output_path = os.path.join(OUTPUT_DIR, f"processed_{base_name}.vtk")
    mesh.save(output_path)
    print(f"Processed {filepath} -> {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="OBJ files to process")
    args = parser.parse_args()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for f in args.files:
        process_obj_file(f)
