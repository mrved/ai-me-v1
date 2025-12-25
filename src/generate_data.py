import numpy as np
import pyvista as pv
import os
import random

OUTPUT_DIR = "data/raw"

def calculate_car_physics(length, width, height, load):
    """
    Simulates car structural analysis.
    For car design, we consider:
    - Aerodynamic load (drag force)
    - Structural stress from chassis loading
    - Deflection under load
    """
    # Material properties for automotive steel
    E = 200e9  # Young's Modulus for Steel (Pa)
    rho = 7850  # Density (kg/m^3)
    
    # Car body approximation as a box beam
    # Cross-sectional area
    A = width * height * 0.1  # Effective area (accounting for hollow structure)
    I = (width * height**3) / 12  # Moment of Inertia
    
    # Aerodynamic drag force approximation
    # Drag = 0.5 * rho_air * Cd * A * v^2
    # For simplification, we use the load parameter as drag force
    drag_force = load
    
    # Stress from aerodynamic loading (distributed load)
    # Max stress occurs at points of maximum moment
    max_stress = (drag_force * length * (height/2)) / I
    
    # Deflection from distributed aerodynamic load
    # Simplified: deflection ~ (q * L^4) / (E * I)
    q = drag_force / length  # Distributed load
    max_deflection = (5 * q * length**4) / (384 * E * I)
    
    return {
        "max_stress": max_stress,
        "max_deflection": max_deflection
    }

def create_car_mesh(length, width, height):
    """
    Creates a car-like geometry with body shape.
    Combines multiple primitives to form a car-like structure.
    """
    # Main body (chassis)
    body = pv.Cube(center=(length/2, 0, height*0.3), 
                   x_length=length*0.9, 
                   y_length=width, 
                   z_length=height*0.4)
    
    # Cabin section (slightly narrower and higher)
    cabin = pv.Cube(center=(length*0.6, 0, height*0.7), 
                    x_length=length*0.5, 
                    y_length=width*0.85, 
                    z_length=height*0.5)
    
    # Hood section (tapered)
    hood = pv.Cube(center=(length*0.2, 0, height*0.25), 
                   x_length=length*0.3, 
                   y_length=width*0.9, 
                   z_length=height*0.3)
    
    # Combine meshes
    mesh = body.boolean_union(cabin)
    mesh = mesh.boolean_union(hood)
    
    # If boolean operations fail, fall back to simple body
    if mesh.n_points == 0:
        mesh = body
    
    # Subdivide for better mesh quality
    mesh = mesh.triangulate()
    if mesh.n_cells > 0:
        mesh = mesh.subdivide(1)
    
    return mesh

def generate_sample(index):
    # Car design parameters (realistic ranges)
    length = random.uniform(3.5, 5.5)  # Car length in meters (compact to full-size)
    width = random.uniform(1.6, 2.0)   # Car width in meters
    height = random.uniform(1.4, 1.8)  # Car height in meters
    load = random.uniform(5000, 25000)  # Aerodynamic drag force in Newtons (at highway speeds)
    
    # Create car-like mesh
    try:
        mesh = create_car_mesh(length, width, height)
    except:
        # Fallback to simple box if car shape fails
        mesh = pv.Cube(center=(length/2, 0, height/2), 
                       x_length=length, 
                       y_length=width, 
                       z_length=height)
        mesh = mesh.triangulate().subdivide(1)
    
    # Calculate physics
    results = calculate_car_physics(length, width, height, load)
    
    # Add stress field to mesh
    points = mesh.points
    x_coords = points[:, 0]
    z_coords = points[:, 2]
    
    # Stress field: higher at front (aerodynamic pressure) and bottom (structural load)
    # Simulate stress concentration at front and base
    x_normalized = (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min() + 1e-10)
    z_normalized = (z_coords - z_coords.min()) / (z_coords.max() - z_coords.min() + 1e-10)
    
    stress_base = results["max_stress"] * 0.5
    stress_front = results["max_stress"] * (1.0 - x_normalized) * 0.3  # Higher at front
    stress_bottom = results["max_stress"] * (1.0 - z_normalized) * 0.2  # Higher at bottom
    
    stress_field = stress_base + stress_front * 0.3 + stress_bottom * 0.2
    stress_field = np.clip(stress_field, 0, results["max_stress"] * 1.2)
    
    mesh.point_data["von_mises_stress"] = stress_field
    
    # Add metadata as field data
    mesh.field_data["length"] = np.array([length])
    mesh.field_data["width"] = np.array([width])
    mesh.field_data["height"] = np.array([height])
    mesh.field_data["load"] = np.array([load])
    mesh.field_data["max_deflection"] = np.array([results["max_deflection"]])
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = os.path.join(OUTPUT_DIR, f"car_design_{index:03d}.vtk")
    mesh.save(filename)
    print(f"Generated {filename} (L={length:.2f}m, W={width:.2f}m, H={height:.2f}m, Load={load:.0f}N)")

if __name__ == "__main__":
    num_samples = 50  # Generate 50 car designs
    print(f"Generating {num_samples} car design samples...")
    for i in range(num_samples):
        generate_sample(i)
    print(f"\nCompleted! Generated {num_samples} car design files in {OUTPUT_DIR}")
