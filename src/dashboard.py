import streamlit as st
import pandas as pd
import joblib
import os
from sqlalchemy import create_engine
import pyvista as pv
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

# Page Configuration
st.set_page_config(page_title="AI Engineering Dashboard", layout="wide")

# Get the project root directory (parent of src/)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DB_PATH = f"sqlite:///{PROJECT_ROOT}/data/metadata.db"
MODEL_PATH = PROJECT_ROOT / "model.pkl"
DB_FILE = PROJECT_ROOT / "data" / "metadata.db"

def get_data():
    try:
        if not DB_FILE.exists():
            return pd.DataFrame()
        engine = create_engine(DB_PATH)
        return pd.read_sql('simulations', engine)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

@st.cache_resource
def load_model():
    """Load model, auto-generating if needed"""
    try:
        if MODEL_PATH.exists():
            return joblib.load(MODEL_PATH)
        
        # If model doesn't exist, try to generate it
        with st.spinner("🔄 Model not found. Setting up... This may take 30-60 seconds on first run."):
            # Check if data exists
            if not DB_FILE.exists():
                # Generate data first
                import subprocess
                import sys
                with st.status("Generating car design data...", expanded=False) as status:
                    result = subprocess.run([sys.executable, str(PROJECT_ROOT / "src" / "generate_data.py")], 
                                          capture_output=True, text=True, cwd=str(PROJECT_ROOT))
                    if result.returncode != 0:
                        st.error(f"Failed to generate data: {result.stderr}")
                        return None
                    status.update(label="✅ Data generated", state="complete")
                
                # Run ETL
                with st.status("Running ETL pipeline...", expanded=False) as status:
                    result = subprocess.run([sys.executable, str(PROJECT_ROOT / "src" / "etl.py")],
                                          capture_output=True, text=True, cwd=str(PROJECT_ROOT))
                    if result.returncode != 0:
                        st.error(f"Failed to run ETL: {result.stderr}")
                        return None
                    status.update(label="✅ ETL completed", state="complete")
            
            # Train model
            with st.status("Training ML model...", expanded=False) as status:
                import subprocess
                import sys
                result = subprocess.run([sys.executable, str(PROJECT_ROOT / "src" / "train.py")],
                                      capture_output=True, text=True, cwd=str(PROJECT_ROOT))
                if result.returncode != 0:
                    st.error(f"Failed to train model: {result.stderr}")
                    return None
                status.update(label="✅ Model trained", state="complete")
        
        # Load the newly created model
        if MODEL_PATH.exists():
            st.success("✅ Setup complete! Model is ready.")
            return joblib.load(MODEL_PATH)
        
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

def _show_guidance(current_value, optimal_value, min_val, max_val, param_name):
    """Show subtle guidance indicator for parameter adjustment"""
    diff = current_value - optimal_value
    diff_pct = abs(diff) / (max_val - min_val) * 100 if (max_val - min_val) > 0 else 0
    
    # Only show if significantly off (more than 2% of range)
    if diff_pct < 2:
        st.caption(f"✓ Optimal: {optimal_value:.2f}")
    else:
        # Calculate position on slider (0 to 100)
        current_pos = ((current_value - min_val) / (max_val - min_val) * 100) if (max_val - min_val) > 0 else 50
        optimal_pos = ((optimal_value - min_val) / (max_val - min_val) * 100) if (max_val - min_val) > 0 else 50
        
        # Determine direction
        if diff > 0:
            direction = "↓ Decrease"
            arrow = "↓"
        else:
            direction = "↑ Increase"
            arrow = "↑"
        
        # Create a simple progress bar visualization
        st.caption(f"{arrow} {direction} to {optimal_value:.2f}")
        
        # Show a subtle progress bar
        col_prog1, col_prog2, col_prog3 = st.columns([1, 8, 1])
        with col_prog1:
            st.write("")  # Spacer
        with col_prog2:
            # Create a visual slider representation
            bar_width = 100
            current_bar_pos = int((current_value - min_val) / (max_val - min_val) * bar_width) if (max_val - min_val) > 0 else 50
            optimal_bar_pos = int((optimal_value - min_val) / (max_val - min_val) * bar_width) if (max_val - min_val) > 0 else 50
            
            # Create HTML-like visualization using markdown
            st.markdown(
                f'<div style="position: relative; height: 4px; background: #e0e0e0; border-radius: 2px; margin: 2px 0;">'
                f'<div style="position: absolute; left: {optimal_bar_pos}%; width: 2px; height: 100%; background: #4CAF50; border-radius: 1px;"></div>'
                f'<div style="position: absolute; left: {current_bar_pos}%; width: 2px; height: 100%; background: #2196F3; border-radius: 1px;"></div>'
                f'</div>',
                unsafe_allow_html=True
            )
        with col_prog3:
            st.write("")  # Spacer

def create_car_mesh_3d(length, width, height, stress_value=None):
    """Create a realistic 3D car mesh for visualization"""
    try:
        # Use MultiBlock to combine parts reliably (avoids boolean union failures)
        parts = []
        
        # Main chassis (lower body) - base of car
        chassis = pv.Cube(
            center=(length/2, 0, height*0.15), 
            x_length=length*0.95, 
            y_length=width, 
            z_length=height*0.25
        )
        parts.append(chassis)
        
        # Upper body (cabin area) - passenger compartment
        upper_body = pv.Cube(
            center=(length*0.55, 0, height*0.65), 
            x_length=length*0.55, 
            y_length=width*0.92, 
            z_length=height*0.5
        )
        parts.append(upper_body)
        
        # Hood (front section)
        hood = pv.Cube(
            center=(length*0.25, 0, height*0.2), 
            x_length=length*0.35, 
            y_length=width*0.95, 
            z_length=height*0.25
        )
        parts.append(hood)
        
        # Trunk/rear section
        trunk = pv.Cube(
            center=(length*0.85, 0, height*0.2), 
            x_length=length*0.25, 
            y_length=width*0.95, 
            z_length=height*0.25
        )
        parts.append(trunk)
        
        # Windshield (front glass area)
        windshield = pv.Cube(
            center=(length*0.4, 0, height*0.75), 
            x_length=length*0.15, 
            y_length=width*0.88, 
            z_length=height*0.15
        )
        parts.append(windshield)
        
        # Add wheels (cylinders)
        wheel_radius = width * 0.15
        wheel_width = width * 0.12
        wheel_z = wheel_radius
        
        # Front wheels
        front_wheel_left = pv.Cylinder(
            center=(length*0.25, -width/2 - wheel_width/2, wheel_z),
            direction=(0, 1, 0),
            radius=wheel_radius,
            height=wheel_width
        )
        parts.append(front_wheel_left)
        
        front_wheel_right = pv.Cylinder(
            center=(length*0.25, width/2 + wheel_width/2, wheel_z),
            direction=(0, 1, 0),
            radius=wheel_radius,
            height=wheel_width
        )
        parts.append(front_wheel_right)
        
        # Rear wheels
        rear_wheel_left = pv.Cylinder(
            center=(length*0.75, -width/2 - wheel_width/2, wheel_z),
            direction=(0, 1, 0),
            radius=wheel_radius,
            height=wheel_width
        )
        parts.append(rear_wheel_left)
        
        rear_wheel_right = pv.Cylinder(
            center=(length*0.75, width/2 + wheel_width/2, wheel_z),
            direction=(0, 1, 0),
            radius=wheel_radius,
            height=wheel_width
        )
        parts.append(rear_wheel_right)
        
        # Manually combine all parts by concatenating vertices and faces
        try:
            all_points = []
            all_faces = []
            point_offset = 0
            
            for part in parts:
                try:
                    # Ensure part is triangulated
                    part_tri = part.triangulate() if hasattr(part, 'triangulate') else part
                    
                    # Get points and faces
                    part_points = part_tri.points
                    part_faces = part_tri.faces
                    
                    if len(part_points) == 0:
                        continue
                    
                    # Add points
                    all_points.append(part_points)
                    
                    # Adjust face indices and add faces
                    if len(part_faces) > 0:
                        # PyVista face format: [n, i1, i2, ..., in, n, i1, ...]
                        i = 0
                        adjusted_faces = []
                        while i < len(part_faces):
                            n_verts = int(part_faces[i])
                            if n_verts >= 3 and i + 1 + n_verts <= len(part_faces):
                                # Add face count
                                adjusted_faces.append(n_verts)
                                # Add vertex indices with offset
                                for j in range(1, n_verts + 1):
                                    adjusted_faces.append(int(part_faces[i + j]) + point_offset)
                                i += n_verts + 1
                            else:
                                i += 1
                        
                        if len(adjusted_faces) > 0:
                            all_faces.extend(adjusted_faces)
                    
                    # Update point offset after processing this part
                    point_offset += len(part_points)
                        
                except Exception as part_err:
                    # Skip this part if it fails
                    continue
            
            # Create combined mesh
            if len(all_points) > 0:
                combined_points = np.vstack(all_points)
                if len(all_faces) > 0:
                    combined_faces = np.array(all_faces, dtype=np.int32)
                    mesh = pv.PolyData(combined_points, combined_faces)
                else:
                    # If no faces, create from points
                    mesh = pv.PolyData(combined_points)
            else:
                mesh = chassis
                
        except Exception as combine_err:
            # Fallback: just use chassis
            mesh = chassis
        
        # Ensure mesh is triangulated
        try:
            if hasattr(mesh, 'triangulate'):
                mesh = mesh.triangulate()
        except:
            pass
        
        # Add stress field if provided
        if stress_value is not None:
            try:
                points = mesh.points
                if len(points) > 0:
                    x_coords = points[:, 0]
                    z_coords = points[:, 2]
                    x_norm = (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min() + 1e-10)
                    z_norm = (z_coords - z_coords.min()) / (z_coords.max() - z_coords.min() + 1e-10)
                    
                    # More realistic stress distribution
                    # Higher at front (aerodynamic), at bottom (structural), and at wheel wells
                    stress_base = stress_value * 0.4
                    stress_front = stress_value * 0.3 * (1.0 - x_norm)  # Higher at front
                    stress_bottom = stress_value * 0.2 * (1.0 - z_norm / height)  # Higher at bottom
                    stress_wheels = stress_value * 0.1 * np.exp(-((x_coords - length*0.25)**2 + (x_coords - length*0.75)**2) / (length*0.1)**2)
                    
                    stress_field = stress_base + stress_front + stress_bottom + stress_wheels
                    stress_field = np.clip(stress_field, 0, stress_value * 1.3)
                    mesh.point_data["stress"] = stress_field
            except:
                pass
        
        return mesh
        
    except Exception as e:
        # Ultimate fallback: simple car shape
        try:
            body = pv.Cube(center=(length/2, 0, height*0.3), x_length=length*0.9, y_length=width, z_length=height*0.4)
            cabin = pv.Cube(center=(length*0.6, 0, height*0.7), x_length=length*0.5, y_length=width*0.85, z_length=height*0.5)
            try:
                mesh = body.boolean_union(cabin)
            except:
                mesh = body
            mesh = mesh.triangulate()
            
            if stress_value is not None:
                points = mesh.points
                if len(points) > 0:
                    x_coords = points[:, 0]
                    z_coords = points[:, 2]
                    x_norm = (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min() + 1e-10)
                    z_norm = (z_coords - z_coords.min()) / (z_coords.max() - z_coords.min() + 1e-10)
                    stress_field = stress_value * (0.5 + 0.3 * (1 - x_norm) + 0.2 * (1 - z_norm))
                    mesh.point_data["stress"] = stress_field
            
            return mesh
        except:
            # Last resort: simple box
            return pv.Cube(center=(length/2, 0, height/2), x_length=length, y_length=width, z_length=height)

def plotly_mesh_from_pyvista(mesh, stress_field=None):
    """Convert PyVista mesh to Plotly 3D visualization"""
    try:
        vertices = mesh.points
        if len(vertices) == 0:
            raise ValueError("Empty mesh")
        
        # Handle faces - check if mesh has faces
        # Use n_cells instead of n_faces (PyVista API change)
        n_faces = mesh.n_cells if hasattr(mesh, 'n_cells') else (mesh.n_faces if hasattr(mesh, 'n_faces') else 0)
        if n_faces == 0:
            # Fallback: create simple box visualization
            return create_simple_box_viz(mesh.bounds)
        
        # PyVista faces format: [n, i1, i2, ..., in, n, i1, i2, ...]
        # where n is the number of vertices in the face
        faces = mesh.faces
        if len(faces) == 0:
            return create_simple_box_viz(mesh.bounds)
        
        # Parse PyVista face connectivity array
        triangles = []
        i = 0
        try:
            while i < len(faces):
                if i >= len(faces):
                    break
                n_verts = int(faces[i])
                if n_verts < 3 or n_verts > 10:  # Sanity check
                    i += 1
                    continue
                
                # Check bounds
                if i + 1 + n_verts > len(faces):
                    break
                
                # Get vertex indices for this face
                face_verts = faces[i+1:i+1+n_verts].astype(int)
                
                # Validate indices
                if np.any(face_verts < 0) or np.any(face_verts >= len(vertices)):
                    i += n_verts + 1
                    continue
                
                # Triangulate if quad or higher (split into triangles)
                if n_verts == 3:
                    # Already a triangle
                    triangles.append(face_verts)
                elif n_verts == 4:
                    # Quad - split into 2 triangles
                    triangles.append([face_verts[0], face_verts[1], face_verts[2]])
                    triangles.append([face_verts[0], face_verts[2], face_verts[3]])
                else:
                    # Polygon - fan triangulation
                    for j in range(1, n_verts - 1):
                        triangles.append([face_verts[0], face_verts[j], face_verts[j+1]])
                
                i += n_verts + 1
        except Exception as parse_error:
            # If parsing fails, use fallback
            pass
        
        if len(triangles) == 0:
            return create_simple_box_viz(mesh.bounds)
        
        # Convert to numpy array
        try:
            triangles = np.array(triangles)
            if triangles.shape[1] != 3:
                return create_simple_box_viz(mesh.bounds)
        except:
            return create_simple_box_viz(mesh.bounds)
        
        # Create mesh3d trace with stress field
        colors = None
        colorscale = 'Viridis'
        if "stress" in mesh.point_data:
            colors = mesh.point_data["stress"]
            # Normalize colors for better visualization
            if len(colors) > 0:
                c_min, c_max = colors.min(), colors.max()
                if c_max > c_min:
                    colors = (colors - c_min) / (c_max - c_min)
                colorscale = [[0, 'blue'], [0.5, 'yellow'], [1, 'red']]
            else:
                colors = None
        
        fig = go.Figure(data=[go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=triangles[:, 0],
            j=triangles[:, 1],
            k=triangles[:, 2],
            intensity=colors,
            colorscale=colorscale,
            showscale=colors is not None,
            colorbar=dict(title="Stress (Pa)") if colors is not None else None,
            opacity=0.8,
            lighting=dict(ambient=0.7, diffuse=0.8, specular=0.1),
        )])
        
        fig.update_layout(
            scene=dict(
                xaxis_title="Length (m)",
                yaxis_title="Width (m)",
                zaxis_title="Height (m)",
                aspectmode='data',
                camera=dict(
                    eye=dict(x=2.0, y=2.0, z=1.2),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                ),
                bgcolor='rgb(245, 245, 250)',
                xaxis=dict(backgroundcolor='white', gridcolor='lightgray', showbackground=True),
                yaxis=dict(backgroundcolor='white', gridcolor='lightgray', showbackground=True),
                zaxis=dict(backgroundcolor='white', gridcolor='lightgray', showbackground=True)
            ),
            height=700,
            margin=dict(l=0, r=0, t=20, b=0),
            paper_bgcolor='white'
        )
        
        return fig
    except Exception as e:
        # Fallback to simple visualization
        try:
            bounds = mesh.bounds if hasattr(mesh, 'bounds') and mesh.bounds else [0, 4.5, -1, 1, 0, 1.6]
            return create_simple_box_viz(bounds)
        except:
            # Ultimate fallback
            return create_simple_box_viz([0, 4.5, -1, 1, 0, 1.6])

def create_simple_box_viz(bounds):
    """Create a simple box visualization as fallback - manually create vertices and faces"""
    try:
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        
        # Manually create 8 vertices of a box
        vertices = np.array([
            [x_min, y_min, z_min],  # 0
            [x_max, y_min, z_min],  # 1
            [x_max, y_max, z_min],  # 2
            [x_min, y_max, z_min],  # 3
            [x_min, y_min, z_max],  # 4
            [x_max, y_min, z_max],  # 5
            [x_max, y_max, z_max],  # 6
            [x_min, y_max, z_max],  # 7
        ])
        
        # Define 12 triangular faces (2 per box face, 6 faces total)
        # Each face is split into 2 triangles
        faces = np.array([
            # Bottom face (z_min)
            [0, 1, 2], [0, 2, 3],
            # Top face (z_max)
            [4, 7, 6], [4, 6, 5],
            # Front face (y_min)
            [0, 4, 5], [0, 5, 1],
            # Back face (y_max)
            [3, 2, 6], [3, 6, 7],
            # Left face (x_min)
            [0, 3, 7], [0, 7, 4],
            # Right face (x_max)
            [1, 5, 6], [1, 6, 2],
        ])
        
        fig = go.Figure(data=[go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color='lightblue',
            opacity=0.8,
            lighting=dict(ambient=0.7, diffuse=0.8, specular=0.1),
        )])
        
        fig.update_layout(
            scene=dict(
                xaxis_title="Length (m)",
                yaxis_title="Width (m)",
                zaxis_title="Height (m)",
                aspectmode='data',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=600,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        return fig
    except Exception as e:
        # Ultimate fallback - just show a message
        fig = go.Figure()
        fig.add_annotation(
            text="3D visualization unavailable<br>Prediction results are still valid",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(height=600)
        return fig

def main():
    st.title("AI-Driven Design Optimization")
    
    tab1, tab2 = st.tabs(["Virtual Test Bench", "Data Lakehouse"])
    
    # --- TAB 1: VIRTUAL TEST BENCH ---
    with tab1:
        st.header("🚗 Virtual Test Bench - Car Design Analysis")
        
        # Explanation section
        with st.expander("📖 What is this?", expanded=True):
            st.markdown("""
            **Virtual Test Bench** uses AI to predict structural performance of car designs without expensive physical testing.
            
            **What we're testing:**
            - **Structural Stress Analysis**: Predicts maximum stress under aerodynamic loads
            - **Design Optimization**: Quickly evaluate different car geometries
            - **Material Safety**: Ensures designs stay within safe stress limits
            
            **How it works:**
            1. Enter car dimensions (length, width, height) and expected aerodynamic load
            2. AI model predicts maximum stress based on 2,000+ real car designs with CFD data
            3. 3D visualization shows stress distribution across the car body
            4. Use results to optimize design before manufacturing
            
            **Data Source:** Trained on DrivAerNet++ dataset with real drag coefficients from CFD simulations
            """)
        
        model = load_model()
        
        if model is None:
            st.error("Model not found. Train the model first.")
        else:
            # Layout: Parameters on left, Results on right
            col1, col2 = st.columns([1, 1.5])
            
            with col1:
                st.subheader("🎛️ Design Parameters")
                
                # Get data for suggestions
                df_check = get_data()
                has_advanced_params = not df_check.empty and 'drag_coefficient' in df_check.columns
                
                # Calculate optimal values from data (lowest stress designs)
                # Define min/max ranges for each parameter
                LENGTH_MIN, LENGTH_MAX = 3.5, 5.5
                WIDTH_MIN, WIDTH_MAX = 1.6, 2.0
                HEIGHT_MIN, HEIGHT_MAX = 1.4, 1.8
                CD_MIN, CD_MAX = 0.20, 0.35
                WHEELBASE_MIN, WHEELBASE_MAX = 2.1, 3.3  # Adjusted to match actual data range
                ROOF_ANGLE_MIN, ROOF_ANGLE_MAX = -30.0, 30.0
                
                if not df_check.empty:
                    # Find designs with lowest stress (top 10%)
                    optimal_designs = df_check.nsmallest(int(len(df_check) * 0.1), 'max_stress')
                    suggested_length = max(LENGTH_MIN, min(LENGTH_MAX, optimal_designs['length'].mean()))
                    suggested_width = max(WIDTH_MIN, min(WIDTH_MAX, optimal_designs['width'].mean()))
                    suggested_height = max(HEIGHT_MIN, min(HEIGHT_MAX, optimal_designs['height'].mean()))
                    if has_advanced_params:
                        suggested_cd = max(CD_MIN, min(CD_MAX, optimal_designs['drag_coefficient'].mean()))
                        suggested_wheelbase = max(WHEELBASE_MIN, min(WHEELBASE_MAX, 
                            optimal_designs['wheelbase'].mean() if 'wheelbase' in optimal_designs.columns else 2.8))
                        suggested_roof_angle = max(ROOF_ANGLE_MIN, min(ROOF_ANGLE_MAX,
                            optimal_designs['roof_angle'].mean() if 'roof_angle' in optimal_designs.columns else 0.0))
                else:
                    suggested_length, suggested_width, suggested_height = 4.5, 1.8, 1.6
                    suggested_cd, suggested_wheelbase, suggested_roof_angle = 0.26, 2.8, 0.0
                
                st.markdown("**Car Dimensions:**")
                
                # Length with suggestion
                length = st.number_input(
                    "Length (m)", 
                    LENGTH_MIN, LENGTH_MAX, float(suggested_length),
                    help="Overall length of the car from front to back",
                    key="length_input"
                )
                # Subtle guidance indicator
                _show_guidance(length, suggested_length, LENGTH_MIN, LENGTH_MAX, "Length")
                
                # Width with suggestion
                width = st.number_input(
                    "Width (m)", 
                    WIDTH_MIN, WIDTH_MAX, float(suggested_width),
                    help="Width of the car (track width)",
                    key="width_input"
                )
                _show_guidance(width, suggested_width, WIDTH_MIN, WIDTH_MAX, "Width")
                
                # Height with suggestion
                height = st.number_input(
                    "Height (m)", 
                    HEIGHT_MIN, HEIGHT_MAX, float(suggested_height),
                    help="Overall height of the car",
                    key="height_input"
                )
                _show_guidance(height, suggested_height, HEIGHT_MIN, HEIGHT_MAX, "Height")
                
                if has_advanced_params:
                    st.markdown("**Advanced Parameters:**")
                    
                    # Drag coefficient with suggestion
                    drag_coefficient = st.number_input(
                        "Drag Coefficient (Cd)", 
                        CD_MIN, CD_MAX, float(suggested_cd),
                        help="Aerodynamic drag coefficient (lower is better, typical: 0.25-0.30)",
                        key="cd_input"
                    )
                    # For drag coefficient, lower is better - show subtle guidance with slider
                    diff_cd = drag_coefficient - suggested_cd
                    diff_cd_pct = abs(diff_cd) / (CD_MAX - CD_MIN) * 100 if (CD_MAX - CD_MIN) > 0 else 0
                    if diff_cd_pct < 2:
                        st.caption(f"✓ Optimal: {suggested_cd:.3f}")
                    elif drag_coefficient > suggested_cd:
                        st.caption(f"↓ Decrease to {suggested_cd:.3f} (lower is better)")
                        # Show slider
                        current_pos = int((drag_coefficient - CD_MIN) / (CD_MAX - CD_MIN) * 100) if (CD_MAX - CD_MIN) > 0 else 50
                        optimal_pos = int((suggested_cd - CD_MIN) / (CD_MAX - CD_MIN) * 100) if (CD_MAX - CD_MIN) > 0 else 50
                        st.markdown(
                            f'<div style="position: relative; height: 4px; background: #e0e0e0; border-radius: 2px; margin: 2px 0;">'
                            f'<div style="position: absolute; left: {optimal_pos}%; width: 2px; height: 100%; background: #4CAF50; border-radius: 1px;"></div>'
                            f'<div style="position: absolute; left: {current_pos}%; width: 2px; height: 100%; background: #2196F3; border-radius: 1px;"></div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.caption(f"✓ Optimal: {suggested_cd:.3f}")
                    
                    # Wheelbase with suggestion
                    wheelbase = st.number_input(
                        "Wheelbase (m)", 
                        WHEELBASE_MIN, WHEELBASE_MAX, float(suggested_wheelbase),
                        help="Distance between front and rear axles",
                        key="wheelbase_input"
                    )
                    _show_guidance(wheelbase, suggested_wheelbase, WHEELBASE_MIN, WHEELBASE_MAX, "Wheelbase")
                    
                    # Roof angle with suggestion
                    roof_angle = st.number_input(
                        "Roof Angle (degrees)", 
                        ROOF_ANGLE_MIN, ROOF_ANGLE_MAX, float(suggested_roof_angle),
                        help="Greenhouse/roof angle",
                        key="roof_angle_input"
                    )
                    _show_guidance(roof_angle, suggested_roof_angle, ROOF_ANGLE_MIN, ROOF_ANGLE_MAX, "Roof Angle")
                
                st.markdown("**Loading Conditions:**")
                if has_advanced_params:
                    # Calculate load from drag coefficient if available
                    air_density = 1.2  # kg/m³
                    velocity = 27.8  # m/s (100 km/h)
                    frontal_area = width * height * 0.8
                    calculated_load = 0.5 * air_density * drag_coefficient * frontal_area * velocity**2
                    
                    load = st.number_input(
                        "Aerodynamic Load (N)", 
                        100.0, 500.0, float(calculated_load),
                        help=f"Drag force (calculated from Cd: {calculated_load:.0f}N, or enter custom)"
                    )
                else:
                    load = st.number_input(
                        "Aerodynamic Load (N)", 
                        5000.0, 25000.0, 15000.0,
                        help="Drag force at highway speeds (typically 10,000-20,000N)"
                    )
                
                st.markdown("---")
                predict_btn = st.button("🔬 Predict Performance", type="primary", use_container_width=True)
                
                # Show parameter summary
                if predict_btn:
                    st.info(f"""
                    **Design Summary:**
                    - Volume: {length*width*height:.2f} m³
                    - Aspect Ratio: {length/width:.2f}
                    - Load/Volume: {load/(length*width*height):.0f} N/m³
                    """)
            
            with col2:
                st.subheader("📊 Analysis Results")
                
                if predict_btn:
                    try:
                        # Build input data with all available features
                        df_check = get_data()
                        has_advanced = not df_check.empty and 'drag_coefficient' in df_check.columns
                        
                        if has_advanced:
                            # Use all 7 parameters for better accuracy
                            input_data = pd.DataFrame([[
                                length, width, height, load, 
                                drag_coefficient, wheelbase, roof_angle
                            ]], columns=['length', 'width', 'height', 'load', 
                                        'drag_coefficient', 'wheelbase', 'roof_angle'])
                        else:
                            # Fallback to basic 4 parameters
                            input_data = pd.DataFrame([[length, width, height, load]], 
                                                    columns=['length', 'width', 'height', 'load'])
                        
                        prediction = model.predict(input_data)[0]
                        
                        # Calculate additional metrics
                        volume = length * width * height
                        stress_per_volume = prediction / volume if volume > 0 else 0
                        
                        # Results display
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric(
                                "Predicted Max Stress", 
                                f"{prediction:,.0f} Pa",
                                help="Maximum von Mises stress in the structure"
                            )
                        with col_b:
                            # Safety factor (assuming yield strength of 250 MPa for steel)
                            yield_strength = 250e6  # 250 MPa
                            safety_factor = yield_strength / prediction if prediction > 0 else 0
                            st.metric(
                                "Safety Factor", 
                                f"{safety_factor:.2f}",
                                delta="Safe" if safety_factor > 1.5 else "Warning" if safety_factor > 1.0 else "Critical"
                            )
                        
                        # Stress interpretation
                        if prediction < 100e6:  # < 100 MPa
                            st.success("✅ Low stress - Design is safe and efficient")
                        elif prediction < 200e6:  # < 200 MPa
                            st.warning("⚠️ Moderate stress - Consider optimization")
                        else:
                            st.error("❌ High stress - Redesign recommended")
                        
                        st.markdown("---")
                        
                        # 3D Visualization
                        st.subheader("🎨 3D Stress Visualization")
                        st.markdown("Interactive 3D model showing predicted stress distribution:")
                        
                        try:
                            # Create enhanced car mesh with stress field
                            with st.spinner("Generating 3D car model..."):
                                mesh = create_car_mesh_3d(length, width, height, prediction)
                                
                                # Debug info (can remove later)
                                debug_info = f"Mesh: {mesh.n_points if hasattr(mesh, 'n_points') else 'N/A'} points"
                                if hasattr(mesh, 'point_data') and 'stress' in mesh.point_data:
                                    debug_info += f", Stress range: {mesh.point_data['stress'].min():.0f}-{mesh.point_data['stress'].max():.0f} Pa"
                                
                                # Convert to Plotly with stress field
                                fig = plotly_mesh_from_pyvista(mesh, stress_field=True)
                            
                            # Display interactive plot
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.caption("💡 **Color coding**: Blue = Low stress, Yellow = Medium, Red = High stress")
                            st.caption("🚗 **Enhanced visualization**: Realistic car shape with wheels, windshield, and body sections")
                            
                        except Exception as e:
                            st.error(f"Visualization error: {str(e)}")
                            # Fallback: simple wireframe
                            st.info("Showing simplified geometry...")
                            try:
                                simple_mesh = pv.Cube(center=(length/2, 0, height/2), 
                                                     x_length=length, 
                                                     y_length=width, 
                                                     z_length=height)
                                simple_fig = plotly_mesh_from_pyvista(simple_mesh)
                                st.plotly_chart(simple_fig, use_container_width=True)
                            except:
                                st.warning("3D visualization unavailable. Prediction still valid.")
                        
                        # Additional insights
                        st.markdown("---")
                        st.subheader("💡 Design Insights")
                        
                        # Compare with training data
                        df = get_data()
                        if not df.empty:
                            avg_stress = df['max_stress'].mean()
                            percentile = (df['max_stress'] < prediction).sum() / len(df) * 100
                            
                            st.write(f"**Compared to training data:**")
                            st.write(f"- Your design: {prediction:,.0f} Pa")
                            st.write(f"- Average in dataset: {avg_stress:,.0f} Pa")
                            st.write(f"- Percentile: {percentile:.1f}% (lower is better)")
                            
                            if prediction < avg_stress:
                                st.success("🎯 Your design performs better than average!")
                            else:
                                st.info("💡 Consider reducing dimensions or load to improve performance")
                    
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")
                        st.exception(e)
                else:
                    st.info("👆 Enter parameters and click 'Predict Performance' to see results")
                    
                    # Show a sample 3D visualization
                    try:
                        sample_mesh = create_car_mesh_3d(4.5, 1.8, 1.6)
                        sample_fig = plotly_mesh_from_pyvista(sample_mesh)
                        st.plotly_chart(sample_fig, use_container_width=True)
                        st.caption("Sample car geometry - Enter parameters and predict to see stress analysis")
                    except:
                        st.markdown("""
                        **What you'll see after prediction:**
                        - Interactive 3D car model
                        - Color-coded stress distribution
                        - Performance metrics and safety analysis
                        - Design optimization insights
                        """)


    # --- TAB 2: DATA EXPLORATION ---
    with tab2:
        st.header("📦 Engineering Data Lakehouse")
        
        # Explanation
        with st.expander("📖 About the Data Lakehouse", expanded=False):
            st.markdown("""
            This data lakehouse contains **real car design data** from DrivAerNet++ dataset with CFD simulations.
            
            **What's stored:**
            - **Design Parameters**: Length, width, height, drag coefficient, and aerodynamic load
            - **Performance Metrics**: Maximum stress and deflection from structural analysis
            - **Real Data**: 2,000+ car designs with actual CFD drag coefficients
            - **Additional Parameters**: Wheelbase, roof angle, windshield angle
            
            **Data Source:**
            - DrivAerNet++: 8,000+ real car designs with high-fidelity CFD simulations
            - Real drag coefficients from computational fluid dynamics
            - Multiple car configurations (fastback, notchback, estateback)
            
            **Use this to:**
            - Explore relationships between design parameters and performance
            - Identify optimal design ranges using real aerodynamic data
            - Understand stress patterns across different car geometries
            """)
        
        df = get_data()
        
        if df.empty:
            st.warning("No data found. Run ETL pipeline first.")
            st.info(f"Looking for database at: {DB_FILE}")
        else:
            # Metrics
            st.subheader("📊 Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Designs", len(df))
            col2.metric("Avg Max Stress", f"{df['max_stress'].mean()/1e6:.1f} MPa")
            col3.metric("Avg Deflection", f"{df['max_deflection'].mean()*1e6:.2f} μm")
            col4.metric("Avg Load", f"{df['load'].mean()/1000:.1f} kN")
            
            st.markdown("---")
            
            # Filter section
            st.subheader("🔍 Filter & Explore")
            col_filter1, col_filter2 = st.columns(2)
            
            with col_filter1:
                stress_threshold = st.slider(
                    "Filter by Max Stress < (Pa)", 
                    float(df['max_stress'].min()), 
                    float(df['max_stress'].max()), 
                    float(df['max_stress'].max()),
                    help="Show only designs with stress below this threshold"
                )
            
            with col_filter2:
                min_load = st.slider(
                    "Minimum Load (N)",
                    float(df['load'].min()),
                    float(df['load'].max()),
                    float(df['load'].min()),
                    help="Filter by minimum aerodynamic load"
                )
            
            filtered_df = df[(df['max_stress'] < stress_threshold) & (df['load'] >= min_load)]
            
            st.write(f"**Showing {len(filtered_df)} of {len(df)} designs**")
            
            # Interactive visualizations
            st.markdown("---")
            st.subheader("📈 Design Analysis")
            
            viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Stress Analysis", "Design Space", "Data Table"])
            
            with viz_tab1:
                # Stress vs dimensions
                fig1 = px.scatter(
                    filtered_df, 
                    x='length', 
                    y='max_stress',
                    color='load',
                    size='height',
                    hover_data=['width', 'height', 'load'],
                    labels={'length': 'Car Length (m)', 'max_stress': 'Max Stress (Pa)', 'load': 'Load (N)', 'height': 'Height (m)'},
                    title='Stress vs Car Length (colored by load, sized by height)'
                )
                st.plotly_chart(fig1, use_container_width=True)
                
                # Stress distribution histogram
                fig2 = px.histogram(
                    filtered_df,
                    x='max_stress',
                    nbins=20,
                    labels={'max_stress': 'Max Stress (Pa)', 'count': 'Number of Designs'},
                    title='Stress Distribution Across Designs'
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            with viz_tab2:
                # 3D scatter of design space
                fig3 = px.scatter_3d(
                    filtered_df,
                    x='length',
                    y='width',
                    z='height',
                    color='max_stress',
                    size='load',
                    hover_data=['max_stress', 'load'],
                    labels={'length': 'Length (m)', 'width': 'Width (m)', 'height': 'Height (m)', 
                            'max_stress': 'Max Stress (Pa)', 'load': 'Load (N)'},
                    title='3D Design Space Exploration'
                )
                fig3.update_layout(height=600)
                st.plotly_chart(fig3, use_container_width=True)
                
                # Correlation heatmap
                corr_cols = ['length', 'width', 'height', 'load', 'max_stress', 'max_deflection']
                corr_matrix = filtered_df[corr_cols].corr()
                fig4 = px.imshow(
                    corr_matrix,
                    labels=dict(x="Parameter", y="Parameter", color="Correlation"),
                    x=corr_cols,
                    y=corr_cols,
                    color_continuous_scale='RdBu',
                    aspect="auto",
                    title='Parameter Correlations'
                )
                st.plotly_chart(fig4, use_container_width=True)
            
            with viz_tab3:
                # Data table with better formatting
                display_df = filtered_df.copy()
                display_df['max_stress'] = display_df['max_stress'].apply(lambda x: f"{x:,.0f}")
                display_df['load'] = display_df['load'].apply(lambda x: f"{x:,.0f}")
                display_df['max_deflection'] = display_df['max_deflection'].apply(lambda x: f"{x:.2e}")
                st.dataframe(display_df, use_container_width=True, height=400)

    if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.exception(e)
        st.info("Please refresh the page. If the error persists, check the console logs.")
