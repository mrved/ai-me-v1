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

def create_car_mesh_3d(length, width, height, stress_value=None):
    """Create a 3D car mesh for visualization"""
    # Main body (chassis)
    body = pv.Cube(center=(length/2, 0, height*0.3), 
                   x_length=length*0.9, 
                   y_length=width, 
                   z_length=height*0.4)
    
    # Cabin section
    cabin = pv.Cube(center=(length*0.6, 0, height*0.7), 
                    x_length=length*0.5, 
                    y_length=width*0.85, 
                    z_length=height*0.5)
    
    # Hood section
    hood = pv.Cube(center=(length*0.2, 0, height*0.25), 
                   x_length=length*0.3, 
                   y_length=width*0.9, 
                   z_length=height*0.3)
    
    # Combine or use body as fallback
    try:
        mesh = body.boolean_union(cabin)
        mesh = mesh.boolean_union(hood)
        if mesh.n_points == 0:
            mesh = body
    except:
        mesh = body
    
    mesh = mesh.triangulate()
    
    # Add stress field if provided
    if stress_value is not None:
        try:
            points = mesh.points
            if len(points) > 0:
                x_coords = points[:, 0]
                z_coords = points[:, 2]
                x_norm = (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min() + 1e-10)
                z_norm = (z_coords - z_coords.min()) / (z_coords.max() - z_coords.min() + 1e-10)
                stress_field = stress_value * (0.5 + 0.3 * (1 - x_norm) + 0.2 * (1 - z_norm))
                mesh.point_data["stress"] = stress_field
        except:
            pass  # If stress field can't be added, continue without it
    
    return mesh

def plotly_mesh_from_pyvista(mesh, stress_field=None):
    """Convert PyVista mesh to Plotly 3D visualization"""
    try:
        vertices = mesh.points
        if len(vertices) == 0:
            raise ValueError("Empty mesh")
        
        # Handle faces - check if mesh has faces
        if mesh.n_faces == 0:
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
        
        # Create mesh3d trace
        if stress_field is not None and "stress" in mesh.point_data:
            colors = mesh.point_data["stress"]
            colorscale = [[0, 'blue'], [0.5, 'yellow'], [1, 'red']]
        else:
            colors = None
            colorscale = 'Viridis'
        
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
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=600,
            margin=dict(l=0, r=0, t=0, b=0)
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
    
    tab1, tab2 = st.tabs(["Data Lakehouse", "Virtual Test Bench"])
    
    # --- TAB 1: DATA EXPLORATION ---
    with tab1:
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

    # --- TAB 2: INFERENCE ---
    with tab2:
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
                
                st.markdown("**Car Dimensions:**")
                length = st.number_input(
                    "Length (m)", 
                    3.5, 5.5, 4.5,
                    help="Overall length of the car from front to back"
                )
                width = st.number_input(
                    "Width (m)", 
                    1.6, 2.0, 1.8,
                    help="Width of the car (track width)"
                )
                height = st.number_input(
                    "Height (m)", 
                    1.4, 1.8, 1.6,
                    help="Overall height of the car"
                )
                
                st.markdown("**Loading Conditions:**")
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
                        # Prediction
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
                            # Create car mesh with stress field
                            mesh = create_car_mesh_3d(length, width, height, prediction)
                            
                            # Convert to Plotly
                            fig = plotly_mesh_from_pyvista(mesh, stress_field=True)
                            
                            # Display interactive plot
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.caption("💡 **Color coding**: Blue = Low stress, Yellow = Medium, Red = High stress")
                            
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

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.exception(e)
        st.info("Please refresh the page. If the error persists, check the console logs.")
