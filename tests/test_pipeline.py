import os
import pytest
import pandas as pd
from sqlalchemy import create_engine
from src import generate_data, etl, train, inference, process_abc

def test_full_pipeline():
    # 1. Test Data Generation
    # Generate just 1 file to keep it fast
    # We need to mock generate_sample or just call it directly
    # Ideally we'd test the logic, but for end-to-end, let's just use the functions
    
    # Clean up previous runs
    if os.path.exists("data/raw"):
        for f in os.listdir("data/raw"):
            if f.endswith(".vtk"):
                os.remove(os.path.join("data/raw", f))
            
    # Generate enough samples for split
    for i in range(10):
        generate_data.generate_sample(f"test_{i}")
        
    assert os.path.exists("data/raw/sim_result_test_0.vtk")
    
    # 1.1 Test ABC Processing
    dummy_obj = "data/raw/test_abc.obj"
    with open(dummy_obj, "w") as f:
        f.write("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
        
    process_abc.process_obj_file(dummy_obj)
    expected_vtk = "data/raw/processed_test_abc.vtk"
    assert os.path.exists(expected_vtk)
    # Cleanup dummy obj
    os.remove(dummy_obj)
    
    # 2. Test ETL
    # Since etl scans the whole dir, it will pick up the test file + others
    # We just want to ensure it runs without error and DB is populated
    etl.run_etl()
    assert os.path.exists("data/metadata.db")
    
    engine = create_engine("sqlite:///data/metadata.db")
    df = pd.read_sql('simulations', engine)
    assert len(df) > 0
    assert 'max_stress' in df.columns
    
    # 3. Test Training
    train.train_model()
    assert os.path.exists("model.pkl")
    
    # 4. Test Inference
    # Predict using the newly trained model
    stress = inference.predict(1.0, 0.2, 0.1, 5000)
    assert stress > 0
    assert isinstance(stress, float)

if __name__ == "__main__":
    test_full_pipeline()
