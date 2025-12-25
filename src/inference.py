import joblib
import pandas as pd
import argparse
import sys

MODEL_PATH = "model.pkl"

def predict(length, width, height, load):
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print("Error: Model file not found. Run src/train.py first.")
        return

    input_data = pd.DataFrame([{
        'length': length,
        'width': width,
        'height': height,
        'load': load
    }])
    
    prediction = model.predict(input_data)[0]
    return prediction

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Max Stress for a Beam")
    parser.add_argument("--length", type=float, required=True, help="Length (m)")
    parser.add_argument("--width", type=float, required=True, help="Width (m)")
    parser.add_argument("--height", type=float, required=True, help="Height (m)")
    parser.add_argument("--load", type=float, required=True, help="Load (N)")
    
    args = parser.parse_args()
    
    stress = predict(args.length, args.width, args.height, args.load)
    print(f"Predicted Max Stress: {stress:.2f} Pa")
