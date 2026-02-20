import pickle
import os
import sys
import pandas as pd
import numpy as np

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now we can import from src
from src.preprocess import clean_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "model", "model.pkl")

def load_model():
    """Load the trained model"""
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully")
        print(f"✅ Model type: {type(model)}")
        return model
    except FileNotFoundError as e:
        print(f"❌ Error loading model: {e}")
        print("Please train the model first using: python src/train_model.py")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

# Load model at module import
model = load_model()

def predict_text(text):
    """
    Predict if text indicates anxiety
    
    Args:
        text (str): Input text to analyze
    
    Returns:
        tuple: (prediction_label, confidence_score, binary_prediction)
    """
    try:
        # Clean the text
        cleaned = clean_text(text)
        
        # For scikit-learn pipelines, we need to pass the text as a list of strings
        # The pipeline will handle vectorization automatically
        text_list = [cleaned]
        
        # Predict
        prediction = int(model.predict(text_list)[0])
        
        # Get probability
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(text_list)[0]
            confidence = float(proba[1]) if prediction == 1 else float(proba[0])
        else:
            confidence = 0.5
        
        # Convert prediction to label
        if prediction == 1:
            label = "⚠ Anxiety Detected"
        else:
            label = "✅ Normal Statement"
        
        return label, confidence, prediction
    
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}", 0.0, -1

def batch_predict(texts):
    """Predict for multiple texts"""
    results = []
    for i, text in enumerate(texts):
        label, confidence, pred = predict_text(text)
        results.append({
            'id': i+1,
            'text': text[:100] + '...' if len(text) > 100 else text,
            'prediction': label,
            'confidence': confidence,
            'binary': pred
        })
    return results

def analyze_csv(file_path, text_column='statement'):
    """Analyze all texts in a CSV file"""
    try:
        df = pd.read_csv(file_path)
        if text_column not in df.columns:
            print(f"Column '{text_column}' not found. Available columns: {df.columns.tolist()}")
            return
        
        print(f"\nAnalyzing {len(df)} statements...")
        results = []
        
        for idx, text in enumerate(df[text_column].fillna('')):
            if text and isinstance(text, str):
                label, confidence, pred = predict_text(text)
                results.append({
                    'id': idx+1,
                    'text': text[:50] + '...' if len(text) > 50 else text,
                    'prediction': label,
                    'confidence': confidence
                })
        
        # Summary
        anxiety_count = sum(1 for r in results if 'Anxiety' in r['prediction'])
        print(f"\n{'='*50}")
        print("ANALYSIS SUMMARY")
        print(f"{'='*50}")
        print(f"Total statements: {len(results)}")
        print(f"Anxiety detected: {anxiety_count} ({anxiety_count/len(results)*100:.1f}%)")
        print(f"Normal statements: {len(results)-anxiety_count} ({(len(results)-anxiety_count)/len(results)*100:.1f}%)")
        
        return results
        
    except Exception as e:
        print(f"Error analyzing CSV: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Interactive prediction interface"""
    print("=" * 60)
    print(" MENTAL HEALTH DETECTION SYSTEM")
    print("=" * 60)
    print("\nModel loaded successfully!")
    print("Type 'exit' to quit, 'batch' for batch mode, 'csv' to analyze a CSV file\n")

    while True:
        user_input = input("\n📝 Enter a statement: ")

        if user_input.lower() == "exit":
            print("👋 Exiting...")
            break
        
        elif user_input.lower() == "batch":
            print("Enter multiple statements (one per line, empty line to finish):")
            texts = []
            while True:
                line = input()
                if line.strip() == "":
                    break
                texts.append(line)
            
            results = batch_predict(texts)
            print("\n📊 Batch Results:")
            for r in results:
                print(f"  {r['id']}. {r['text']}")
                print(f"     → {r['prediction']} ({r['confidence']:.2%})")
        
        elif user_input.lower() == "csv":
            csv_path = input("Enter path to CSV file: ")
            if os.path.exists(csv_path):
                text_col = input("Enter text column name (default: 'statement'): ") or 'statement'
                analyze_csv(csv_path, text_col)
            else:
                print(f"File not found: {csv_path}")
        
        elif user_input.strip():
            label, confidence, pred = predict_text(user_input)
            
            # Color coding for terminal
            if pred == 1:
                print(f"\n🔍 \033[91m{label}\033[0m")
            else:
                print(f"\n🔍 \033[92m{label}\033[0m")
            print(f"📈 Confidence: {confidence:.2%}")
            print("-" * 40)

if __name__ == "__main__":
    main()