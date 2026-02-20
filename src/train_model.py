import pandas as pd
import pickle
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now we can import from src
from src.preprocess import clean_text

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "Combined Data.csv")
model_dir = os.path.join(BASE_DIR, "model")

# Create model directory if it doesn't exist
os.makedirs(model_dir, exist_ok=True)

def load_and_prepare_data():
    """Load and prepare the dataset"""
    print("Loading data...")
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"❌ Data file not found at: {data_path}")
        print("Please make sure 'Combined Data.csv' is in the 'data' folder.")
        sys.exit(1)
    
    df = pd.read_csv(data_path)
    
    # Drop the unnamed index column if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    print(f"Columns in dataset: {df.columns.tolist()}")
    print(f"Dataset shape: {df.shape}")
    
    # Check if required columns exist
    if 'statement' not in df.columns:
        print("❌ Column 'statement' not found in the dataset!")
        print(f"Available columns: {df.columns.tolist()}")
        sys.exit(1)
    
    if 'status' not in df.columns:
        print("❌ Column 'status' not found in the dataset!")
        print(f"Available columns: {df.columns.tolist()}")
        sys.exit(1)
    
    # Check class distribution
    print("\nOriginal class distribution:")
    print(df['status'].value_counts())
    
    # Clean the data
    df = df.dropna(subset=['statement', 'status'])
    
    # Create binary labels (Anxiety = 1, Normal = 0)
    df['label'] = df['status'].apply(lambda x: 1 if x == 'Anxiety' else 0)
    
    # Clean the statements with progress indicator
    print("\nCleaning texts...")
    cleaned_statements = []
    total = len(df)
    for i, text in enumerate(df['statement']):
        if i % 1000 == 0:
            print(f"  Progress: {i}/{total} texts cleaned")
        cleaned_statements.append(clean_text(text))
    
    df['cleaned_statement'] = cleaned_statements
    
    # Remove empty strings after cleaning
    df = df[df['cleaned_statement'].str.strip() != '']
    
    print(f"\n✅ Final dataset size: {len(df)} samples")
    print(f"Final class distribution:")
    print(df['label'].value_counts())
    print(f"  - Normal (0): {sum(df['label'] == 0)} samples")
    print(f"  - Anxiety (1): {sum(df['label'] == 1)} samples")
    
    return df['cleaned_statement'], df['label']

def plot_confusion_matrix(y_true, y_pred, labels=['Normal', 'Anxiety']):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(model_dir, 'confusion_matrix.png')
    plt.savefig(plot_path)
    print(f"✅ Confusion matrix saved to: {plot_path}")
    plt.close()

def plot_feature_importance(feature_names, coefficients, top_n=20):
    """Plot top features for each class"""
    # Get top features for anxiety
    top_anxiety_idx = np.argsort(coefficients)[-top_n:]
    top_anxiety_features = [feature_names[i] for i in top_anxiety_idx]
    top_anxiety_coef = [coefficients[i] for i in top_anxiety_idx]
    
    # Get top features for normal
    top_normal_idx = np.argsort(coefficients)[:top_n]
    top_normal_features = [feature_names[i] for i in top_normal_idx]
    top_normal_coef = [coefficients[i] for i in top_normal_idx]
    
    # Plot anxiety features
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), top_anxiety_coef[::-1])
    plt.yticks(range(top_n), top_anxiety_features[::-1])
    plt.xlabel('Coefficient Value')
    plt.title(f'Top {top_n} Words Indicating Anxiety')
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'top_anxiety_words.png'))
    plt.close()
    
    # Plot normal features
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), top_normal_coef)
    plt.yticks(range(top_n), top_normal_features)
    plt.xlabel('Coefficient Value')
    plt.title(f'Top {top_n} Words Indicating Normal')
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'top_normal_words.png'))
    plt.close()
    
    print(f"✅ Feature importance plots saved to {model_dir}")

def train_model():
    """Train the model"""
    print("=" * 60)
    print(" TRAINING MENTAL HEALTH DETECTION MODEL")
    print("=" * 60)
    
    # Load data
    X, y = load_and_prepare_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Create pipeline with TF-IDF and classifier
    print("\n🔧 Creating pipeline...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            sublinear_tf=True
        )),
        ('clf', LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced',
            C=1.0
        ))
    ])
    
    # Train model
    print("📚 Training model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    print("📊 Evaluating model...")
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print(" MODEL PERFORMANCE")
    print(f"{'='*50}")
    print(f"✅ Accuracy: {accuracy:.4f}")
    print(f"\n📈 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Anxiety']))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)
    
    # Get feature importance
    try:
        feature_names = pipeline.named_steps['tfidf'].get_feature_names_out()
        coefficients = pipeline.named_steps['clf'].coef_[0]
        
        # Top words for Anxiety
        top_anxiety_idx = np.argsort(coefficients)[-20:]
        top_anxiety_words = [(feature_names[i], coefficients[i]) for i in top_anxiety_idx]
        
        # Top words for Normal
        top_normal_idx = np.argsort(coefficients)[:20]
        top_normal_words = [(feature_names[i], coefficients[i]) for i in top_normal_idx]
        
        print(f"\n{'='*50}")
        print("🔝 TOP WORDS INDICATING ANXIETY:")
        print(f"{'='*50}")
        for word, coef in reversed(top_anxiety_words):
            print(f"  • {word}: {coef:.4f}")
        
        print(f"\n{'='*50}")
        print("🔝 TOP WORDS INDICATING NORMAL:")
        print(f"{'='*50}")
        for word, coef in top_normal_words:
            print(f"  • {word}: {coef:.4f}")
        
        # Plot feature importance
        plot_feature_importance(feature_names, coefficients)
        
    except Exception as e:
        print(f"⚠️ Could not extract feature importance: {e}")
    
    # Save model
    model_path = os.path.join(model_dir, "model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    
    print(f"\n✅ Model saved to: {model_path}")
    print(f"✅ Model type: {type(pipeline)}")
    print(f"✅ Pipeline steps: {list(pipeline.named_steps.keys())}")
    
    return pipeline, accuracy

def test_model(model):
    """Test the model with some examples"""
    print(f"\n{'='*50}")
    print("🧪 TESTING WITH EXAMPLES")
    print(f"{'='*50}")
    
    examples = [
        "I feel so anxious and worried all the time",
        "I can't stop thinking about bad things happening",
        "My heart is racing and I feel scared",
        "I'm so excited for the concert tomorrow!",
        "Just finished watching a great movie",
        "Having dinner with friends tonight",
        "I'm restless and can't sleep at night",
        "Feeling nervous about my presentation",
        "Beautiful weather today!",
        "I'm worried something bad will happen"
    ]
    
    results = []
    for text in examples:
        cleaned = clean_text(text)
        proba = model.predict_proba([cleaned])[0]
        pred = model.predict([cleaned])[0]
        
        confidence = proba[1] if pred == 1 else proba[0]
        result = "⚠ ANXIETY" if pred == 1 else "✅ NORMAL"
        
        print(f"\n📝 Text: {text[:50]}...")
        print(f"   → {result} (confidence: {confidence:.2%})")
        
        results.append({
            'text': text,
            'prediction': result,
            'confidence': confidence
        })
    
    return results

if __name__ == "__main__":
    # Train model
    model, accuracy = train_model()
    
    # Test examples
    test_model(model)
    
    print(f"\n{'='*50}")
    print("🎉 TRAINING COMPLETE!")
    print(f"{'='*50}")
    print(f"✅ Model accuracy: {accuracy:.4f}")
    print(f"✅ Model saved in: {model_dir}")
    print(f"\nTo start the API, run: uvicorn src.api:app --reload")
    print(f"To start the web app, run: streamlit run app.py")