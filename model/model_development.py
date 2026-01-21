import pandas as pd
import joblib
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# --- 1. Load Data (Efficiently from sklearn) ---
try:
    print("Loading Wine dataset...")
    data = load_wine()
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['cultivar'] = data.target
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load dataset. {e}")
    exit()

# --- 2. Feature Selection ---
# Instructions require exactly 6 features. We select 6 numerical features.
selected_features = [
    'alcohol', 
    'malic_acid', 
    'total_phenols', 
    'flavanoids', 
    'color_intensity', 
    'hue'
]

X = df[selected_features]
y = df['cultivar']

# --- 3. Split Data (Validation Step) ---
# Stratified split ensures all 3 cultivars are represented in train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 4. Build Pipeline (Efficiency Upgrade) ---
# "Pipeline" bundles the Scaler and Model into one object.
# This ensures that when we predict later, we don't need to manually scale inputs.
pipeline = Pipeline([
    ('scaler', StandardScaler()),       # Step 1: Standardize features
    ('clf', RandomForestClassifier(random_state=42)) # Step 2: Random Forest
])

# --- 5. Train the Model ---
print("Training model...")
pipeline.fit(X_train, y_train)

# --- 6. Evaluate (Rubric Requirement) ---
print("\n--- Model Evaluation ---")
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- 7. Save the Model ---
# We save the WHOLE pipeline. This means the scaler is saved inside it!
model_filename = 'wine_cultivar_model.pkl'
joblib.dump(pipeline, model_filename)

print(f"\nSUCCESS: Pipeline saved as '{model_filename}'")