import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib

print("--- Starting Binary AI Training ---")

# 1. LOAD
df = pd.read_csv('data/processed_data.csv')
X = df.drop(columns=['Target'])
y = df['Target']

# 2. SMOTE (Balance the classes)
print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# 3. SPLIT
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# 4. TRAIN
print("Training XGBoost...")
model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# 5. EVALUATE
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n>>> FINAL ACCURACY: {acc * 100:.2f}%")
print(classification_report(y_test, y_pred))

# 6. SAVE
joblib.dump(model, 'data/model.pkl')
print("Model saved.")