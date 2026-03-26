import os
import joblib
from src.preprocess import preprocess
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test, scaler = preprocess()

model = LinearRegression()
model.fit(X_train, y_train)

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
print("Model saved!")