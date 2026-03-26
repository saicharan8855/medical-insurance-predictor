import joblib
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.preprocess import preprocess

X_train, X_test, y_train, y_test, scaler = preprocess()


model = joblib.load('models/model.pkl')
print("Model loaded ✅")


y_pred = model.predict(X_test)


r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n── Model Evaluation ──────────────────")
print(f"R²   : {r2:.4f}")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
