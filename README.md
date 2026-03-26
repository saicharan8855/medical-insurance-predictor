\# 🏥 Medical Insurance Cost Predictor



A machine learning project that predicts medical insurance charges based on personal attributes using Linear Regression.



🔗 \*\*Live App\*\*: \[saicharan-medical-insurance-predictor.streamlit.app](https://saicharan-medical-insurance-predictor.streamlit.app/)



\---



\## 📁 Project Structure



```

medical\_insurance/

│

├── data/

│   └── insurance.csv

│

├── models/

│   ├── model.pkl

│   └── scaler.pkl

│

├── notebooks/

│   └── exploration.ipynb

│

├── src/

│   ├── \_\_init\_\_.py

│   ├── preprocess.py

│   ├── model.py

│   └── evaluate.py

│

├── main.py

├── app.py

└── requirements.txt

```



\---



\## 📊 Dataset



The dataset contains \*\*1338 records\*\* with the following features:



| Feature | Type | Description |

|---|---|---|

| age | Numeric | Age of the individual |

| sex | Categorical | male / female |

| bmi | Numeric | Body Mass Index |

| children | Numeric | Number of dependents |

| smoker | Categorical | yes / no |

| region | Categorical | northeast / northwest / southeast / southwest |

| charges | Numeric | Medical insurance cost (target) |



\---



\## ⚙️ How It Works



1\. \*\*EDA\*\* — Explored distributions, correlations, and feature relationships

2\. \*\*Preprocessing\*\* — Encoded categoricals, split data, applied StandardScaler

3\. \*\*Model\*\* — Trained Linear Regression on 80% of data

4\. \*\*Evaluate\*\* — Measured R², MAE, RMSE on test set

5\. \*\*Deploy\*\* — Streamlit app deployed on Streamlit Cloud



\---



\## 📈 Model Performance



| Metric | Score |

|---|---|

| R² | 0.74 |

| MAE | 4281.68 |

| RMSE | 6039.97 |



\---



\## 🚀 Run Locally



```bash

\# Install dependencies

pip install -r requirements.txt



\# Train the model

python main.py



\# Launch the app

streamlit run app.py

```



\---



\## 🛠️ Tech Stack



\- Python

\- Scikit-learn

\- Pandas \& NumPy

\- Streamlit

\- Joblib



\---



\## 👤 Author



\*\*Sai Charan\*\*

