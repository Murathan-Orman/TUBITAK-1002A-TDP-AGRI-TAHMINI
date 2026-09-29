import numpy as np
import pandas as pd
import io
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

np.random.seed(42)

patient_ids = [f"P{i:03d}" for i in range(1, 111)]
visits_per_patient = 14 

data_list = []
for p_id in patient_ids:
    base_sbp = np.random.normal(120, 10)
    base_dbp = np.random.normal(80, 5)
    base_hr = np.random.normal(75, 8)
    base_mcgill = np.random.normal(20, 10)
    
    for v in range(1, visits_per_patient + 1):
        noise = np.random.normal(0, 2)
        time_code = 1 if v <= 7 else 2
        
        sbp = base_sbp + noise * 2
        dbp = base_dbp + noise
        hr = base_hr + noise * 1.5
        mcgill = np.clip(base_mcgill + noise * 3, 0, 45)
        facs_score = np.clip(np.random.beta(2, 5) + noise * 0.01, 0, 1)
        
        pain_class = 0 if mcgill < 15 else (1 if mcgill < 30 else 2)
        data_list.append([p_id, sbp, dbp, hr, facs_score, mcgill, time_code, pain_class])

columns = ['Hasta_ID', 'Sistolik_KB', 'Diyastolik_KB', 'Kalp_Hizi', 'FACS_Skoru', 'McGill_Skoru', 'Zaman_Kodu', 'Agri_Sinifi']
df = pd.read_csv(io.StringIO("\n".join([",".join(map(str, row)) for row in data_list])), names=columns)

df['Nabiz_Basinci'] = df['Sistolik_KB'] - df['Diyastolik_KB']
df['RPP_Stres'] = (df['Sistolik_KB'] * df['Kalp_Hizi']) / 100

features = ['Sistolik_KB', 'Diyastolik_KB', 'Kalp_Hizi', 'FACS_Skoru', 'McGill_Skoru', 'Zaman_Kodu', 'Nabiz_Basinci', 'RPP_Stres']
X = df[features]
y = df['Agri_Sinifi']
groups = df['Hasta_ID']

gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.08,
    max_depth=3,
    subsample=0.85,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Basariyla Egitildi. Dogruluk (Accuracy): %{acc * 100:.2f}")

model_dosya_adi = 'tdp_pain_model.pkl'
joblib.dump(model, model_dosya_adi)
print(f"Model '{model_dosya_adi}' adiyla kaydedildi.")

yuklenen_model = joblib.load(model_dosya_adi)

yeni_hasta = pd.DataFrame([{
    'Sistolik_KB': 138,
    'Diyastolik_KB': 88,
    'Kalp_Hizi': 86,
    'FACS_Skoru': 0.62,
    'McGill_Skoru': 28,
    'Zaman_Kodu': 2,
    'Nabiz_Basinci': 50,
    'RPP_Stres': 118.68
}])

tahmin = yuklenen_model.predict(yeni_hasta)[0]
olasiliklar = yuklenen_model.predict_proba(yeni_hasta)[0]

siniflar = {0: 'Hafif Agri (0)', 1: 'Orta Siddetli Agri (1)', 2: 'Siddetli Agri (2)'}

print("\n" + "="*45)
print("KLINIK KARAR DESTEK CIKTISI")
print(f"Tahmin Edilen Durum: {siniflar[tahmin]}")
print(f"Hafif Agri Olasiligi:     %{olasiliklar[0]*100:.2f}")
print(f"Orta Agri Olasiligi:      %{olasiliklar[1]*100:.2f}")
print(f"Siddetli Agri Olasiligi:  %{olasiliklar[2]*100:.2f}")
print("="*45)
