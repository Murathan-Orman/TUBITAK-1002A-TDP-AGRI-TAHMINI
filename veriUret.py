import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")

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
df = pd.DataFrame(data_list, columns=columns)

df['Nabiz_Basinci'] = df['Sistolik_KB'] - df['Diyastolik_KB']
df['RPP_Stres'] = (df['Sistolik_KB'] * df['Kalp_Hizi']) / 100

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("--- 110 HASTANIN İLK ÖLÇÜMLERİ (110 Satır) ---")
pd.set_option('display.max_rows', 120)
print(df.drop_duplicates(subset=['Hasta_ID']))

print("\n--- 110 HASTANIN ORTALAMA DEĞERLERİ (110 Satır) ---")
df_110_hasta = df.groupby('Hasta_ID').mean(numeric_only=True).reset_index()
print(df_110_hasta)

df.to_csv("sentetik_veri_seti_tum_olcumler.csv", index=False)
df_110_hasta.to_csv("110_hasta_ozeti.csv", index=False)
print("\n[BİLGİ] 'sentetik_veri_seti_tum_olcumler.csv' ve '110_hasta_ozeti.csv' başarıyla oluşturuldu.")

plt.figure(figsize=(7, 4))
sns.countplot(x='Agri_Sinifi', data=df)
plt.title('1. Agri Sinif Dagilimi')
plt.show()

plt.figure(figsize=(7, 4))
sns.histplot(df['Sistolik_KB'], kde=True, color='teal')
plt.title('2. Sistolik Kan Basinci Dagilimi')
plt.show()

plt.figure(figsize=(7, 4))
sns.histplot(df['Kalp_Hizi'], kde=True, color='coral')
plt.title('3. Kalp Hizi Dagilimi')
plt.show()

plt.figure(figsize=(7, 4))
sns.boxplot(x='Agri_Sinifi', y='Nabiz_Basinci', data=df)
plt.title('4. Siniflara Gore Nabiz Basinci')
plt.show()

plt.figure(figsize=(7, 4))
sns.boxplot(x='Agri_Sinifi', y='RPP_Stres', data=df)
plt.title('5. Siniflara Gore RPP Stres Indeksi')
plt.show()

plt.figure(figsize=(7, 4))
sns.boxplot(x='Agri_Sinifi', y='FACS_Skoru', data=df)
plt.title('6. Siniflara Gore FACS Yuz Puanlari')
plt.show()

plt.figure(figsize=(7, 4))
sns.scatterplot(x='RPP_Stres', y='McGill_Skoru', hue='Agri_Sinifi', palette='coolwarm', data=df)
plt.title('7. RPP ve McGill Skoru Korelasyonu')
plt.show()

plt.figure(figsize=(7, 5))
numeric_cols = ['Sistolik_KB', 'Diyastolik_KB', 'Kalp_Hizi', 'FACS_Skoru', 'McGill_Skoru', 'Nabiz_Basinci', 'RPP_Stres']
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='Blues', fmt='.2f')
plt.title('8. Parametreler Arasi Korelasyon Matrisi')
plt.tight_layout()
plt.show()