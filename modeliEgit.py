import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

print("1. Zenginleştirilmiş veri seti okunuyor...")
df = pd.read_csv(r'C:\Users\morma\OneDrive\Resimler\Desktop\sentetikVeri.csv')

print("2. Veriler hasta bazlı ayrılıyor (%80 Eğitim, %20 Test)...")
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(df, groups=df['Hasta_ID']))

train_df = df.iloc[train_idx]
test_df = df.iloc[test_idx]

ozellikler = [
    'Yas', 'BMI', 'Sistolik_KB', 'Diyastolik_KB', 'Kalp_Hizi', 
    'Nabiz_Basinci', 'Stres_Indeksi_RPP', 'Gunun_Saati_Numerik',
    'SF_MPQ_Skoru', 'Yuz_FACS_Agri_Skoru', 
    'Cinsiyet', 'Duygu_Durumu_EMA'
]
hedef = 'Agri_Sinifi'

X_train, y_train = train_df[ozellikler], train_df[hedef]
X_test, y_test = test_df[ozellikler], test_df[hedef]

print("3. Küçük YZ Modeli (Gradient Boosting) eğitiliyor...")
one_hot = ColumnTransformer(
    transformers=[('kategorik', OneHotEncoder(handle_unknown='ignore'), ['Cinsiyet', 'Duygu_Durumu_EMA'])],
    remainder='passthrough'
)

model = Pipeline([
    ('donusturucu', one_hot),
    ('siniflandirici', GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42))
])

model.fit(X_train, y_train)

y_tahmin = model.predict(X_test)

dogru_sayisi = (y_test == y_tahmin).sum()
toplam_olcum = len(y_test)
dogruluk_orani = accuracy_score(y_test, y_tahmin) * 100

print("\n" + "="*45)
print("             MODEL TEST ÇIKTILARI            ")
print("="*45)
print(f"Test Edilen Toplam Ölçüm Sayısı : {toplam_olcum}")
print(f"Modelin Doğru Bildiği Sayı       : {dogru_sayisi}")
print(f"Modelin Yanıldığı Sayı          : {toplam_olcum - dogru_sayisi}")
print(f"Genel Test Doğruluğu (Accuracy) : %{dogruluk_orani:.2f}")
print("="*45)

print("\n--- SINIFLANDIRMA RAPORU (F1-SKORLARI) ---")
print(classification_report(y_test, y_tahmin, target_names=['Hafif Ağrı (0)', 'Orta Ağrı (1)', 'Şiddetli Ağrı (2)']))

print("--- KARMAŞIKLIK MATRİSİ (Hangi sınıftan kaç doğru bildi?) ---")
print(confusion_matrix(y_test, y_tahmin))