import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import pandas as pd
import numpy as np

# --- 1. VERİ KÜMESİ (TRAINING DATA) ---
# Sorular (X) ve Niyetler (y)
data = {
    'soru': [
        "PLC deneyimin var mı?", "TIA Portal biliyor musun?", "Siemens otomasyon tecrüben nedir?", 
        "HMI programlamayı biliyor musun?", "Python'da iyi misin?", "Hangi yazılım dillerini biliyorsun?",
        "Görüntü İşleme projen var mı?", "ROS2 ile çalıştın mı?", "Vanderlande stajında ne yaptın?",
        "Neocom'da ne gibi işler yaptın?", "Staj tecrübelerinden bahseder misin?", "Eğitim bilgilerini alabilir miyim?",
        "Hangi üniversiteden mezunsun?", "Mekatronik bilgin nedir?"
    ],
    'niyet': [
        'PLC', 'PLC', 'PLC', 
        'PLC', 'Yazılım', 'Yazılım', 
        'Yazılım', 'Yazılım', 'Staj', 
        'Staj', 'Staj', 'Eğitim', 
        'Eğitim', 'Eğitim'
    ]
}

df = pd.DataFrame(data)

# --- 2. MODEL EĞİTİMİ (Training) ---
# Basit bir metin sınıflandırma modeli eğitimi
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(df['soru'])
model = LinearSVC()
model.fit(X_vectorized, df['niyet'])

# --- 3. CEVAP HAVUZU (CV'den çıkarılan bilgiler [cite: 41-82]) ---
CEVAPLAR = {
    'PLC': "Vanderlande stajında Siemens PLC (TIA Portal) kullanarak sistem izleme ve temel müdahaleler yaptım. Ayrıca SCADA ve HMI programlama tecrübem var[cite: 79, 82].",
    'Yazılım': "Python, C/C++ ve MS SQL gibi dillerde iyi seviyedeyim[cite: 57, 58, 66]. Özellikle Görüntü İşleme ve ROS2 tecrübem otomasyon alanında güçlü yönlerimdir[cite: 64, 65].",
    'Staj': "Neocom'da Zayıf Akım Sistemleri (Kamera/Yangın/Anons) [cite: 71, 72, 73] ve Vanderlande'da Lojistik Otomasyon sistemlerinde çalıştım[cite: 77].",
    'Eğitim': "Kocaeli Üniversitesi Mekatronik Mühendisliği (%30 İngilizce) bölümünden mezunum[cite: 50]."
}

def niyet_siniflandir_ve_cevapla(soru):
    # Soruyu vektörleştir ve niyetini tahmin et
    soru_vectorized = vectorizer.transform([soru])
    tahmin_edilen_niyet = model.predict(soru_vectorized)[0]
    
    # Tahmin edilen niyete göre cevap ver
    return CEVAPLAR.get(tahmin_edilen_niyet, "Ne yazık ki bu konudaki bilgiyi CV'mden tam olarak çıkaramadım. Lütfen farklı bir açıdan sorun.")

# --- 4. STREAMLIT ARAYÜZÜ (Gelişmiş) ---
st.set_page_config(page_title="Yahya Osman Tamdoğan CV Chatbot", layout="wide")

st.title("🤖 Yahya Osman Tamdoğan CV Asistanı: Mekatronik Yetkinlikler")
st.markdown("---")
st.caption("Bu prototip, metin sınıflandırma modelini kullanarak soruları **PLC, Yazılım, Staj veya Eğitim** niyetlerinden birine göre yanıtlar.")


# Mesaj geçmişini tutma
if "mesajlar" not in st.session_state:
    st.session_state.mesajlar = []

# Daha önceki mesajları gösterme
for gonderici, mesaj in st.session_state.mesajlar:
    st.chat_message(gonderici).write(mesaj)

# Kullanıcı girişi
if prompt := st.chat_input("HMI tecrüben nedir?", disabled=(len(st.session_state.mesajlar) >= 20)):
    # Kullanıcının mesajını kaydet ve göster
    st.session_state.mesajlar.append(("user", prompt))
    st.chat_message("user").write(prompt)
    
    # Chatbot cevabını al
    cevap = niyet_siniflandir_ve_cevapla(prompt)
    
    # Chatbot cevabını kaydet ve göster
    st.session_state.mesajlar.append(("assistant", cevap))
    st.chat_message("assistant").write(f"**Tahmin Edilen Niyet:** {model.predict(vectorizer.transform([prompt]))[0]}\n\n{cevap}")
