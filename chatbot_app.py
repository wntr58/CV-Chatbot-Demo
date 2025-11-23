import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import pandas as pd
import numpy as np
import base64

# --- 1. VERİ KÜMESİ (TRAINING DATA) ---
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
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(df['soru'])
model = LinearSVC()
model.fit(X_vectorized, df['niyet'])

# --- 3. GÜNCELLENMİŞ CEVAP HAVUZU ---
CEVAPLAR = {
    'PLC': "Vanderlande stajında Siemens PLC (TIA Portal) kullanarak sistem izleme ve temel müdahaleler yaptım. Ayrıca SCADA ve HMI programlama tecrübem var.",
    # Yazılım cevabı güncellendi ve kaynakçalar düzenlendi:
    [cite_start]'Yazılım': "Python, C/C++ ve MS SQL gibi dillerde iyi seviyede yetkinliğe sahibim[cite: 57, 58, 66]. [cite_start]Otomasyon alanındaki güçlü yönlerim arasında Görüntü İşleme ve ROS2 tecrübesi yer almaktadır[cite: 64, 65].",
    'Staj': "Neocom'da Zayıf Akım Sistemleri (Kamera/Yangın/Anons) ve Vanderlande'da Lojistik Otomasyon sistemlerinde çalıştım.",
    'Eğitim': "Kocaeli Üniversitesi Mekatronik Mühendisliği (%30 İngilizce) bölümünden mezunum."
}

def niyet_siniflandir_ve_cevapla(soru):
    soru_vectorized = vectorizer.transform([soru])
    tahmin_edilen_niyet = model.predict(soru_vectorized)[0]
    return tahmin_edilen_niyet, CEVAPLAR.get(tahmin_edilen_niyet, "Ne yazık ki bu konudaki bilgiyi CV'mden tam olarak çıkaramadım. Lütfen farklı bir açıdan sorun.")

# --- 4. STREAMLIT ARAYÜZÜ (Gelişmiş) ---
st.set_page_config(page_title="Yahya Osman Tamdoğan CV Chatbot", layout="wide")

# Kenar çubuğu (Sidebar)
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Robot_icon.svg/1024px-Robot_icon.svg.png", width=100)
    st.header("🤖 CV Asistanı Hakkında")
    st.info(
        [cite_start]"Bu Chatbot, Yahya Osman Tamdoğan'ın CV'sini [cite: 41-82] temel alarak, mülakat simülasyonu amacıyla geliştirilmiş basit bir prototiptir. "
        "Sorularınızı **PLC, Yazılım, Staj veya Eğitim** niyetlerinden birine göre sınıflandırmaya çalışır."
    )
    st.markdown("---")
    st.subheader("Hızlı Bağlantılar")
    [cite_start]st.markdown(f"**LinkedIn:** [Yahya Osman Tamdoğan LinkedIn](https://www.linkedin.com/in/yahyaosmantamdogan) [cite: 54]")
    
    # Sohbeti Temizle Butonu
    if st.button("Sohbeti Temizle", help="Sohbet geçmişini siler ve sıfırdan başlatır."):
        st.session_state.mesajlar = []
        st.experimental_rerun() # Uygulamayı yeniden yükler

st.title("👨‍💻 Yahya Osman Tamdoğan CV Asistanı")
st.markdown("Mekatronik Mühendisi Yahya Osman Tamdoğan'ın yetkinlikleri hakkında soru sormaya başlayın:")

# Mesaj geçmişini tutma
if "mesajlar" not in st.session_state:
    st.session_state.mesajlar = []

# Daha önceki mesajları gösterme
for gonderici, mesaj, niyet in st.session_state.mesajlar:
    st.chat_message(gonderici).write(mesaj)
    if gonderici == "assistant":
        st.caption(f"🤖 Tahmin Edilen Niyet: {niyet}")

# Kullanıcı girişi
if prompt := st.chat_input("Hangi otomasyon tecrübelerine sahipsin?"):
    # Kullanıcının mesajını kaydet ve göster
    st.session_state.mesajlar.append(("user", prompt, None))
    st.chat_message("user").write(prompt)
    
    # Chatbot cevabını al
    niyet, cevap = niyet_siniflandir_ve_cevapla(prompt)
    
    # Chatbot cevabını kaydet ve göster
    st.session_state.mesajlar.append(("assistant", cevap, niyet))
    
    # Cevabı arayüzde göster
    st.chat_message("assistant").write(cevap)
    st.caption(f"🤖 Tahmin Edilen Niyet: {niyet}")

st.markdown("---")
# Alt Bilgi (Footer)
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: #808080;
        text-align: center;
        padding: 5px;
        font-size: 0.8em;
    }
    </style>
    <div class="footer">
        CV'deki bilgilere dayanarak oluşturulmuş yapay zeka prototipidir.
    </div>
    """, 
    unsafe_allow_html=True
)
