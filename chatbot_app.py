import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import pandas as pd
import numpy as np

# --- 1. VERİ KÜMESİ (TRAINING DATA) ---
# Model, bu soru-cevap eşleşmelerini kullanarak niyetleri öğrenir
data = {
    'soru': [
        "PLC deneyimin var mı?", "TIA Portal biliyor musun?", "Siemens otomasyon tecrüben nedir?", 
        "HMI programlamayı biliyor musun?", "Python'da iyi misin?", "Hangi yazılım dillerini biliyorsun?",
        "Görüntü İşleme projen var mı?", "ROS2 ile çalıştın mı?", "Vanderlande stajında ne yaptın?",
        "Neocom'da ne gibi işler yaptın?", "Staj tecrübelerinden bahseder misin?", "Eğitim bilgilerini alabilir miyim?",
        "Hangi üniversiteden mezunsun?", "Mekatronik bilgin nedir?", "Hangi dilleri biliyorsun?",
        "Otomasyon becerilerin neler?", "Lojistik sistemlerde çalıştın mı?", "Diploman ne?",
        "Sql biliyor musun?" 
    ],
    'niyet': [
        'PLC', 'PLC', 'PLC', 
        'PLC', 'Yazılım', 'Yazılım', 
        'Yazılım', 'Yazılım', 'Staj', 
        'Staj', 'Staj', 'Eğitim', 
        'Eğitim', 'Eğitim', 'Yazılım',
        'PLC', 'Staj', 'Eğitim',
        'Yazılım'
    ]
}

df = pd.DataFrame(data)

# --- 2. MODEL EĞİTİMİ (Training) ---
# Metinleri sayısal vektörlere dönüştürür
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(df['soru'])

# Basit bir destek vektör makinesi sınıflandırıcısı eğitilir
model = LinearSVC()
model.fit(X_vectorized, df['niyet'])

# --- 3. CEVAP HAVUZU (CV'den çıkarılan bilgiler) ---
CEVAPLAR = {
    'PLC': "Vanderlande stajında **Siemens PLC (TIA Portal)** kullanarak sistem izleme ve temel müdahaleler yaptım. Ayrıca **SCADA ve HMI** programlama tecrübem var.",
    'Yazılım': "Python, C/C++ ve MS SQL gibi dillerde iyi seviyede yetkinliğe sahibim. Otomasyon alanındaki güçlü yönlerim arasında **Görüntü İşleme ve ROS2** tecrübesi yer almaktadır.",
    'Staj': "Neocom'da **Zayıf Akım Sistemleri** (Kamera/Yangın/Anons) ve Vanderlande'da **Lojistik Otomasyon sistemlerinde** saha operasyonlarına destek verdim.",
    'Eğitim': "Kocaeli Üniversitesi **Mekatronik Mühendisliği** (%30 İngilizce) bölümünden 2025 yılında mezunum."
}

def niyet_siniflandir_ve_cevapla(soru):
    """Gelen soruyu sınıflandırır ve ilgili CV cevabını döndürür."""
    # Soruyu vektörleştir
    soru_vectorized = vectorizer.transform([soru])
    
    # Niyeti tahmin et
    tahmin_edilen_niyet = model.predict(soru_vectorized)[0]
    
    # Tahmin edilen niyete göre cevap ver
    return tahmin_edilen_niyet, CEVAPLAR.get(tahmin_edilen_niyet, "Ne yazık ki bu konudaki bilgiyi CV'mden tam olarak çıkaramadım. Lütfen farklı bir açıdan sorun.")

# --- 4. STREAMLIT ARAYÜZÜ ---
st.set_page_config(page_title="Yahya Osman Tamdoğan CV Chatbot", layout="wide")

# Kenar çubuğu (Sidebar)
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Robot_icon.svg/1024px-Robot_icon.svg.png", width=100)
    st.header("🤖 CV Asistanı Hakkında")
    st.info(
        "Bu Chatbot, Yahya Osman Tamdoğan'ın özgeçmişini temel alarak geliştirilmiş bir prototiptir. "
        "Sorularınızı **PLC, Yazılım, Staj veya Eğitim** niyetlerinden birine göre sınıflandırarak yanıtlar."
    )
    st.markdown("---")
    
    # LinkedIn Linki (Opsiyonel: Kendi LinkedIn adresinizi ekleyebilirsiniz)
    st.subheader("Hızlı Bağlantılar")
    st.markdown(f"**LinkedIn:** [Yahya Osman Tamdoğan LinkedIn Bağlantısı](https://www.linkedin.com/in/yahyaosmantamdogan)") # Lütfen bu linki kontrol edin
    
    # Sohbeti Temizle Butonu
    st.markdown("---")
    if st.button("Sohbeti Temizle", help="Sohbet geçmişini siler ve sıfırdan başlatır."):
        st.session_state.mesajlar = []
        st.rerun() # Uygulamayı yeniden yükler

st.title("👨‍💻 Yahya Osman Tamdoğan CV Asistanı")
st.markdown("Mekatronik Mühendisi Yahya Osman Tamdoğan'ın yetkinlikleri hakkında soru sormaya başlayın:")

# Mesaj geçmişini tutma (Session State)
if "mesajlar" not in st.session_state:
    st.session_state.mesajlar = []

# Daha önceki mesajları gösterme
for gonderici, mesaj, niyet in st.session_state.mesajlar:
    st.chat_message(gonderici).write(mesaj)
    if gonderici == "assistant":
        # Niyeti ayrı bir başlık altında göstererek profesyonel bir görünüm sağlar
        st.caption(f"**Tahmin Edilen Niyet:** :blue[{niyet}]")

# Kullanıcı girişi ve cevaplama döngüsü
if prompt := st.chat_input("Örneğin: 'PLC tecrüben ne kadar?' veya 'Hangi yazılım dillerini biliyorsun?'"):
    
    # 1. Kullanıcı mesajı
    st.session_state.mesajlar.append(("user", prompt, None))
    st.chat_message("user").write(prompt)
    
    # 2. Chatbot cevabı
    niyet, cevap = niyet_siniflandir_ve_cevapla(prompt)
    
    # 3. Cevabı kaydet ve göster
    st.session_state.mesajlar.append(("assistant", cevap, niyet))
    
    st.chat_message("assistant").write(cevap)
    st.caption(f"**Tahmin Edilen Niyet:** :blue[{niyet}]")

# Alt Bilgi (Footer) - Uygulamanın sonuna yerleştirilir
st.markdown("---")
st.markdown("<sub>*Bu, CV'deki bilgilere dayanarak oluşturulmuş yapay zeka prototipidir ve **Streamlit Cloud** üzerinde yayınlanmıştır.*</sub>", unsafe_allow_html=True)
