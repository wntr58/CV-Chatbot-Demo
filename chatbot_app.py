import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import pandas as pd
import numpy as np

# --- 1. GÜÇLENDİRİLMİŞ VERİ KÜMESİ (Niyet Tanıma İçin) ---
# Model, bu soru-cevap eşleşmelerini kullanarak niyetleri öğrenir.
data = {
    'soru': [
        # PLC
        "PLC deneyimin var mı?", "TIA Portal biliyor musun?", "Siemens otomasyon tecrüben nedir?", 
        "HMI programlamayı biliyor musun?", "Otomasyon becerilerin neler?", 
        # Yazılım (Daha fazla spesifik kodlama terimi eklendi)
        "Python'da iyi misin?", "Hangi yazılım dillerini biliyorsun?", "Görüntü İşleme projen var mı?", 
        "ROS2 ile çalıştın mı?", "Sql biliyor musun?", "Kodlama yeteneklerin nelerdir?", "C++ bilgine ne dersin?", 
        "Hangi dilleri biliyorsun?",
        # Staj
        "Vanderlande stajında ne yaptın?", "Neocom'da ne gibi işler yaptın?", "Staj tecrübelerinden bahseder misin?", 
        "Lojistik sistemlerde çalıştın mı?", "Neocom'daki görevin neydi?",
        # Eğitim (Üniversite ve okul odaklı sorular eklendi)
        "Eğitim bilgilerini alabilir miyim?", "Hangi üniversiteden mezunsun?", "Mekatronik bilgin nedir?",
        "Diploman ne?", "Nerede okudun?", "Üniversitenin adı ne?", "Lisans derecen nedir?", "Okulun hakkında bilgi ver.",
        "Mezun olduğun okul neresi?"
    ],
    'niyet': [
        # PLC
        'PLC', 'PLC', 'PLC', 'PLC', 'PLC', 
        # Yazılım
        'Yazılım', 'Yazılım', 'Yazılım', 'Yazılım', 'Yazılım', 'Yazılım', 'Yazılım', 'Yazılım', 
        # Staj
        'Staj', 'Staj', 'Staj', 'Staj', 'Staj', 
        # Eğitim
        'Eğitim', 'Eğitim', 'Eğitim', 'Eğitim', 'Eğitim', 'Eğitim', 'Eğitim', 'Eğitim', 'Eğitim'
    ]
}

df = pd.DataFrame(data)

# --- 2. MODEL EĞİTİMİ (Training) ---
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(df['soru'])
model = LinearSVC()
model.fit(X_vectorized, df['niyet'])

# --- 3. KURUMSAL CEVAP HAVUZU ---
CEVAPLAR = {
    'PLC': "**Otomasyon Kontrol Sistemleri:** Vanderlande stajımda **Siemens PLC (TIA Portal)** kullanarak sistem izleme ve temel müdahaleler yaptım. Ayrıca **SCADA ve HMI** arayüz programlama prensiplerini uyguladım.",
    'Yazılım': "**Geliştirme Yetkinlikleri:** Python, C/C++ ve MS SQL gibi dillerde iyi seviyede yetkinliğe sahibim. Otomasyon projelerindeki güçlü yönlerim arasında özellikle **Görüntü İşleme** ve **ROS2 (Robot İşletim Sistemi)** tecrübesi yer almaktadır.",
    'Staj': "**Saha Deneyimi:** Neocom'da **Zayıf Akım Sistemleri** (Kamera/Yangın/Anons) ve Vanderlande'da büyük ölçekli **Lojistik Otomasyon sistemlerinde** saha operasyonlarına destek vererek pratik tecrübe kazandım.",
    'Eğitim': "**Lisans Eğitimi:** Kocaeli Üniversitesi **Mekatronik Mühendisliği** (%30 İngilizce) bölümünden 2025 yılında başarıyla mezun oldum. Mühendislik temelimi bu alanda sağlamlaştırdım."
}

def niyet_siniflandir_ve_cevapla(soru):
    """Gelen soruyu sınıflandırır ve ilgili CV cevabını döndürür."""
    soru_vectorized = vectorizer.transform([soru])
    tahmin_edilen_niyet = model.predict(soru_vectorized)[0]
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
    
    st.subheader("Hızlı Bağlantılar")
    # LÜTFEN AŞAĞIDAKİ URL'Yİ KENDİ GERÇEK LINKEDIN ADRESİNİZLE DEĞİŞTİRİN
    st.markdown(f"**LinkedIn:** [Yahya Osman Tamdoğan LinkedIn Bağlantısı](https://www.linkedin.com/in/yahyaosmantamdogan)") 
    
    # Sohbeti Temizle Butonu
    st.markdown("---")
    if st.button("Sohbeti Temizle", help="Sohbet geçmişini siler ve sıfırdan başlatır."):
        st.session_state.mesajlar = []
        st.rerun()

st.title("👨‍💻 Yahya Osman Tamdoğan CV Asistanı")
st.markdown("Mekatronik Mühendisi Yahya Osman Tamdoğan'ın yetkinlikleri hakkında soru sormaya başlayın:")

# Mesaj geçmişini tutma (Session State)
if "mesajlar" not in st.session_state:
    st.session_state.mesajlar = []

# Daha önceki mesajları gösterme
for gonderici, mesaj, niyet in st.session_state.mesajlar:
    st.chat_message(gonderici).write(mesaj)
    if gonderici == "assistant":
        st.caption(f"**Tahmin Edilen Niyet:** :blue[{niyet}]")

# Kullanıcı girişi ve cevaplama döngüsü
if prompt := st.chat_input("Örneğin: 'Hangi üniversiteden mezunsun?' veya 'Görüntü işleme tecrüben var mı?'"):
    
    # 1. Kullanıcı mesajı
    st.session_state.mesajlar.append(("user", prompt, None))
    st.chat_message("user").write(prompt)
    
    # 2. Chatbot cevabı
    niyet, cevap = niyet_siniflandir_ve_cevapla(prompt)
    
    # 3. Cevabı kaydet ve göster
    st.session_state.mesajlar.append(("assistant", cevap, niyet))
    
    st.chat_message("assistant").write(cevap)
    st.caption(f"**Tahmin Edilen Niyet:** :blue[{niyet}]")

# Alt Bilgi (Footer)
st.markdown("---")
st.markdown("<sub>*Bu, CV'deki bilgilere dayanarak oluşturulmuş yapay zeka prototipidir ve **Streamlit Cloud** üzerinde yayınlanmıştır.*</sub>", unsafe_allow_html=True)
