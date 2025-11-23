import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from datetime import datetime

# ==================== SAYFA AYARLARI ====================
st.set_page_config(
    page_title="Yahya Osman Tamdoğan - CV Asistanı",
    page_icon="👨‍💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== ÖZEL CSS STİLLERİ ====================
st.markdown("""
<style>
    /* Ana başlık stili */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Alt başlık stili */
    .sub-header {
        font-size: 1.2rem;
        color: #64748B;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Niyet badge'i */
    .intent-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 8px;
    }
    
    .intent-plc { background-color: #DBEAFE; color: #1E40AF; }
    .intent-yazilim { background-color: #D1FAE5; color: #065F46; }
    .intent-staj { background-color: #FEF3C7; color: #92400E; }
    .intent-egitim { background-color: #E9D5FF; color: #6B21A8; }
    
    /* Sidebar iyileştirmeleri */
    [data-testid="stSidebar"] {
        background-color: #F8FAFC;
    }
    
    /* Buton stili */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Chat input stili */
    .stChatInput>div {
        border-radius: 12px;
    }
    
    /* Metrik kartları */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ==================== GÜÇLENDİRİLMİŞ VERİ SETİ ====================
@st.cache_data
def load_training_data():
    """Eğitim verisini yükler ve önbelleğe alır"""
    data = {
        'soru': [
            # PLC & Otomasyon (15 örnek)
            "PLC deneyimin var mı?", "TIA Portal biliyor musun?", "Siemens otomasyon tecrüben nedir?", 
            "HMI programlamayı biliyor musun?", "Otomasyon becerilerin neler?", "SCADA sistemleri bilgin var mı?",
            "Endüstriyel otomasyon tecrüben nedir?", "PLC programlama yapabiliyor musun?",
            "Siemens sistemlerinde çalıştın mı?", "Kontrol sistemleri hakkında ne biliyorsun?",
            "TIA Portal ile proje yaptın mı?", "Otomasyon mühendisliği bilgin nedir?",
            "PLC ladder logic biliyor musun?", "HMI arayüz tasarlayabiliyor musun?",
            "Endüstriyel kontrol sistemleri deneyimin var mı?",
            
            # Yazılım & Programlama (25 örnek)
            "Python'da iyi misin?", "Hangi yazılım dillerini biliyorsun?", "Görüntü İşleme projen var mı?", 
            "ROS2 ile çalıştın mı?", "SQL biliyor musun?", "Kodlama yeteneklerin nelerdir?", 
            "C++ bilgine ne dersin?", "Hangi dilleri biliyorsun?", 
            "Hangi programlama dillerinde yetkinsin?", "Programlama tecrüben nedir?",
            "Bitirme projen neydi?", "Otonom araç projesini anlatır mısın?", 
            "SolidWorks bilgin nedir?", "Sensör füzyonu kullandın mı?", 
            "Hangi CAD programlarını biliyorsun?", "Tasarım yazılımların nelerdir?",
            "MATLAB kullanıyor musun?", "AutoCAD bilgin nedir?", "Yazılım geliştirme deneyimin var mı?",
            "Python projelerin neler?", "C programlama biliyor musun?", "Veri tabanı yönetimi yapabiliyor musun?",
            "Görüntü işleme algoritmaları bilgin var mı?", "ROS deneyimin nedir?",
            "Hangi IDE'leri kullanıyorsun?",
            
            # Staj & İş Deneyimi (12 örnek)
            "Vanderlande stajında ne yaptın?", "Neocom'da ne gibi işler yaptın?", 
            "Staj tecrübelerinden bahseder misin?", "Lojistik sistemlerde çalıştın mı?", 
            "Neocom'daki görevin neydi?", "İş deneyimin nedir?", "Hangi şirketlerde çalıştın?",
            "Stajlarını anlat", "Vanderlande'da ne iş yaptın?", "İş tecrüben var mı?",
            "Zayıf akım sistemleri deneyimin var mı?", "Saha çalışması yaptın mı?",
            
            # Eğitim (15 örnek)
            "Eğitim bilgilerini alabilir miyim?", "Hangi üniversiteden mezunsun?", 
            "Mekatronik bilgin nedir?", "Diploman ne?", "Nerede okudun?", 
            "Üniversitenin adı ne?", "Lisans derecen nedir?", "Okulun hakkında bilgi ver.",
            "Mezun olduğun okul neresi?", "Eğitim durumun nedir?", 
            "Üniversite eğitimin hakkında konuşalım.", "Okulun nerede?",
            "Hangi bölümden mezunsun?", "Akademik geçmişin nedir?", "Mezuniyet yılın ne?"
        ],
        'niyet': (
            ['PLC'] * 15 + 
            ['Yazılım'] * 25 + 
            ['Staj'] * 12 + 
            ['Eğitim'] * 15
        )
    }
    return pd.DataFrame(data)

# ==================== DETAYLI CEVAP HAVUZU ====================
CEVAPLAR = {
    'PLC': {
        'kisa': "Siemens PLC ve TIA Portal deneyimim var.",
        'detayli': """
**🔧 Otomasyon Kontrol Sistemleri Yetkinliklerim:**

📌 **PLC Programlama:**
   • Siemens PLC sistemleri ile proje deneyimi
   • TIA Portal (Totally Integrated Automation) kullanımı
   • Ladder Logic ve Function Block programlama

📌 **HMI & SCADA:**
   • HMI arayüz tasarımı ve programlama
   • SCADA sistemleri ile sistem izleme
   • Operatör paneli konfigürasyonu

📌 **Pratik Deneyim:**
   • Vanderlande stajımda büyük ölçekli lojistik otomasyon sistemlerinde çalıştım
   • Gerçek zamanlı sistem izleme ve arıza müdahalesi deneyimi
   • Endüstriyel kontrol sistemleri entegrasyonu
        """
    },
    'Yazılım': {
        'kisa': "Python, C/C++, SQL ve CAD programlarında yetkinim. ROS2 ve görüntü işleme deneyimim var.",
        'detayli': """
**💻 Yazılım & Tasarım Yetkinliklerim:**

📌 **Programlama Dilleri:**
   • Python (İleri Seviye) - Veri analizi, otomasyon, görüntü işleme
   • C/C++ (İyi Seviye) - Gömülü sistemler, algoritma geliştirme
   • SQL (MS SQL) - Veri tabanı yönetimi ve sorgulama

📌 **CAD & Tasarım Yazılımları:**
   • SolidWorks - Mekanik tasarım ve montaj
   • AutoCAD - Teknik çizim ve 2D tasarım
   • E-Plan - Elektrik şema tasarımı (temel seviye)
   • MATLAB/Simulink - Simülasyon ve analiz

📌 **Robot & Otomasyon:**
   • ROS2 (Robot Operating System 2) - Robot yazılım geliştirme
   • Görüntü İşleme - OpenCV, Computer Vision algoritmaları
   • Sensör Füzyonu - Çoklu sensör verisi entegrasyonu

📌 **Öne Çıkan Proje:**
   🚗 **Otonom Araç Bitirme Projesi:**
      - Görüntü işleme ve sensör füzyonu teknikleri kullanımı
      - Gerçek zamanlı veri işleme ve karar verme algoritmaları
      - ROS2 tabanlı yazılım mimarisi
        """
    },
    'Staj': {
        'kisa': "Neocom ve Vanderlande'da staj yaptım.",
        'detayli': """
**🏢 İş Deneyimim:**

📌 **Vanderlande Stajı:**
   • Büyük ölçekli lojistik otomasyon sistemleri
   • Siemens PLC ve TIA Portal ile sistem programlama
   • Konveyör sistemleri ve malzeme taşıma otomasyonu
   • Saha operasyonları ve bakım desteği
   • SCADA sistemleri ile gerçek zamanlı izleme

📌 **Neocom Stajı:**
   • Zayıf akım sistemleri kurulumu ve entegrasyonu
   • Güvenlik kamera sistemleri (CCTV)
   • Yangın algılama ve anons sistemleri
   • Yapısal kablolama ve sistem testleri
   • Saha çalışması ve müşteri koordinasyonu

**🎯 Kazanılan Deneyimler:**
   ✓ Endüstriyel otomasyon sistemlerinde pratik deneyim
   ✓ Ekip çalışması ve proje yönetimi
   ✓ Problem çözme ve arıza giderme becerileri
   ✓ Gerçek dünya mühendislik uygulamaları
        """
    },
    'Eğitim': {
        'kisa': "Kocaeli Üniversitesi Mekatronik Mühendisliği mezunuyum (2025).",
        'detayli': """
**🎓 Akademik Geçmişim:**

📌 **Lisans Eğitimi:**
   • **Üniversite:** Kocaeli Üniversitesi
   • **Bölüm:** Mekatronik Mühendisliği (%30 İngilizce)
   • **Mezuniyet Yılı:** 2025
   • **Konum:** Kocaeli, Türkiye

📌 **Mekatronik Mühendisliği Uzmanlık Alanları:**
   ✓ Mekanik Sistemler - Tasarım ve analiz
   ✓ Elektronik & Kontrol - Devre tasarımı, PLC
   ✓ Yazılım & Programlama - Algoritma geliştirme
   ✓ Otomasyon Sistemleri - Endüstriyel uygulamalar
   ✓ Robot Teknolojileri - ROS, kinematik, sensörler

📌 **Disiplinler Arası Yetkinlik:**
   Mekatronik mühendisliği, makine, elektrik-elektronik ve bilgisayar 
   mühendisliğinin kesişim noktasında yer alır. Bu interdisipliner eğitim 
   sayesinde karmaşık sistemleri bütünsel olarak tasarlayıp geliştirebiliyorum.
        """
    }
}

# ==================== ÖNERİLEN SORULAR ====================
ORNEK_SORULAR = {
    'PLC': [
        "TIA Portal deneyimin var mı?",
        "SCADA sistemleri hakkında ne biliyorsun?",
        "PLC programlama yapabiliyor musun?"
    ],
    'Yazılım': [
        "Python projelerini anlatır mısın?",
        "Hangi CAD programlarını kullanıyorsun?",
        "Otonom araç projen nasıl gelişti?",
        "ROS2 deneyimin nedir?"
    ],
    'Staj': [
        "Vanderlande stajında neler öğrendin?",
        "Neocom'daki görevlerin nelerdi?",
        "Saha deneyimin var mı?"
    ],
    'Eğitim': [
        "Hangi üniversiteden mezunsun?",
        "Mekatronik mühendisliği nedir?",
        "Akademik geçmişin nasıl?"
    ]
}

# ==================== MODEL EĞİTİMİ ====================
@st.cache_resource
def train_model():
    """ML modelini eğitir ve önbelleğe alır"""
    df = load_training_data()
    
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),  # Unigram ve bigram kullan
        max_features=500,
        min_df=1
    )
    
    X_vectorized = vectorizer.fit_transform(df['soru'])
    
    model = LinearSVC(
        C=1.0,
        max_iter=2000,
        random_state=42
    )
    model.fit(X_vectorized, df['niyet'])
    
    return vectorizer, model, X_vectorized, df

# ==================== YARDIMCI FONKSİYONLAR ====================
def niyet_siniflandir(soru, vectorizer, model, X_train, df):
    """
    Gelişmiş niyet sınıflandırma: 
    - Güven skoru hesaplama
    - Benzerlik analizi
    - Alternatif öneriler
    """
    soru_vectorized = vectorizer.transform([soru])
    tahmin = model.predict(soru_vectorized)[0]
    
    # Karar fonksiyonu skorları (güven seviyesi için)
    decision_scores = model.decision_function(soru_vectorized)[0]
    
    # En yüksek skoru bul
    max_score = np.max(decision_scores)
    confidence = 1 / (1 + np.exp(-max_score))  # Sigmoid ile normalize et
    
    # Eğitim verileriyle benzerlik
    similarities = cosine_similarity(soru_vectorized, X_train)[0]
    max_similarity = np.max(similarities)
    most_similar_idx = np.argmax(similarities)
    
    return {
        'niyet': tahmin,
        'guven': confidence,
        'benzerlik': max_similarity,
        'en_benzer_soru': df.iloc[most_similar_idx]['soru']
    }

def format_cevap(niyet, detayli=True):
    """Cevabı formatlar"""
    cevap_dict = CEVAPLAR.get(niyet, {})
    return cevap_dict.get('detayli' if detayli else 'kisa', 
                          "Bu konuda bilgi bulunamadı. Lütfen başka bir soru sorun.")

def get_intent_color(niyet):
    """Niyet için renk kodu döndürür"""
    colors = {
        'PLC': '#1E40AF',
        'Yazılım': '#065F46',
        'Staj': '#92400E',
        'Eğitim': '#6B21A8'
    }
    return colors.get(niyet, '#64748B')

# ==================== ANA UYGULAMA ====================
def main():
    # Model yükleme
    vectorizer, model, X_train, df = train_model()
    
    # ==================== SIDEBAR ====================
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
        st.markdown("### 👨‍💻 CV Asistanı Hakkında")
        
        st.info(
            "Bu chatbot, **Yahya Osman Tamdoğan**'ın özgeçmişini yapay zeka "
            "ile analiz ederek sorularınızı yanıtlar. Sorularınız otomatik olarak "
            "kategorize edilir: **PLC, Yazılım, Staj, Eğitim**"
        )
        
        # İstatistikler
        st.markdown("---")
        st.markdown("### 📊 Model İstatistikleri")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Toplam Eğitim Verisi", f"{len(df)} soru")
        with col2:
            st.metric("Niyet Kategorisi", "4 adet")
        
        # Hızlı Linkler
        st.markdown("---")
        st.markdown("### 🔗 Hızlı Bağlantılar")
        LINKEDIN_URL = "https://www.linkedin.com/in/yahyaosmantamdogan"
        st.markdown(f"[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)]({LINKEDIN_URL})")
        
        # Örnek Sorular
        st.markdown("---")
        st.markdown("### 💡 Örnek Sorular")
        
        kategori = st.selectbox(
            "Kategori seçin:",
            ['PLC', 'Yazılım', 'Staj', 'Eğitim']
        )
        
        for soru in ORNEK_SORULAR[kategori]:
            if st.button(soru, key=f"btn_{soru}", use_container_width=True):
                st.session_state.ornek_soru = soru
        
        # Sohbeti temizle
        st.markdown("---")
        if st.button("🗑️ Sohbeti Temizle", type="secondary", use_container_width=True):
            st.session_state.mesajlar = []
            st.session_state.istatistikler = {
                'toplam_soru': 0,
                'niyet_dagilim': {'PLC': 0, 'Yazılım': 0, 'Staj': 0, 'Eğitim': 0}
            }
            st.rerun()
        
        # Footer
        st.markdown("---")
        st.caption(f"Son güncelleme: {datetime.now().strftime('%d.%m.%Y')}")
        st.caption("Streamlit + scikit-learn ile geliştirilmiştir")
    
    # ==================== ANA İÇERİK ====================
    st.markdown("<h1 class='main-header'>👨‍💻 Yahya Osman Tamdoğan</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Mekatronik Mühendisi | CV Asistanı Chatbot</p>", unsafe_allow_html=True)
    
    # Session state başlatma
    if "mesajlar" not in st.session_state:
        st.session_state.mesajlar = []
    
    if "istatistikler" not in st.session_state:
        st.session_state.istatistikler = {
            'toplam_soru': 0,
            'niyet_dagilim': {'PLC': 0, 'Yazılım': 0, 'Staj': 0, 'Eğitim': 0}
        }
    
    # Hoş geldin mesajı
    if len(st.session_state.mesajlar) == 0:
        with st.chat_message("assistant"):
            st.markdown("""
👋 **Merhaba! Yahya Osman Tamdoğan'ın CV Asistanına hoş geldiniz.**

Aşağıdaki konularda bana soru sorabilirsiniz:
- 🔧 **PLC ve Otomasyon** sistemleri
- 💻 **Yazılım ve Programlama** becerileri  
- 🏢 **Staj ve İş** deneyimleri
- 🎓 **Eğitim** geçmişi

Soldaki menüden örnek sorulara göz atabilir veya doğrudan soru sorabilirsiniz!
            """)
    
    # Önceki mesajları göster
    for msg in st.session_state.mesajlar:
        gonderici = msg['role']
        icerik = msg['content']
        
        with st.chat_message(gonderici):
            st.markdown(icerik)
            
            if gonderici == "assistant" and 'metadata' in msg:
                metadata = msg['metadata']
                niyet = metadata['niyet']
                guven = metadata.get('guven', 0)
                
                # Niyet badge'i
                badge_class = f"intent-{niyet.lower()}"
                st.markdown(
                    f"<span class='intent-badge {badge_class}'>🏷️ {niyet}</span> "
                    f"<span style='color: #64748B; font-size: 0.85rem;'>Güven: {guven:.0%}</span>",
                    unsafe_allow_html=True
                )
    
    # Örnek soru seçildiyse
    if 'ornek_soru' in st.session_state:
        prompt = st.session_state.ornek_soru
        del st.session_state.ornek_soru
    else:
        prompt = st.chat_input("Bir soru sorun... (örn: 'Python bilgin nedir?' veya 'Staj deneyimlerini anlatır mısın?')")
    
    # Kullanıcı sorusu işleme
    if prompt:
        # Kullanıcı mesajını göster
        st.session_state.mesajlar.append({
            'role': 'user',
            'content': prompt
        })
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Chatbot cevabı
        with st.chat_message("assistant"):
            with st.spinner("Düşünüyorum..."):
                # Niyet analizi
                sonuc = niyet_siniflandir(prompt, vectorizer, model, X_train, df)
                niyet = sonuc['niyet']
                guven = sonuc['guven']
                
                # Cevap oluştur
                cevap = format_cevap(niyet, detayli=True)
                
                # Düşük güven durumu
                if guven < 0.5:
                    cevap = f"⚠️ Bu soruyu tam olarak anlayamadım (Güven: {guven:.0%}). " \
                            f"Belki şunu sormak istediniz: *\"{sonuc['en_benzer_soru']}\"*?\n\n{cevap}"
                
                st.markdown(cevap)
                
                # Metadata
                badge_class = f"intent-{niyet.lower()}"
                st.markdown(
                    f"<span class='intent-badge {badge_class}'>🏷️ {niyet}</span> "
                    f"<span style='color: #64748B; font-size: 0.85rem;'>Güven: {guven:.0%}</span>",
                    unsafe_allow_html=True
                )
                
                # Mesajı kaydet
                st.session_state.mesajlar.append({
                    'role': 'assistant',
                    'content': cevap,
                    'metadata': {
                        'niyet': niyet,
                        'guven': guven,
                        'benzerlik': sonuc['benzerlik']
                    }
                })
                
                # İstatistikleri güncelle
                st.session_state.istatistikler['toplam_soru'] += 1
                st.session_state.istatistikler['niyet_dagilim'][niyet] += 1

if __name__ == "__main__":
    main()
