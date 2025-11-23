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
            "Hangi bölümden mezunsun?", "Akademik geçmişin nedir?", "Mezuniyet yılın ne?",
            
            # İletişim & Kişisel Bilgiler (15 örnek)
            "Sana nasıl ulaşabilirim?", "İletişim bilgilerin neler?", "Mail adresin ne?",
            "Telefon numaran var mı?", "Nerede yaşıyorsun?", "LinkedIn profilin var mı?",
            "Seninle nasıl iletişime geçebilirim?", "İletişim bilgilerini verir misin?",
            "Yaşın kaç?", "Doğum tarihin ne?", "Medeni durumun nedir?",
            "Askerlik durumun ne?", "Ehliyet var mı?", "Hangi dilleri konuşuyorsun?",
            "İngilizce seviyen nedir?"
        ],
        'niyet': (
            ['PLC'] * 15 + 
            ['Yazılım'] * 25 + 
            ['Staj'] * 12 + 
            ['Eğitim'] * 15 +
            ['İletişim'] * 15
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
   • Python (İyi Seviye) - Veri analizi, otomasyon, görüntü işleme
   • C/C++ (İyi Seviye) - Gömülü sistemler, algoritma geliştirme
   • SQL (MS SQL - İyi Seviye) - Veri tabanı yönetimi ve sorgulama

📌 **CAD & Tasarım Yazılımları:**
   • SolidWorks (İyi) - Mekanik tasarım ve montaj
   • AutoCAD (İyi) - Teknik çizim ve 2D tasarım
   • E-Plan (Temel) - Elektrik şema tasarımı
   • MATLAB/Simulink (İyi) - Simülasyon ve analiz
   • Ofis Programları (İyi) - MS Office Suite

📌 **Robot & Otomasyon:**
   • ROS2 (İyi Seviye) - Robot Operating System 2
   • Görüntü İşleme (İyi Seviye) - OpenCV, Computer Vision
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

📌 **Vanderlande Industries B.V. (Stajyer)**
   📍 İstanbul Havalimanı, Lojistik/Otomasyon
   📅 Ağustos 2025 - Eylül 2025
   
   • Bagaj taşıma ve lojistik otomasyon sistemlerinin saha operasyonlarına destek
   • Siemens PLC (TIA Portal) kullanarak sistem izleme, hata tespiti ve temel müdahaleler
   • Sensörler, motor sürücüleri ve konveyör hatlarının kontrolü üzerine uygulamalı deneyim
   • Otomasyon ekibiyle birlikte arıza giderme, bakım ve sistem entegrasyonu çalışmaları
   • SCADA ve HMI programlama deneyimi

📌 **Neocom İletişim Teknolojleri A.Ş. (Stajyer)**
   📍 Kıbrıs Ercan Havalimanı – Zayıf Akım Sistemleri
   📅 Haziran 2023 - Eylül 2023
   
   • Kamera sistemlerinin kurulumu, IP ataması, devreye alınması ve test edilmesi
   • Yangın panelleri kurulumu, dedektör adresleme ve senaryo testleri
   • Acil anons sistemlerinin devreye alınması, arıza tespiti ve giderilmesi
   • Proje planlarına uygun saha uygulamaları, kablolama ve sistem entegrasyonu
   • Yapılan işlerin raporlanıp bildirilmesi

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
   • **Dönem:** 2021 - 2025
   • **Durum:** Mezun
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
    },
    'İletişim': {
        'kisa': "E-posta: yahyaosman696@gmail.com | Telefon: 0506 115 68 45",
        'detayli': """
**📞 İletişim ve Kişisel Bilgilerim:**

📌 **İletişim Bilgileri:**
   • **E-posta:** yahyaosman696@gmail.com
   • **Telefon:** 0506 115 68 45
   • **Konum:** İstanbul / Beşiktaş
   • **LinkedIn:** [linkedin.com/in/yahyaosmantamdogan](https://www.linkedin.com/in/yahyaosmantamdogan)

📌 **Kişisel Bilgiler:**
   • **Ad-Soyad:** Yahya Osman Tamdoğan
   • **Doğum Tarihi:** 19.08.2003 (21 yaşında)
   • **Medeni Durum:** Bekar
   • **Askerlik Durumu:** 2 yıl tecilli
   • **Sürücü Belgesi:** B sınıfı

📌 **Yabancı Dil:**
   • **İngilizce:** B2 Seviyesi (Orta-İleri)
   
💼 Profesyonel işbirlikleri ve kariyer fırsatları için benimle iletişime geçmekten çekinmeyin!
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
    ],
    'İletişim': [
        "Sana nasıl ulaşabilirim?",
        "İletişim bilgilerin neler?",
        "İngilizce seviyen nedir?"
    ]
}

# ==================== MODEL EĞİTİMİ ====================
@st.cache_resource
def train_model():
    """ML modelini eğitir ve önbelleğe alır"""
    df = load_training_data()
    
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
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
    """Gelişmiş niyet sınıflandırma"""
    soru_vectorized = vectorizer.transform([soru])
    tahmin = model.predict(soru_vectorized)[0]
    
    decision_scores = model.decision_function(soru_vectorized)[0]
    max_score = np.max(decision_scores)
    confidence = 1 / (1 + np.exp(-max_score))
    
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
        'Eğitim': '#6B21A8',
        'İletişim': '#9F1239'
    }
    return colors.get(niyet, '#64748B')

# ==================== ÖZEL CSS STİLLERİ ====================
def apply_custom_css():
    st.markdown("""
    <style>
        /* Tema uyumlu arka plan renkleri */
        [data-testid="stSidebar"] {
            background-color: var(--background-color);
            border-right: 1px solid var(--border-color);
        }
        
        /* Light mode için */
        @media (prefers-color-scheme: light) {
            [data-testid="stSidebar"] {
                background-color: #F8FAFC;
                border-right: 1px solid #E2E8F0;
            }
        }
        
        /* Dark mode için */
        @media (prefers-color-scheme: dark) {
            [data-testid="stSidebar"] {
                background-color: #1E293B;
                border-right: 1px solid #334155;
            }
        }
        
        /* Ana başlık stili */
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
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
        
        /* Niyet badge'leri */
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
        .intent-iletisim { background-color: #FCE7F3; color: #9F1239; }
        
        /* Dark mode için badge renkleri */
        @media (prefers-color-scheme: dark) {
            .intent-plc { background-color: #1E3A8A; color: #BFDBFE; }
            .intent-yazilim { background-color: #064E3B; color: #A7F3D0; }
            .intent-staj { background-color: #78350F; color: #FEF3C7; }
            .intent-egitim { background-color: #581C87; color: #E9D5FF; }
            .intent-iletisim { background-color: #831843; color: #FCE7F3; }
        }
        
        /* Buton stili */
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        
        /* Chat input stili */
        .stChatInput>div {
            border-radius: 12px;
        }
        
        /* Scrollbar stilini iyileştir */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #CBD5E1;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #94A3B8;
        }
        
        /* Dark mode için scrollbar */
        @media (prefers-color-scheme: dark) {
            ::-webkit-scrollbar-thumb {
                background: #475569;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: #64748B;
            }
        }
    </style>
    """, unsafe_allow_html=True)

# ==================== ANA UYGULAMA ====================
def main():
    # CSS uygula
    apply_custom_css()
    
    # Model yükleme
    vectorizer, model, X_train, df = train_model()
    
    # ==================== SIDEBAR ====================
    with st.sidebar:
        # Profil resmi - tema uyumlu
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        width: 120px; height: 120px; border-radius: 60px; 
                        margin: 0 auto; display: flex; align-items: center; 
                        justify-content: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <span style='font-size: 50px;'>👨‍💻</span>
            </div>
            <h3 style='margin-top: 15px; margin-bottom: 5px;'>Yahya Osman Tamdoğan</h3>
            <p style='color: #64748B; font-size: 0.9rem;'>Mekatronik Mühendisi</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📋 CV Asistanı Hakkında")
        
        st.info(
            "Bu chatbot, **Yahya Osman Tamdoğan**'ın özgeçmişini yapay zeka "
            "ile analiz ederek sorularınızı yanıtlar. Sorularınız otomatik olarak "
            "kategorize edilir: **PLC, Yazılım, Staj, Eğitim, İletişim**"
        )
        
        # İstatistikler
        st.markdown("---")
        st.markdown("### 📊 Model İstatistikleri")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Toplam Eğitim Verisi", f"{len(df)} soru")
        with col2:
            st.metric("Niyet Kategorisi", "5 adet")
        
        # İletişim Bilgileri
        st.markdown("---")
        st.markdown("### 📞 İletişim Bilgileri")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown("📧")
        with col2:
            st.markdown("[yahyaosman696@gmail.com](mailto:yahyaosman696@gmail.com)")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown("📱")
        with col2:
            st.markdown("0506 115 68 45")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown("📍")
        with col2:
            st.markdown("İstanbul / Beşiktaş")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown("💼")
        with col2:
            LINKEDIN_URL = "https://www.linkedin.com/in/yahyaosmantamdogan"
            st.markdown(f"[LinkedIn Profilim]({LINKEDIN_URL})")
        
        # Yabancı Dil
        st.markdown("---")
        st.markdown("### 🌍 Yabancı Dil")
        st.markdown("🇬🇧 **İngilizce:** B2 (Orta-İleri)")
        
        # Kişisel Bilgiler
        st.markdown("---")
        st.markdown("### 👤 Kişisel Bilgiler")
        st.markdown("""
        - **Doğum Tarihi:** 19.08.2003
        - **Medeni Durum:** Bekar
        - **Askerlik:** 2 yıl tecilli
        - **Sürücü Belgesi:** B sınıfı
        """)
        
        # Örnek Sorular
        st.markdown("---")
        st.markdown("### 💡 Örnek Sorular")
        
        kategori = st.selectbox(
            "Kategori seçin:",
            ['PLC', 'Yazılım', 'Staj', 'Eğitim', 'İletişim']
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
                'niyet_dagilim': {'PLC': 0, 'Yazılım': 0, 'Staj': 0, 'Eğitim': 0, 'İletişim': 0}
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
            'niyet_dagilim': {'PLC': 0, 'Yazılım': 0, 'Staj': 0, 'Eğitim': 0, 'İletişim': 0}
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
- 📞 **İletişim ve Kişisel** bilgiler

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
