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
# Streamlit'in caching mekanizması kullanılarak veri yüklemesi hızlandırılır.
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
            
            # Yazılım & Programlama (25 örnek) - CAD/Proje soruları dahil
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

📌 **Pratik Deneyim:**
    • Vanderlande stajımda büyük ölçekli lojistik otomasyon sistemlerinde çalıştım
    • Endüstriyel kontrol sistemleri entegrasyonu
        """
    },
    'Yazılım': {
        'kisa': "Python, C/C++, SQL, SolidWorks ve AutoCAD programlarında yetkinim. ROS2 ve görüntü işleme deneyimim var.",
        'detayli': """
**💻 Yazılım & Tasarım Yetkinliklerim:**

📌 **Programlama Dilleri:**
    • **Python** (İyi Seviye) - Veri analizi, otomasyon, görüntü işleme
    • **C/C++** (İyi Seviye) - Gömülü sistemler, algoritma geliştirme
    • **MS SQL** (İyi Seviye) - Veri tabanı yönetimi

📌 **CAD & Tasarım Yazılımları:**
    • **SolidWorks** (İyi) - Mekanik tasarım ve montaj
    • **AutoCAD** (İyi) - Teknik çizim ve 2D tasarım
    • **MATLAB/Simulink** (İyi) - Simülasyon ve analiz
    • **E-Plan** (Temel) - Elektrik şema tasarımı

📌 **Robotik & Proje:**
    • **ROS2** (İyi Seviye) - Robot Operating System 2
    • **Görüntü İşleme** (İyi Seviye) - OpenCV, Computer Vision
    • **Sensör Füzyonu** - Çoklu sensör verisi entegrasyonu

📌 **Öne Çıkan Proje:**
    🚗 **Otonom Araç Bitirme Projesi:** Görüntü işleme ve sensör füzyonu teknikleri kullanılarak ROS2 tabanlı yazılım mimarisiyle geliştirilmiştir.
        """
    },
    'Staj': {
        'kisa': "Neocom ve Vanderlande'da otomasyon ve zayıf akım sistemlerinde staj yaptım.",
        'detayli': """
**🏢 İş Deneyimim:**

📌 **Vanderlande Industries B.V. (Stajyer)**
    📍 Lojistik/Otomasyon - İstanbul Havalimanı
    • Siemens PLC (TIA Portal) kullanarak sistem izleme ve temel müdahaleler
    • Sensörler, motor sürücüleri ve konveyör hatlarının kontrolü üzerine uygulamalı deneyim

📌 **Neocom İletişim Teknolojleri A.Ş. (Stajyer)**
    📍 Zayıf Akım Sistemleri - Kıbrıs Ercan Havalimanı
    • Kamera, Yangın paneli ve Acil anons sistemlerinin kurulumu ve devreye alınması
    • Proje planlarına uygun saha uygulamaları ve sistem entegrasyonu
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

📌 **Mekatronik Mühendisliği Uzmanlık Alanları:**
    ✓ Mekanik, Elektronik, Kontrol ve Yazılım disiplinlerinin kesişim noktasında uzmanlık
    ✓ Karmaşık otomasyon sistemlerini bütünsel olarak tasarlama ve geliştirme yeteneği
        """
    },
    'İletişim': {
        'kisa': "E-posta: yahyaosman696@gmail.com | Telefon: 0506 115 68 45",
        'detayli': """
**📞 İletişim ve Kişisel Bilgilerim:**

📌 **İletişim Bilgileri:**
    • **E-posta:** yahyaosman696@gmail.com
    • **Telefon:** 0506 115 68 45
    • **LinkedIn:** [linkedin.com/in/yahyaosmantamdogan](https://www.linkedin.com/in/yahyaosmantamdogan)
    
📌 **Diğer:**
    • **Konum:** İstanbul / Beşiktaş
    • **Yabancı Dil:** İngilizce - B2 Seviyesi (Orta-İleri)
    • **Askerlik:** 2 yıl tecilli | **Sürücü Belgesi:** B sınıfı
        """
    }
}

# ==================== ÖNERİLEN SORULAR ====================
ORNEK_SORULAR = {
    'PLC': ["TIA Portal deneyimin var mı?", "SCADA sistemleri hakkında ne biliyorsun?", "Endüstriyel otomasyon tecrüben nedir?"],
    'Yazılım': ["Hangi CAD programlarını kullanıyorsun?", "Otonom araç projen nasıl gelişti?", "Sensör füzyonu kullandın mı?", "ROS2 deneyimin nedir?"],
    'Staj': ["Vanderlande stajında neler öğrendin?", "Neocom'daki görevlerin nelerdi?", "Saha deneyimin var mı?"],
    'Eğitim': ["Hangi üniversiteden mezunsun?", "Mekatronik mühendisliği nedir?", "Akademik geçmişin nasıl?"],
    'İletişim': ["Sana nasıl ulaşabilirim?", "İletişim bilgilerin neler?", "İngilizce seviyen nedir?"]
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
    
    model = LinearSVC(C=1.0, max_iter=2000, random_state=42)
    model.fit(X_vectorized, df['niyet'])
    
    return vectorizer, model, X_vectorized, df

# ==================== YARDIMCI FONKSİYONLAR ====================
def niyet_siniflandir(soru, vectorizer, model, X_train, df):
    """Gelişmiş niyet sınıflandırma"""
    soru_vectorized = vectorizer.transform([soru])
    tahmin = model.predict(soru_vectorized)[0]
    
    # Güven puanı hesaplama (confidence score)
    decision_scores = model.decision_function(soru_vectorized)[0]
    max_score = np.max(decision_scores)
    confidence = 1 / (1 + np.exp(-max_score))
    
    # En benzer soruyu bulma (düşük güven için öneri)
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

# ==================== ÖZEL CSS STİLLERİ (Hata Düzeltildi) ====================
def apply_custom_css():
    """CSS stilini Streamlit'e uygular."""
    st.markdown("""
    <style>
        /* Temel Sidebar Stilleri */
        [data-testid="stSidebar"] {
            background-color: var(--background-color);
            border-right: 1px solid var(--border-color);
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
        
        /* Niyet badge'leri - Hata alınan kısım doğru şekilde tırnak içine alındı */
        .intent-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-top: 8px;
        }
        
        /* Renkler (Light Mode) */
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
        
        /* Diğer UI iyileştirmeleri */
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
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
        # Profil Resmi & Başlık
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
        
        # İletişim Bilgileri
        st.markdown("---")
        st.markdown("### 📞 İletişim Bilgileri")
        
        LINKEDIN_URL = "https://www.linkedin.com/in/yahyaosmantamdogan" # Lütfen kontrol edin
        
        st.markdown(f"""
        - 📧 [yahyaosman696@gmail.com](mailto:yahyaosman696@gmail.com)
        - 📱 0506 115 68 45
        - 💼 [LinkedIn Profilim]({LINKEDIN_URL})
        - 📍 İstanbul / Beşiktaş
        """, unsafe_allow_html=True)
        
        # Örnek Sorular
        st.markdown("---")
        st.markdown("### 💡 Örnek Sorular")
        
        kategori = st.selectbox(
            "Kategori seçin:",
            ['Yazılım', 'PLC', 'Staj', 'Eğitim', 'İletişim']
        )
        
        for soru in ORNEK_SORULAR[kategori]:
            if st.button(soru, key=f"btn_{soru}", use_container_width=True):
                st.session_state.ornek_soru = soru
                st.rerun()
        
        # Sohbeti temizle
        st.markdown("---")
        if st.button("🗑️ Sohbeti Temizle", type="secondary", use_container_width=True):
            st.session_state.mesajlar = []
            st.rerun()
        
        # Footer
        st.markdown("---")
        st.caption(f"Son güncelleme: {datetime.now().strftime('%d.%m.%Y')}")
    
    # ==================== ANA İÇERİK ====================
    st.markdown("<h1 class='main-header'>👨‍💻 Yahya Osman Tamdoğan</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Mekatronik Mühendisi | CV Asistanı Chatbot</p>", unsafe_allow_html=True)
    
    # Session state başlatma
    if "mesajlar" not in st.session_state:
        st.session_state.mesajlar = []
    
    # Hoş geldin mesajı
    if len(st.session_state.mesajlar) == 0:
        with st.chat_message("assistant"):
            st.markdown("""
👋 **Merhaba! Yahya Osman Tamdoğan'ın CV Asistanına hoş geldiniz.**

Aşağıdaki konularda bana soru sorabilirsiniz:
- 💻 **Yazılım & Tasarım** (CAD, Python, Otonom Araç)
- 🔧 **PLC ve Otomasyon** sistemleri
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
                niyet = msg['metadata']['niyet']
                guven = msg['metadata'].get('guven', 0)
                
                # Niyet badge'i
                badge_class = f"intent-{niyet.lower()}"
                st.markdown(
                    f"<span class='intent-badge {badge_class}'>🏷️ {niyet}</span> "
                    f"<span style='color: #64748B; font-size: 0.85rem;'>Güven: {guven:.0%}</span>",
                    unsafe_allow_html=True
                )
    
    # Örnek soru seçildiyse veya yeni soru girildiyse
    prompt = None
    if 'ornek_soru' in st.session_state:
        prompt = st.session_state.ornek_soru
        del st.session_state.ornek_soru
    else:
        prompt = st.chat_input("Bir soru sorun... (örn: 'Hangi CAD programlarını biliyorsun?' veya 'Bitirme projen neydi?')")
    
    # Kullanıcı sorusu işleme
    if prompt:
        # Kullanıcı mesajını göster
        st.session_state.mesajlar.append({'role': 'user', 'content': prompt})
        
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
                
                # Düşük güven durumu uyarısı
                if guven < 0.5:
                    cevap = f"⚠️ Bu soruyu tam olarak anlayamadım (Güven: {guven:.0%}). " \
                            f"Belki şunu sormak istediniz: *\"{sonuc['en_benzer_soru']}\"*?\n\n{cevap}"
                
                st.markdown(cevap)
                
                # Metadata (Niyet & Güven)
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

if __name__ == "__main__":
    main()
