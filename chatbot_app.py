import streamlit as st

# --- CV Bilgi Havuzu ---
# Bu kısım, CV'nizden manuel olarak çıkarılmıştır
CV_BILGILERI = {
    "plc": "Vanderlande stajında Siemens PLC (TIA Portal) kullandım. Ayrıca SCADA ve HMI programlama tecrübem var.",
    "python": "Python, C/C++ ve MS SQL'de iyiyim. Ayrıca Görüntü İşleme ve ROS2 tecrübem var.",
    "tecrübe": "Neocom (Zayıf Akım Sistemleri) ve Vanderlande (Lojistik/Otomasyon) şirketlerinde staj yaptım.",
    "mekatronik": "Kocaeli Üniversitesi Mekatronik Mühendisliği mezunuyum.",
    "ingilizce": "İngilizce seviyem B2'dir.",
    "neocom": "Kamera sistemleri kurulumu, yangın panelleri ve acil anons sistemleri devreye alınmasında çalıştım.",
    "vanderlande": "Bagaj taşıma sistemleri otomasyonunda saha operasyonlarına destek verdim."
}


def chatbot_cevap_uret(soru):
    soru_kucuk = soru.lower()

    # Anahtar kelime eşleştirme ile cevap bulma
    for anahtar, yanit in CV_BILGILERI.items():
        if anahtar in soru_kucuk:
            return yanit

    return "CV'deki bilgilerime özgü bir soru sorun (Örn: PLC, Python, Vanderlande). Unutmayın, ben sadece CV'mdeki bilgilere dayanarak cevap verebilen bir prototipim."


# --- STREAMLIT ARAYÜZÜ ---

st.title("Yahya Osman Tamdoğan CV Asistanı 🤖")
st.markdown(
    "Mekatronik Mühendisi Yahya Osman Tamdoğan'ın CV'sini [cite: 50] kullanarak bu prototip AI Chatbot geliştirilmiştir.")

# Mesaj geçmişini tutma
if "mesajlar" not in st.session_state:
    st.session_state.mesajlar = []

# Daha önceki mesajları gösterme
for gonderici, mesaj in st.session_state.mesajlar:
    st.chat_message(gonderici).write(mesaj)

# Kullanıcı girişi
if prompt := st.chat_input("Bana Yahya Osman Tamdoğan'ın tecrübelerini sor..."):
    # Kullanıcının mesajını kaydet ve göster
    st.session_state.mesajlar.append(("user", prompt))
    st.chat_message("user").write(prompt)

    # Chatbot cevabını al
    cevap = chatbot_cevap_uret(prompt)

    # Chatbot cevabını kaydet ve göster
    st.session_state.mesajlar.append(("assistant", cevap))
    st.chat_message("assistant").write(cevap)