import streamlit as st
import google.generativeai as genai

# Прямая проверка секретов
if "GEMINI_API_KEY" in st.secrets:
    st.write("✅ Секреты загружены!")
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    try:
        response = model.generate_content("Напиши 'Привет, Марсель! Система работает'.")
        st.write("🤖 Ответ ИИ:", response.text)
    except Exception as e:
        st.error(f"❌ Ошибка API: {e}")
else:
    st.error("🔴 Ключ не найден в .streamlit/secrets.toml")