import google.generativeai as genai

# Вставь свой ключ сюда:
API_KEY = "AIzaSyBse7MaaHNOyxM1s_u39aKIq4YPgvIU-BY" 

genai.configure(api_key=API_KEY)

print("Доступные модели для твоего ключа:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"✅ {m.name}")