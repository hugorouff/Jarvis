import os
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import google.generativeai as genai

app = FastAPI()

# Récupération des clés API sécurisées (NOMS des variables, pas les vraies clés)
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
ELEVEN_KEY = os.getenv("ELEVENLABS_API_KEY")

genai.configure(api_key=GEMINI_KEY)

@app.get("/")
def home():
    return {"status": "Jarvis est en ligne et fonctionnel !"}

@app.post("/vocal")
async def traiter_vocal(file: UploadFile = File(...)):
    # 1. Sauvegarder l'audio envoyé par ton téléphone
    audio_bytes = await file.read()
    with open("input.wav", "wb") as f:
        f.write(audio_bytes)

    # 2. Transcrire & Générer la réponse avec Gemini
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = "Tu es Jarvis, l'assistant vocal d'Iron Man. Réponds de manière concise, élégante et directe."
    
    # Envoi de l'audio + prompt à Gemini
    uploaded_file = genai.upload_file("input.wav")
    response = model.generate_content([prompt, uploaded_file])
    texte_reponse = response.text

    # 3. Transformer le texte en voix avec ElevenLabs
    voice_id = "21m00Tcm4TlvDq8ikWAM"  # Voix de base (Rachel/Adam)
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVEN_KEY
    }
    
    data = {
        "text": texte_reponse,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
    }
    
    res = requests.post(url, json=data, headers=headers)
    with open("output.mp3", "wb") as f:
        f.write(res.content)

    # 4. Renvoyer le fichier vocal à ton téléphone
    return FileResponse("output.mp3", media_type="audio/mpeg")
