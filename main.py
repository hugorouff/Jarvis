import os
import requests
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import Response, HTMLResponse
from google import genai

app = FastAPI()

# Initialisation des API
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
ELEVEN_KEY = os.getenv("ELEVENLABS_API_KEY")

client = genai.Client(api_key=GEMINI_KEY)

@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/vocal")
async def process_vocal(
    text_prompt: str = Form(None),
    file: UploadFile = File(None)
):
    try:
        prompt = text_prompt or "Bonjour Jarvis"

        # Appel à Gemini avec la version demandée par l'API (gemini-3.6-flash)
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
        )
        
        reply_text = response.text or "Instruction reçue, Monsieur."

        # Synthèse vocale avec ElevenLabs si la clé est présente
        if ELEVEN_KEY:
            voice_id = "21m00Tcm4TlvDq8ikWAM"
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": ELEVEN_KEY
            }
            data = {
                "text": reply_text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.5}
            }
            res = requests.post(url, json=data, headers=headers)
            if res.status_code == 200:
                return Response(
                    content=res.content, 
                    media_type="audio/mpeg", 
                    headers={"X-Jarvis-Text": reply_text}
                )

        return Response(content=reply_text.encode('utf-8'), media_type="text/plain")

    except Exception as e:
        print(f"Erreur backend : {e}")
        return Response(content=f"Erreur : {str(e)}".encode('utf-8'), status_code=500)
