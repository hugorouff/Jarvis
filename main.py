import os
import json
import base64
import requests
from email.message import EmailMessage
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, HTMLResponse
from google import genai
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

app = FastAPI()

# Initialisation des clés et clients d'API
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
ELEVEN_KEY = os.getenv("ELEVENLABS_API_KEY")

client = genai.Client(api_key=GEMINI_KEY)

# Fonction d'envoi d'e-mail via l'API Gmail
def envoyer_email_gmail(destinataire: str, sujet: str, corps: str):
    token_raw = os.getenv("GMAIL_TOKEN_JSON")
    if not token_raw:
        raise ValueError("La variable GMAIL_TOKEN_JSON est absente des variables d'environnement.")

    token_info = json.loads(token_raw)
    creds = Credentials.from_authorized_user_info(token_info)
    service = build('gmail', 'v1', credentials=creds)

    message = EmailMessage()
    message.set_content(corps)
    message['To'] = destinataire
    message['From'] = 'rouff.hugo28@gmail.com'
    message['Subject'] = sujet

    encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
    create_message = {'raw': encoded_message}

    service.users().messages().send(userId="me", body=create_message).execute()

# Route d'accueil : Interface HUD Stark
@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

# Route unique /vocal pour gérer la parole et le texte
@app.post("/vocal")
async def traiter_commande(file: UploadFile = File(None), text_prompt: str = Form(None)):
    user_input = ""

    # 1. Extraction du message (Vocal ou Texte)
    if file:
        audio_bytes = await file.read()
        with open("input.wav", "wb") as f:
            f.write(audio_bytes)
        
        # Envoi de l'audio à Gemini pour transcription/analyse
        uploaded_file = client.files.upload(file="input.wav")
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                "Tu es Jarvis, l'assistant IA de Stark Industries. Réponds de façon concise et directe.", 
                uploaded_file
            ]
        )
        user_input = response.text
    elif text_prompt:
        user_input = text_prompt
    else:
        user_input = "Bonjour Jarvis."

    # 2. Détection de l'ordre d'envoi de mail ou réponse LLM standard
    if "envoie un mail" in user_input.lower() or "envoyer un mail" in user_input.lower():
        try:
            envoyer_email_gmail(
                destinataire="rouff.hugo28@gmail.com",
                sujet="Message de Jarvis",
                corps=user_input
            )
            texte_reponse = "J'ai envoyé l'e-mail à votre destinataire, Monsieur."
        except Exception as e:
            texte_reponse = f"Une erreur est survenue lors de l'envoi : {str(e)}"
    else:
        if not file:  # Si c'était un envoi texte
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=f"Tu es Jarvis, l'assistant IA de Tony Stark. Réponds de manière très concise et élégante : {user_input}"
            )
            texte_reponse = response.text
        else:
            texte_reponse = user_input

    # 3. Synthèse vocale avec ElevenLabs
    voice_id = "21m00Tcm4TlvDq8ikWAM"
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

    # 4. Envoi du fichier MP3 généré au client
    return FileResponse("output.mp3", media_type="audio/mpeg")
