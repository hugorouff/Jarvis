import os
import json
import base64
import traceback
import requests
from email.mime.text import MIMEText
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import Response, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

app = FastAPI()

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
ELEVEN_KEY = os.getenv("ELEVENLABS_API_KEY")

client = genai.Client(api_key=GEMINI_KEY)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/manifest.json")
def get_manifest():
    return FileResponse("manifest.json", media_type="application/json")


@app.get("/favicon.ico")
def get_favicon():
    return FileResponse("static/favicon.ico", media_type="image/x-icon")


def _send_email_gmail(to_email: str, subject: str, body: str):
    """
    Envoie réellement un e-mail via l'API Gmail (HTTPS, port 443) en
    utilisant le jeton OAuth stocké dans GMAIL_TOKEN_JSON.
    Retourne (success: bool, message: str).
    """
    token_raw = os.getenv("GMAIL_TOKEN_JSON")
    if not token_raw:
        return False, "GMAIL_TOKEN_JSON est absent des variables d'environnement Render."

    try:
        token_info = json.loads(token_raw)
        creds = Credentials.from_authorized_user_info(
            token_info,
            scopes=["https://www.googleapis.com/auth/gmail.send"],
        )

        if creds.expired and creds.refresh_token:
            creds.refresh(Request())

        service = build("gmail", "v1", credentials=creds)

        message = MIMEText(body, "plain", "utf-8")
        message["to"] = to_email
        message["subject"] = subject

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        service.users().messages().send(userId="me", body={"raw": raw}).execute()

        return True, f"E-mail envoyé avec succès à {to_email}."

    except Exception as e:
        return False, f"Échec de l'envoi : {str(e)}"


# --- Outil réel exposé à Gemini (pas un leurre) ---
def send_email(to_email: str, subject: str, body: str) -> str:
    """Envoie un e-mail réel au destinataire indiqué via l'API Gmail.

    Args:
        to_email: adresse e-mail complète du destinataire.
        subject: objet de l'e-mail.
        body: contenu du message.
    """
    success, message = _send_email_gmail(to_email, subject, body)
    return message


SYSTEM_INSTRUCTION = (
    "Tu es JARVIS, un assistant IA élégant et concis. "
    "Si l'utilisateur demande d'envoyer un e-mail, utilise l'outil `send_email` "
    "avec un destinataire, un sujet et un corps clairs, puis confirme brièvement "
    "le résultat retourné par l'outil. Ne réponds jamais en anglais sauf si on te le demande."
)


@app.post("/vocal")
async def process_vocal(
    text_prompt: str = Form(None),
    file: UploadFile = File(None),
):
    try:
        prompt = text_prompt or "Bonjour Jarvis"

        # Pattern recommandé par Google pour le function calling automatique :
        # Chat.send_message plutôt que Models.generate_content direct.
        chat = client.chats.create(
            model="gemini-3.6-flash",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                tools=[send_email],
            ),
        )

        response = chat.send_message(prompt)
        reply_text = response.text or "Instruction reçue, Monsieur."

        # Synthèse vocale (optionnelle, seulement si la clé est configurée)
        if ELEVEN_KEY:
            try:
                voice_id = "21m00Tcm4TlvDq8ikWAM"
                url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
                headers = {
                    "Accept": "audio/mpeg",
                    "Content-Type": "application/json",
                    "xi-api-key": ELEVEN_KEY,
                }
                data = {
                    "text": reply_text,
                    "model_id": "eleven_monolingual_v1",
                    "voice_settings": {"stability": 0.5, "similarity_boost": 0.5},
                }
                res = requests.post(url, json=data, headers=headers, timeout=8)
                if res.status_code == 200:
                    return Response(
                        content=res.content,
                        media_type="audio/mpeg",
                        headers={"X-Jarvis-Text": reply_text},
                    )
            except Exception as audio_err:
                print(f"Erreur ElevenLabs (fallback texte) : {audio_err}")

        return Response(content=reply_text.encode("utf-8"), media_type="text/plain")

    except Exception as e:
        print("Erreur backend :", e)
        traceback.print_exc()
        return Response(content=f"Erreur : {str(e)}".encode("utf-8"), status_code=500)
