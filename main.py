import os
import json
import base64
import requests
from email.mime.text import MIMEText
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import Response, HTMLResponse, FileResponse
from google import genai
from google.genai import types
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

app = FastAPI()

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
ELEVEN_KEY = os.getenv("ELEVENLABS_API_KEY")

client = genai.Client(api_key=GEMINI_KEY)


@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/manifest.json")
def get_manifest():
    return FileResponse("manifest.json", media_type="application/json")


def send_email_gmail(to_email: str, subject: str, body: str):
    """
    Sends an email through the Gmail API (HTTPS, port 443) using the OAuth
    token stored in GMAIL_TOKEN_JSON. This avoids the SMTP port-blocking /
    timeout issues that caused the previous 502 errors.
    Returns (success: bool, message: str).
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

        # Refresh the access token if it has expired (refresh_token handles this)
        if creds.expired and creds.refresh_token:
            from google.auth.transport.requests import Request
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


@app.post("/vocal")
async def process_vocal(
    text_prompt: str = Form(None),
    file: UploadFile = File(None),
):
    try:
        prompt = text_prompt or "Bonjour Jarvis"

        # Tool declaration exposed to Gemini
        def send_email(to_email: str, subject: str, body: str) -> str:
            """Envoie un e-mail réel au destinataire indiqué via l'API Gmail."""
            return "PENDING_SEND"  # placeholder, actual send handled below

        system_instruction = (
            "Tu es JARVIS. Si l'utilisateur demande d'envoyer un e-mail, "
            "utilise l'outil `send_email` avec un destinataire, un sujet et un corps clairs. "
            "Réponds toujours de façon concise et élégante."
        )

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[send_email],
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=config,
        )

        reply_text = "Instruction reçue, Monsieur."

        if response.function_calls:
            function_call = response.function_calls[0]
            if function_call.name == "send_email":
                args = function_call.args
                to_email = args.get("to_email")
                subject = args.get("subject", "Message de Jarvis")
                body = args.get("body", "")

                # Send synchronously so we can report the REAL result.
                # The Gmail API call is a quick HTTPS request, not a slow
                # SMTP handshake, so this won't trigger a 502 timeout.
                success, result_message = send_email_gmail(to_email, subject, body)

                if success:
                    reply_text = f"C'est fait, Monsieur. {result_message}"
                else:
                    reply_text = f"Je n'ai pas pu envoyer l'e-mail : {result_message}"
        else:
            reply_text = response.text or reply_text

        # Text-to-speech (optional, only if configured)
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
        print(f"Erreur backend : {e}")
        return Response(content=f"Erreur : {str(e)}".encode("utf-8"), status_code=500)
