import os
import json
import time
import base64
import traceback
import requests
from email.mime.text import MIMEText
from fastapi import FastAPI, Form, UploadFile, File, BackgroundTasks
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
UPSTASH_URL = os.getenv("UPSTASH_REDIS_REST_URL")
UPSTASH_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN")

client = genai.Client(api_key=GEMINI_KEY)

# Dossier static pour l'icône et autres médias si existant
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

HISTORY_KEY = "jarvis:history"
MAX_HISTORY_MESSAGES = 20  # ~10 échanges user/modèle conservés


@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/manifest.json")
def get_manifest():
    if os.path.exists("manifest.json"):
        return FileResponse("manifest.json", media_type="application/json")
    return Response(content="{}", media_type="application/json")


# ---------------------------------------------------------------------------
# MÉMOIRE PERSISTANTE (Upstash Redis via API REST)
# ---------------------------------------------------------------------------
def _upstash_available():
    return bool(UPSTASH_URL and UPSTASH_TOKEN)


def load_history():
    """Récupère l'historique de conversation stocké dans Upstash Redis."""
    if not _upstash_available():
        return []
    try:
        res = requests.get(
            f"{UPSTASH_URL}/get/{HISTORY_KEY}",
            headers={"Authorization": f"Bearer {UPSTASH_TOKEN}"},
            timeout=5,
        )
        data = res.json()
        raw = data.get("result")
        if not raw:
            return []
        return json.loads(raw)
    except Exception as e:
        print(f"[Upstash] Erreur de lecture : {e}")
        return []


def save_history(history):
    """Sauvegarde l'historique de conversation dans Upstash Redis."""
    if not _upstash_available():
        return
    try:
        trimmed = history[-MAX_HISTORY_MESSAGES:]
        payload = json.dumps(trimmed, ensure_ascii=False)
        requests.post(
            f"{UPSTASH_URL}/set/{HISTORY_KEY}",
            headers={"Authorization": f"Bearer {UPSTASH_TOKEN}"},
            data=payload.encode("utf-8"),
            timeout=5,
        )
    except Exception as e:
        print(f"[Upstash] Erreur d'écriture : {e}")


def clear_history():
    """Efface l'historique dans Upstash Redis."""
    if not _upstash_available():
        return
    try:
        requests.get(
            f"{UPSTASH_URL}/del/{HISTORY_KEY}",
            headers={"Authorization": f"Bearer {UPSTASH_TOKEN}"},
            timeout=5,
        )
    except Exception as e:
        print(f"[Upstash] Erreur de suppression : {e}")


# ---------------------------------------------------------------------------
# ENVOI D'E-MAIL (Gmail API)
# ---------------------------------------------------------------------------
def _send_email_gmail(to_email: str, subject: str, body: str):
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


def send_email(to_email: str, subject: str, body: str) -> str:
    """Envoie un e-mail réel au destinataire indiqué via l'API Gmail."""
    success, message = _send_email_gmail(to_email, subject, body)
    return message


SYSTEM_INSTRUCTION = (
    "Tu es JARVIS, un assistant IA élégant, réactif et concis. "
    "Tu te souviens parfaitement du contexte de la conversation. "
    "Si l'utilisateur te demande d'envoyer un e-mail, utilise l'outil `send_email` "
    "avec un destinataire, un sujet et un corps de texte."
)


def build_chat_with_history():
    """Recrée une session de chat Gemini à partir de l'historique persistant."""
    history_raw = load_history()
    genai_history = [
        types.Content(role=h["role"], parts=[types.Part(text=h["text"])])
        for h in history_raw
    ]
    return client.chats.create(
        model="gemini-3.6-flash",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            tools=[send_email],
        ),
        history=genai_history,
    )


@app.post("/reset")
def reset_memory():
    """Efface la mémoire de conversation (Upstash) et repart à zéro."""
    clear_history()
    return {"status": "ok", "message": "Mémoire de Jarvis réinitialisée."}


def send_message_with_retry(chat, contents, retries=2, delay=1.5):
    last_err = None
    for attempt in range(retries + 1):
        try:
            return chat.send_message(contents)
        except Exception as e:
            last_err = e
            print(f"[Gemini] Tentative {attempt + 1}/{retries + 1} échouée : {e}")
            if attempt < retries:
                time.sleep(delay)
    raise last_err


@app.post("/vocal")
async def process_vocal(
    text_prompt: str = Form(None),
    file: UploadFile = File(None),
    mime_type: str = Form(None),
):
    try:
        # 1) Charge l'historique et initialise le Chat Gemini
        chat = build_chat_with_history()
        user_display_text = text_prompt or "[Message vocal]"

        if file is not None:
            audio_bytes = await file.read()
            detected_mime = mime_type or file.content_type or "audio/webm"
            contents = [
                "Voici une instruction vocale de l'utilisateur. Transcris-la "
                "mentalement puis réponds-y directement en tant que JARVIS.",
                types.Part.from_bytes(data=audio_bytes, mime_type=detected_mime),
            ]
            response = send_message_with_retry(chat, contents)
        else:
            prompt = text_prompt or "Bonjour Jarvis"
            response = send_message_with_retry(chat, prompt)

        reply_text = response.text or "Instruction reçue, Monsieur."

        # 2) Sauvegarde dans la mémoire Upstash Redis
        history = load_history()
        history.append({"role": "user", "text": user_display_text})
        history.append({"role": "model", "text": reply_text})
        save_history(history)

        # 3) Synthèse vocale ElevenLabs (facultatif)
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
