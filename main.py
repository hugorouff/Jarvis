import os
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from fastapi import FastAPI, Form, UploadFile, File, BackgroundTasks
from fastapi.responses import Response, HTMLResponse, FileResponse
from google import genai
from google.genai import types

app = FastAPI()

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
ELEVEN_KEY = os.getenv("ELEVENLABS_API_KEY")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")

client = genai.Client(api_key=GEMINI_KEY)

@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/manifest.json")
def get_manifest():
    return FileResponse("manifest.json", media_type="application/json")

def send_email_async(to_email: str, subject: str, body: str):
    """Envoi du mail en tâche de fond (évite le timeout 502)."""
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        print("Erreur SMTP: Identifiants absents dans l'environnement.")
        return
    
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain', 'utf-8'))

        with smtplib.SMTP('smtp.gmail.com', 587, timeout=10) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        print(f"E-mail transmis avec succès à {to_email}.")
    except Exception as e:
        print(f"Erreur lors de l'envoi SMTP : {e}")

@app.post("/vocal")
async def process_vocal(
    background_tasks: BackgroundTasks,
    text_prompt: str = Form(None),
    file: UploadFile = File(None)
):
    try:
        prompt = text_prompt or "Bonjour Jarvis"

        # Déclaration de l'outil pour Gemini
        def send_email(to_email: str, subject: str, body: str) -> str:
            """Permet d'envoyer un e-mail au destinataire indiqué."""
            return f"Demande d'envoi enregistrée pour {to_email}."

        system_instruction = (
            "Tu es JARVIS. Si l'utilisateur demande d'envoyer un e-mail, utilise l'outil `send_email`. "
            "Réponds immédiatement de manière concise en confirmant que le message est en cours d'envoi."
        )

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[send_email]
        )

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
            config=config
        )

        reply_text = "Instruction enregistrée, Monsieur."

        # Si Gemini déclenche l'envoi de mail
        if response.function_calls:
            function_call = response.function_calls[0]
            if function_call.name == "send_email":
                args = function_call.args
                to_email = args.get("to_email")
                subject = args.get("subject", "Message de Jarvis")
                body = args.get("body", "")

                # Lancement de l'envoi en arrière-plan sans bloquer la réponse
                background_tasks.add_task(send_email_async, to_email, subject, body)
                reply_text = f"Très bien Monsieur, l'e-mail pour {to_email} est en cours d'envoi."
        else:
            reply_text = response.text or reply_text

        # Synthèse vocale ElevenLabs (facultative)
        if ELEVEN_KEY:
            try:
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
                res = requests.post(url, json=data, headers=headers, timeout=5)
                if res.status_code == 200:
                    return Response(
                        content=res.content, 
                        media_type="audio/mpeg", 
                        headers={"X-Jarvis-Text": reply_text}
                    )
            except Exception as audio_err:
                print(f"Erreur ElevenLabs (fallback texte) : {audio_err}")

        return Response(content=reply_text.encode('utf-8'), media_type="text/plain")

    except Exception as e:
        print(f"Erreur backend : {e}")
        return Response(content=f"Erreur : {str(e)}".encode('utf-8'), status_code=500)
