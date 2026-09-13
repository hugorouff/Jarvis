import os
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import Response, HTMLResponse
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

def get_recent_emails():
    """Simule ou récupère les e-mails reçus."""
    return [
        {"from": "support@google.com", "subject": "Alerte de sécurité", "snippet": "Nouvelle connexion détectée..."},
        {"from": "contact@service.fr", "subject": "Facture disponible", "snippet": "Votre facture de septembre est en ligne..."}
    ]

def send_email(to_email: str, subject: str, body: str) -> str:
    """Fonction exécutée par le serveur pour envoyer un e-mail réel via SMTP Gmail."""
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        return "Erreur : Identifiants SMTP non configurés sur le serveur."
    
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain', 'utf-8'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
        return f"E-mail envoyé avec succès à {to_email}."
    except Exception as e:
        return f"Erreur lors de l'envoi de l'e-mail : {str(e)}"

@app.post("/vocal")
async def process_vocal(
    text_prompt: str = Form(None),
    file: UploadFile = File(None)
):
    try:
        prompt = text_prompt or "Bonjour Jarvis"

        # Déclaration de la fonction à Gemini (Function Calling)
        tools = [send_email]

        system_instruction = (
            "Tu es JARVIS, un assistant IA hautement avancé. "
            "Tu as la capacité d'envoyer des e-mails en utilisant l'outil `send_email`. "
            "Si l'utilisateur te demande d'envoyer un mail, utilise directement cette fonction. "
            "Sois concis, élégant et précis dans tes réponses."
        )

        # Ajout du contexte si l'utilisateur demande à lire ses mails
        extra_context = ""
        if any(w in prompt.lower() for w in ["lit", "lire", "recu", "reçu", "derniers mails"]):
            emails = get_recent_emails()
            extra_context = f"\n\n[DONNÉES SERVEUR - DERNIERS MAILS REÇUS]: {emails}"

        # Appel Gemini avec déclenchement d'outil automatique
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=tools
        )

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=f"{prompt}{extra_context}",
            config=config
        )

        # Si Gemini a appelé la fonction d'envoi de mail
        if response.function_calls:
            function_call = response.function_calls[0]
            if function_call.name == "send_email":
                args = function_call.args
                # Exécution de l'envoi
                result_message = send_email(
                    to_email=args.get("to_email"),
                    subject=args.get("subject"),
                    body=args.get("body")
                )
                
                # Deuxième passage pour que Gemini confirme l'envoi à l'utilisateur
                confirm_response = client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=f"L'action d'envoi d'e-mail a été exécutée. Résultat du serveur : {result_message}. Confirme-le à l'utilisateur brièvement.",
                    config=types.GenerateContentConfig(system_instruction=system_instruction)
                )
                reply_text = confirm_response.text or result_message
        else:
            reply_text = response.text or "Instruction reçue, Monsieur."

        # Synthèse vocale ElevenLabs
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
