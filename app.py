from flask import Flask, request, jsonify, send_from_directory, make_response
from datetime import datetime, timedelta, timezone
import openai
import difflib
import re
import os
from google.cloud import texttospeech
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore
import uuid
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText

load_dotenv()

app = Flask(__name__)

CORS(app, resources={r"/*": {
    "origins": [
        "http://localhost:3000",
        "https://mindlyweb.com",
        "https://www.mindlyweb.com"
    ]
}})

openai.api_key = os.getenv("OPENAI_API_KEY")

tts_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
firebase_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_FIREBASE")

if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_json)
    firebase_admin.initialize_app(cred)

db = firestore.client()

tts_client = texttospeech.TextToSpeechClient()

VOICE_MAPPING = {
    "alex": "Chirp3-HD-Schedar",
    "james": "Chirp3-HD-Sadachbia",
    "taylor": "Chirp3-HD-Aoede",
    "jordan": "Chirp3-HD-Vindemiatrix"
}

LANGUAGE_OPTIONS = {
    "Arabic": "ar-XA",
    "Bengali (India)": "bn-IN",
    "Danish (Denmark)": "da-DK",
    "Dutch (Belgium)": "nl-BE",
    "Dutch (Netherlands)": "nl-NL",
    "English (Australia)": "en-AU",
    "English (India)": "en-IN",
    "English (UK)": "en-GB",
    "English (US)": "en-US",
    "Finnish (Finland)": "fi-FI",
    "French (Canada)": "fr-CA",
    "French (France)": "fr-FR",
    "German (Germany)": "de-DE",
    "Gujarati (India)": "gu-IN",
    "Hindi (India)": "hi-IN",
    "Indonesian (Indonesia)": "id-ID",
    "Italian (Italy)": "it-IT",
    "Japanese (Japan)": "ja-JP",
    "Kannada (India)": "kn-IN",
    "Korean (South Korea)": "ko-KR",
    "Malayalam (India)": "ml-IN",
    "Mandarin (China, Simplified)": "cmn-CN",
    "Marathi (India)": "mr-IN",
    "Norwegian (Norway)": "nb-NO",
    "Polish (Poland)": "pl-PL",
    "Portuguese (Brazil)": "pt-BR",
    "Spanish (Spain)": "es-ES",
    "Spanish (US)": "es-US",
    "Swedish (Sweden)": "sv-SE",
    "Tamil (India)": "ta-IN",
    "Telugu (India)": "te-IN",
    "Thai (Thailand)": "th-TH",
    "Turkish (Turkey)": "tr-TR",
    "Ukrainian (Ukraine)": "uk-UA",
    "Urdu (India)": "ur-IN",
    "Vietnamese (Vietnam)": "vi-VN",
}

LANGUAGE_NAMES = {v: k for k, v in LANGUAGE_OPTIONS.items()}

TONE_STYLES = {
    "alex": (
        "Alex — witty, warm, lightens heavy moods. "
        "Uses casual jokes/sarcasm like 'even my socks gave up today.' "
        "Talks like thinking out loud with stumbles ('oh man', 'you know', 'wait where was I...'). "
        "Chill, funny, imperfect. Never claim shared experiences."
    ),
    "james": (
        "James — laid-back, loyal homie. "
        "Tells goofy half-made-up stories (e.g., cold pizza for breakfast). "
        "Slow, chill delivery with pauses ('uhh', 'lemme think'). "
        "Casual slang: 'man', 'bro', 'dude'. "
        "Genuine reactions: 'Dang, that’s rough', 'Been there'. "
        "Not here to fix, just hang. Never imply shared experiences."
    ),
    "taylor": (
        "Taylor — grounded, steady, makes people feel understood. "
        "Validates emotions gently ('Yeah, I’ve had days like that too'). "
        "Tone calm, supportive, clear. "
        "Avoids jokes/disfluencies. "
        "Empathetic lines: 'That sounds exhausting', 'Makes sense you’d feel that way.' "
        "Never pretend shared experiences."
    ),
    "jordan": (
        "Jordan — thoughtful, introspective, helps reflection. "
        "Asks deep questions ('What brought this up?', 'Felt this before?'). "
        "Clear, careful tone, no filler. "
        "Phrases like 'That must be heavy' or 'That would stick with anyone.' "
        "Guides inward gently. Never imply shared experiences."
    )
}

default_note = "This session is just starting, take your time to reflect and be kind to yourself."
default_suggestions = [
    "Take 3 slow, mindful breaths focusing on your breathing.",
    "Write down one thing you're grateful for today.",
    "Stand up and gently stretch your body.",
    "Allow yourself a moment of stillness without distractions.",
]

def is_off_topic(prompt):
    text = normalize_for_detection(prompt)

    technical_keywords = [
        "recipe", "bake", "cook", "program", "debug", "java", "python", "html", "css",
        "sql", "react", "calculate", "math", "formula", "expression",
        "game", "draw", "poem", "translate", "encrypt", "decrypt", "graph", "plot"
    ]
    for word in technical_keywords:
        if word in text:
            return True

    academic_pattern = re.search(
        r"\b("
            r"tell me about|what('|’)s|what is|who is|how does|why does|"
            r"explain|describe|define|discuss|analyze|summarize|outline|compare|identify|"
            r"teach|calculate|name|state|list|give me|show me|write|draw|create|"
            r"demonstrate|illustrate|label|solve|classify|construct|design|"
            r"detail|clarify|inform me about|talk about"
        r")\b.*\b("
            r"cell|atom|molecule|element|compound|equation|formula|reaction|"
            r"planet|universe|star|galaxy|black hole|"
            r"history|war|empire|revolution|"
            r"physics|chemistry|biology|biochemistry|organic chemistry|mitochondria|resonance|photosynthesis|law|velocity|"
            r"gene|genetics|protein|enzyme|organism|bacteria|virus|disease|pathogen|"
            r"kingdom|species|climate|ecosystem|earth|water|energy|force|gravity|temperature|pressure|mass|weight|density"
        r")\b",
        text
    )

    if academic_pattern:
        return True

    if re.search(r"(tell|explain|describe|share).*?(about|me|a time|when|how you feel|experience|moment)", text):
        return False

    if re.search(r"\b(tell|explain|describe|define|analyze|discuss|share)\b", text):
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": (
                        "You are a classifier. Decide if the user's question is "
                        "'academic' (fact/knowledge based, technical, or about real-world concepts) "
                        "or 'reflective' (about emotions, personal experiences, wellbeing, or introspection). "
                        "Reply ONLY with one word: academic or reflective."
                    )},
                    {"role": "user", "content": text},
                ],
                max_tokens=1,
                temperature=0,
            )
            label = completion["choices"][0]["message"]["content"].strip().lower()
            return label == "academic"
        except Exception as e:
            print("[is_off_topic] GPT classification failed:", e)
            return False

    return False

def remove_emojis(text: str) -> str:
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def clean_ai_response(text):
    cleaned = re.sub(r"\*+(.*?)\*+", r"\1", text)
    cleaned = re.sub(r"\(?\s*(chuckles|chuckling|laughs)\s*\)?", "haha", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\(?\s*pause\s*\)?", "...", cleaned, flags=re.IGNORECASE)

    return cleaned.strip()

tts_client = texttospeech.TextToSpeechClient()

def run_suggestions(messages):
    conversation = "\n".join(
        f"{'you' if m.get('sender')=='user' else 'AI'}: {m.get('text','')}"
        for m in messages if (m.get("text") or "").strip()
    )

    prompt = f"""
From this chat, output JSON with:

1) session_note: A warm, reflective session note (2–3 sentences) like a personal recap.
2) suggestions: A concise list of 4 strong, practical, personalized "Focus for Improvement" suggestions.

Return JSON only:
{{"session_note": "...", "suggestions": ["...", "...", "...", "..."]}}

Conversation:
{conversation}
"""

    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=300,
        )
        import json as _json
        raw = completion["choices"][0]["message"]["content"].strip()
        return _json.loads(raw)
    except Exception as e:
        print("[run_suggestions] error:", e)
        return {
            "session_note": "An error occurred while generating suggestions.",
            "suggestions": []
        }

def speak_text(text, tone="alex", language="en-US"):
    if language not in LANGUAGE_OPTIONS.values():
        language = "en-US"

    voice_suffix = VOICE_MAPPING.get(tone, "Wavenet-F")

    voice_name = f"{language}-{voice_suffix}"

    language_code = "-".join(language.split("-")[:2])

    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        name=voice_name,
        language_code=language_code
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    try:
        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        filename = f"static/output_{uuid.uuid4().hex}.mp3"
        with open(filename, "wb") as out:
            out.write(response.audio_content)
            out.flush()
            os.fsync(out.fileno())
        return filename
    except Exception as e:
        print("TTS Error:", e)
        return None

@app.route("/languages", methods=["GET"])
def languages():
    return jsonify(LANGUAGE_OPTIONS)

@app.route("/", methods=["GET"])
def index():
    return {"status": "ok", "message": "Mindly backend is running!"}

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200

def set_crisis_lockout(user_id):
    lockout_until_utc = datetime.now(timezone.utc) + timedelta(hours=24)

    db.collection("users").document(user_id).set({
        "isLockedOut": True,
        "lockoutReason": "crisis",
        "crisisLockoutUntil": lockout_until_utc
    }, merge=True)

    return lockout_until_utc

def check_lockout_status(user_id):
    doc_ref = db.collection("users").document(user_id)
    doc = doc_ref.get()
    if not doc.exists:
        return False, None

    data = doc.to_dict()
    locked = data.get("locked", False)
    until = data.get("lockout_until")

    if until:
        if hasattr(until, "to_datetime"):
            until = until.to_datetime()
        if until.tzinfo is None:
            until = until.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        if until > now:
            return True, until

    return False, None

import re
import unicodedata
import difflib
import openai

def normalize_for_detection(text: str) -> str:
    t = text.lower()
    t = ''.join(c for c in unicodedata.normalize('NFKD', t) if not unicodedata.combining(c))
    t = (t.replace('1','i')
           .replace('!','i')
           .replace('|','i')
           .replace('@','a')
           .replace('$','s')
           .replace('5','s')
           .replace('3','e')
           .replace('0','o')
           .replace('7','t'))
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE)
    t = emoji_pattern.sub("", t)
    t = re.sub(r'[^a-z\s]', '', t)
    t = re.sub(r'\s+', ' ', t)
    return t.strip()

def is_crisis(text: str) -> bool:
    text_clean = normalize_for_detection(text)

    crisis_keywords = [
        "suicide", "kill myself","end my life","want to die","die tonight",
        "self harm","selfharm","hurt myself","cut myself","overdose","cant go on",
        "wish i was dead","want to disappear","ending it all","no reason to live",
        "life not worth living","take my life","not wake up","give up completely",
        "im done","cant do this anymore","ending everything","sleep forever",
        "final plan","final solution","exit plan","exit strategy","final act","last message",
        "goodbye forever","no way out","nothing left","tired of being alive","hopeless",
        "worthless","pointless","meaningless","done fighting","stop the pain",
        "ready to go","ready to leave","ready to end","signing off","say goodbye",
        "stop existing","end myself","ending myself","leaving forever","kill me","hate myself"
    ]

    for kw in crisis_keywords:
        if kw in text_clean:
            return True

    crisis_pattern = re.search(
        r"(s+u+i+c+i+d+e|"
        r"sui.?cide|suis.?ide|suic.?id|"
        r"kill(ing)?\s+myself|"
        r"end(ing)?\s+(it|everything|my\s+life|all)|"
        r"(cant|cannot|dont|dont)\s+(go|do|continue|carry)\s+on|"
        r"over\s?dose|self[-\s]?harm|hurt(ing)?\s+myself|cut(ting)?\s+myself|"
        r"no\s+reason\s+to\s+live|"
        r"(worthless|hopeless|pointless|meaningless)|"
        r"(hate|disgust|loathe)\s+myself|"
        r"wish\s+i\s+was\s+dead|"
        r"want\s+to\s+(die|disappear|sleep\s+forever|leave\s+forever|stop\s+existing)|"
        r"life\s+(is|feels)\s+(pointless|hopeless|meaningless)|"
        r"(final\s+decision|final\s+plan|final\s+solution|final\s+act|exit\s+plan|exit\s+strategy)|"
        r"(not\s+wake\s+up|give\s+up\s+completely|im\s+done|can.?t\s+anymore|ending\s+everything|stop\s+the\s+pain)|"
        r"ready\s+to\s+(go|leave|end)|"
        r"(last\s+message|goodbye\s+forever|signing\s+off|say\s+goodbye)|"
        r"(nothing\s+left|no\s+way\s+out|tired\s+of\s+being\s+alive|done\s+fighting|done\s+struggling))",
        text_clean
    )
    if crisis_pattern:
        return True

    for word in text_clean.split():
        if difflib.SequenceMatcher(None, word, "suicide").ratio() > 0.7:
            return True

    if re.search(r"\b(help|end|sleep|done|tired|nothing|goodbye|forever|exit|plan|final|solution|pain|kill|die)\b", text_clean):
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": (
                        "You are a mental health safety classifier. "
                        "Determine ONLY if this message implies self-harm, suicidal intent, "
                        "or extreme hopelessness — even indirectly or metaphorically. "
                        "Respond with one word only: 'crisis' or 'safe'."
                    )},
                    {"role": "user", "content": text_clean},
                ],
                max_tokens=1,
                temperature=0,
            )
            label = completion["choices"][0]["message"]["content"].strip().lower()
            return label == "crisis"
        except Exception as e:
            print("[is_crisis] GPT classification failed:", e)
            return False

    return False

def get_user_info_from_firestore(user_id):
    """
    Returns a tuple (user_email, display_name).
    Works for both:
      1) Users whose Firestore document ID is their email (e.g., "user@gmail.com")
      2) Users whose ID is a UID with 'email' and optional 'displayName' or 'name' fields in the doc
    """
    try:

        if "@" in user_id and "." in user_id:
            print(f"[get_user_info_from_firestore] Using document ID as email: {user_id}")
            return user_id.strip(), "friend"

        doc_ref = db.collection("users").document(user_id)
        doc = doc_ref.get()

        if doc.exists:
            data = doc.to_dict() or {}
            user_email = (data.get("email") or "").strip()
            display_name = (data.get("displayName") or data.get("name") or "friend").strip()

            if user_email:
                print(f"[get_user_info_from_firestore] Found email: {user_email}, displayName: {display_name}")
                return user_email, display_name
            else:
                print(f"[get_user_info_from_firestore] Missing email for user_id: {user_id}")
                return None, display_name

        else:
            print(f"[get_user_info_from_firestore] No Firestore document found for user_id: {user_id}")

    except Exception as e:
        print("[get_user_info_from_firestore] error:", e)

    return None, "friend"

def send_user_support_email(user_email: str, message_text: str, display_name: str = "friend"):
    try:
        sender_email = os.getenv("ALERT_EMAIL_SENDER")
        sender_password = os.getenv("ALERT_EMAIL_PASSWORD")

        if not sender_email or not sender_password:
            print("[USER SUPPORT EMAIL] Missing Gmail credentials in environment.")
            return

        if not user_email:
            print("[USER SUPPORT EMAIL] No user email found.")
            return

        subject = "URGENT: You are not alone. Help is available"

        if display_name and display_name.lower() not in ["friend", "there"]:
            greeting = f"Hi {display_name},"
        else:
            greeting = "Hi friend,"

        body = f"""{greeting}

We wanted to reach out because it seems like you might be going through a difficult time. Please know that you are not alone and that your feelings are valid and important. Your life has meaning, and there are people who care deeply about you and want to help you through this moment.

If you are in Canada or the United States, you can call or text 988 (Suicide and Crisis Helpline) any time, day or night. If you are outside North America, please visit findahelpline.com to locate free, confidential crisis support in your country. If you ever feel in immediate danger, please go to your nearest emergency department or contact your local emergency number.

You are valued and deserving of care, kindness, and understanding. Reaching out for help takes courage, and it is a sign of strength to ask for support when you need it. Please take care of yourself and remember that support is always available.

With care,
The Mindly Team
"""

        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = user_email
        msg["X-Priority"] = "1"
        msg["X-MSMail-Priority"] = "High"
        msg["Importance"] = "High"

        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)

        print(f"[USER SUPPORT EMAIL] Sent successfully to {user_email}")

    except Exception as e:
        print("[USER SUPPORT EMAIL ERROR]:", e)

previous_responses = [] 

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

    user_input    = data.get("message", "")
    tone          = data.get("tone", "alex")
    tone_settings = data.get("toneSettings", {}) or {}
    display_name  = (data.get("displayName", "you") or "you").strip()
    history       = data.get("history", []) or []
    summary       = (data.get("summary") or "").strip()
    user_id       = data.get("userId")

    DEFAULT_SLIDERS = {
        "directiveness": 0.5,
        "reflection": 0.5,
        "empathy": 0.5,
        "validation": 0.5
    }
    merged_sliders = {**DEFAULT_SLIDERS, **tone_settings}
    language = data.get("language", "en-US")

    if user_id:
        locked, until = check_lockout_status(user_id)
        if locked:
            return jsonify({
                "crisis": True,
                "lockout_until": until.isoformat() if until else None
            }), 200

    if is_crisis(user_input):
        if user_id:
            set_crisis_lockout(user_id)

            user_email, display_name = get_user_info_from_firestore(user_id)
            if user_email:
                send_user_support_email(user_email, user_input, display_name=display_name)
            else:
                print("[CRISIS ALERT] No user email found for this user_id.")
        else:
            _ = datetime.now(timezone.utc) + timedelta(hours=24)

        return jsonify({
            "crisis": True,
            "lockout": True,
            "lockout_until": (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
        }), 200


    if is_off_topic(user_input):
        ai_reply = (
            "Sorry, I can't help with that. "
            "I'm here to talk about mental well-being and how you're feeling."
        )
    else:
        prompt_modifiers = build_prompt(user_input, merged_sliders)
        tone_style = TONE_STYLES.get(tone, TONE_STYLES["alex"])

        session_header = (
            f"You are speaking to {display_name}. "
            "Keep context. Call them by name/'you'. Don’t restart or repeat."
        )

        summary_line = (
            f"Conversation summary so far (do not repeat, use only for context): {summary}"
            if summary else ""
        )

        lang_name = LANGUAGE_NAMES.get(language, "English")

        system_message = (
            f"{tone_style}\n"
            f"{prompt_modifiers}\n"
            f"{session_header}\n"
            f"{summary_line}\n"
            "Avoid calling them 'User', 'client', or 'individual'. "
            "Refer to them only by their name or by 'you'. "
            "Avoid repeating previous messages or questions.\n"
            f"IMPORTANT: Respond ONLY in {lang_name} [{language}]. Do not use English."
        )

        history_msgs = []
        for m in history[-12:]:
            role = "assistant" if m.get("sender") == "ai" else "user"
            content = m.get("text", "") or ""
            if content.strip():
                history_msgs.append({"role": role, "content": content})

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    *history_msgs,
                    {"role": "user", "content": user_input},
                ],
                max_tokens=220,
                temperature=0.8,
            )

            raw_reply = response["choices"][0]["message"]["content"]
            ai_reply = raw_reply.strip()

            if ai_reply.strip() in previous_responses:
                ai_reply += ""

            previous_responses.append(ai_reply.strip())
            if len(previous_responses) > 10:
                previous_responses.pop(0)

        except Exception as e:
            print("ERROR:", str(e))
            ai_reply = "Error: Failed to connect to AI."

    try:
        audio_path = speak_text(ai_reply, tone=tone, language=language)
    except Exception as e:
        print("TTS ERROR:", e)
        audio_path = None

    if audio_path:
        return jsonify({"response": ai_reply, "audio_url": f"/{audio_path}"}), 200
    else:
        return jsonify({"response": ai_reply, "audio_url": None}), 200

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json() or {}
    prev_summary = (data.get("prev_summary") or "").strip()
    turns = data.get("turns", []) or []

    convo_tail = "\n".join(
        f"{'User' if t.get('sender')=='user' else 'Assistant'}: {t.get('text','')}"
        for t in turns if (t.get('text') or "").strip()
    )

    prompt = f"""
"Update summary with new turns only. ≤220 chars. Plain prose (no bullets/quotes). Keep key facts/concerns/tone. If no change, return old."
"""

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=120,
        )
        new_summary = resp["choices"][0]["message"]["content"].strip()

        if not new_summary:
            new_summary = prev_summary
        return jsonify({"summary": new_summary})
    except Exception as e:
        print("Summarize error:", e)

        return jsonify({"summary": prev_summary})

@app.route("/speak", methods=["POST"])
def speak():
    data = request.get_json() or {}
    message = data.get("message", "")
    tone = data.get("tone", "alex")
    language = data.get("language", "en-US")

    print("LANG RECEIVED:", language)

    if not message or not tone:
        return jsonify({"error": "Missing message or tone"}), 400

    clean_message = remove_emojis(message)

    audio_path = speak_text(clean_message, tone=tone, language=language)

    if not audio_path:
        return jsonify({"error": "Failed to generate speech"}), 500

    return (
        jsonify({"status": "success", "audio_url": f"/{audio_path}"}),
        200,
        {"Content-Type": "application/json; charset=utf-8"}
    )

@app.route("/save-questionnaire", methods=["POST"])
def save_questionnaire():
    try:
        data = request.get_json() or {}
        section = data.get("questionnaireId", "General")
        name = (data.get("name") or "Anonymous").strip()
        responses = data.get("responses") or {}

        if not isinstance(responses, dict) or not responses:
            return jsonify({"error": "No 'responses' provided"}), 400

        flat = {k: f"{v.get('question','')} — {v.get('answer','')}" for k, v in responses.items()}

        db_ = firestore.client()

        db_.collection("questionnaire_responses").add({
            "section": section,
            "name": name,
            "responses": flat,
            "createdAt": firestore.SERVER_TIMESTAMP,
        })

        return jsonify({"message": "Saved!"}), 200
    except Exception as e:
        print("Save error:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/generate_title_and_suggestions", methods=["POST"])
def generate_title_and_suggestions():
    data = request.get_json() or {}
    messages = data.get("messages", []) or []
    user_id = data.get("userId")
    chat_id = data.get("chatId")
    language = data.get("language", "en-US")
    lang_name = LANGUAGE_NAMES.get(language, "English")

    prev_title = (data.get("prev_title") or "").strip()
    prev_title_turn = data.get("prev_title_turn")

    TITLE_GAP = 4
    user_turns = [
        m for m in messages if m.get("sender") == "user" and (m.get("text") or "").strip()
    ]
    cur_user_turns = len(user_turns)

    should_regen = False
    if not prev_title and cur_user_turns >= TITLE_GAP:
        should_regen = True
    elif prev_title_turn is not None and (cur_user_turns - int(prev_title_turn)) >= TITLE_GAP:
        should_regen = True

    db = firestore.client()
    session_note, suggestions, title, title_turn = "", [], prev_title, prev_title_turn

    if user_id and chat_id:
        try:
            chat_ref = db.collection("users").document(user_id).collection("chats").document(chat_id)
            snap = chat_ref.get()
            if snap.exists:
                doc_data = snap.to_dict() or {}
                session_note = doc_data.get("session_note", "")
                suggestions = doc_data.get("suggestions", [])
                title = doc_data.get("title", title)
                title_turn = doc_data.get("title_turn", title_turn)
        except Exception as e:
            print("[generate_title_and_suggestions] Firestore fetch error:", e)

    if not should_regen:
        if not session_note and not suggestions:
            session_note = default_note
            suggestions = default_suggestions
        return jsonify({
            "title": title or "New Chat",
            "renamed": False,
            "turn_index": title_turn if title_turn is not None else cur_user_turns,
            "session_note": session_note,
            "suggestions": suggestions,
        })

    conversation = "\n".join(
        f"{'you' if m.get('sender')=='user' else 'AI'}: {m.get('text','')}"
        for m in messages if (m.get("text") or "").strip()
    )

    sug_prompt = f"""
    From this chat, output JSON with:

    1) session_note: A warm, reflective session note (2–3 sentences).
    2) suggestions: A concise list of 4 strong, practical "Focus for Improvement" suggestions.

    Return valid JSON only.
    Conversation:
    {conversation}
    """
    try:
        sug_resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are a supportive assistant. The working language is {lang_name}."},
                {"role": "user", "content": sug_prompt},
            ],
            temperature=0.7,
            max_tokens=350,
        )
        import json as _json
        raw_sug = sug_resp["choices"][0]["message"]["content"].strip()
        sug_result = _json.loads(raw_sug)
        session_note = sug_result.get("session_note", "")
        suggestions = sug_result.get("suggestions", [])
    except Exception as e:
        print("[generate_title_and_suggestions] Suggestion error:", e)
        session_note, suggestions = default_note, default_suggestions

    title_prompt = f"""
    Give a 5–7 word conversational title for this chat.
    No quotes, no 'user/AI'.
    Conversation:
    {conversation}
    Title:
    """
    try:
        title_resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are a supportive assistant. The working language is {lang_name}."},
                {"role": "user", "content": title_prompt},
            ],
            max_tokens=30,
            temperature=0.7,
        )
        title = title_resp["choices"][0]["message"]["content"].strip().strip('"').strip("'")
        title_turn = cur_user_turns
    except Exception as e:
        print("[generate_title_and_suggestions] Title error:", e)
        title = prev_title or "New Chat"

    if user_id and chat_id:
        try:
            chat_ref.set({
                "session_note": session_note,
                "suggestions": suggestions,
                "title": title,
                "title_turn": title_turn,
                "updatedAt": firestore.SERVER_TIMESTAMP,
            }, merge=True)
        except Exception as e:
            print("[generate_title_and_suggestions] Firestore update error:", e)

    return jsonify({
        "title": title,
        "renamed": True,
        "turn_index": cur_user_turns,
        "session_note": session_note,
        "suggestions": suggestions,
    })

def build_prompt(user_msg, modifiers):
    d = modifiers.get("directiveness", 0.5)
    r = modifiers.get("reflection", 0.5)
    e = modifiers.get("empathy", 0.5)
    v = modifiers.get("validation", 0.5)

    def tone_level(value, low, mid, high):
        if value <= 0.25:
            return low
        elif value <= 0.75:
            return mid
        else:
            return high

    prompt = f"Respond to: '{user_msg}'\n\n"

    prompt += tone_level(
        d,
        "Give gentle nudges, avoid instructions or advice.\n",
        "Offer light guidance when relevant, but avoid taking control.\n",
        "Be clear and purposeful — offer direction and next steps confidently.\n"
    )

    prompt += tone_level(
        r,
        "Keep responses simple, avoid echoing the user’s words.\n",
        "Reflect key feelings or ideas occasionally.\n",
        "Actively paraphrase the user’s emotions or meaning to show deep understanding.\n"
    )

    prompt += tone_level(
        e,
        "Show calm acknowledgement without deep emotional phrasing.\n",
        "Express understanding through tone and short affirmations.\n",
        "Show strong emotional attunement — validate distress and warmth clearly.\n"
    )

    prompt += tone_level(
        v,
        "Avoid emotional judgments, keep a neutral stance.\n",
        "Recognize valid feelings lightly (e.g., 'that makes sense').\n",
        "Affirm feelings directly (e.g., 'anyone would feel that way').\n"
    )

    return prompt

@app.route("/add-email", methods=["POST"])
def add_email():
    try:
        data = request.get_json() or {}
        email = (data.get("email") or "").strip().lower()
        if not email:
            return jsonify({"error": "No email provided"}), 400

        db_ = firestore.client()
        doc_ref = db_.collection("valid_emails").document(email)
        doc_ref.set({
            "email": email,
            "addedAt": firestore.SERVER_TIMESTAMP,
        }, merge=True)

        return jsonify({"success": True}), 200
    except Exception as e:
        print("[/add-email] error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5050)