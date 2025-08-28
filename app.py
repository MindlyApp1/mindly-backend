from flask import Flask, request, jsonify, send_from_directory, make_response
import openai
import difflib
import re
import os
from google.cloud import texttospeech
import stripe
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore
import uuid
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "https://mindly-95916.web.app"]}})

openai.api_key = os.getenv("OPENAI_API_KEY")
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

tts_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_TTS")
firebase_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_FIREBASE")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tts_json

if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_json)
    firebase_admin.initialize_app(cred)

db = firestore.client()

VOICE_MAPPING = {
    "alex": "en-US-Chirp3-HD-Schedar",
    "james": "en-US-Chirp3-HD-Sadachbia",
    "taylor": "en-US-Chirp3-HD-Aoede",
    "jordan": "en-US-Chirp3-HD-Vindemiatrix"
}

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

NOTALLOWEDWORDS = [
    "bake", "baking", "recipe", "cook", "coding", "program", "debug", "java", "python", "html", "css",
    "sql", "react", "calculate", "math", "formula", "expression", "game", "draw", "story", "poem",
    "translate", "encrypt", "decrypt", "plot", "graph"
]

PATTERNS = [
    r"\d+\s*[+\-*/x]\s*\d+",
    r"how (do|to) .*?\d+",
]

default_note = "This session is just starting, take your time to reflect and be kind to yourself."
default_suggestions = [
    "Take 3 slow, mindful breaths focusing on your breathing.",
    "Write down one thing you're grateful for today.",
    "Stand up and gently stretch your body.",
    "Allow yourself a moment of stillness without distractions.",
]

def is_off_topic(prompt):
    prompt_lower = prompt.lower()
    for keyword in NOTALLOWEDWORDS:
        if keyword in prompt_lower or difflib.SequenceMatcher(None, keyword, prompt_lower).ratio() > 0.8:
            return True
    for pattern in PATTERNS:
        if re.search(pattern, prompt_lower):
            return True
    return False

def remove_emojis(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF" u"\U0001F1E0-\U0001F1FF"
        u"\U00002700-\U000027BF" u"\U000024C2-\U0001F251"
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

def speak_text(text, tone="alex"):
    voice_name = VOICE_MAPPING.get(tone, "en-US-Wavenet-F")
    language_code = "-".join(voice_name.split("-")[:2])
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(name=voice_name, language_code=language_code)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

    try:
        response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        filename = f"static/output_{uuid.uuid4().hex}.mp3"
        with open(filename, "wb") as out:
            out.write(response.audio_content)
            out.flush()
            os.fsync(out.fileno())
        return filename
    except Exception as e:
        print("TTS Error:", e)
        return None

@app.route("/", methods=["GET"])
def index():
    return {"status": "ok", "message": "Mindly backend is running!"}

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200

previous_responses = []

@app.route("/chat", methods=["POST"])
def chat():
    global previous_responses
    data = request.get_json()

    user_input    = data.get("message", "")
    tone          = data.get("tone", "alex")
    tone_settings = data.get("toneSettings", {}) or {}
    display_name  = (data.get("displayName", "you") or "you").strip()
    history       = data.get("history", []) or []
    summary       = (data.get("summary") or "").strip()

    DEFAULT_SLIDERS = {
        "directiveness": 0.5,
        "reflection": 0.5,
        "empathy": 0.5,
        "validation": 0.5
    }

    merged_sliders = {**DEFAULT_SLIDERS, **tone_settings}

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

        system_message = (
            f"{tone_style}\n"
            f"{prompt_modifiers}\n"
            f"{session_header}\n"
            f"{summary_line}\n"
            "Avoid calling them 'User', 'client', or 'individual'. "
            "Refer to them only by their name or by 'you'. "
            "Avoid repeating previous messages or questions."
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
            ai_reply = clean_ai_response(
                remove_emojis(response["choices"][0]["message"]["content"])
            )

            if ai_reply.strip() in previous_responses:
                ai_reply += " (Trying not to repeat myself!)"

            previous_responses.append(ai_reply.strip())
            if len(previous_responses) > 10:
                previous_responses.pop(0)

        except Exception as e:
            print("ERROR:", str(e))
            ai_reply = "Error: Failed to connect to AI."

    audio_path = speak_text(ai_reply, tone=tone)
    return jsonify({"response": ai_reply, "audio_url": f"/{audio_path}"})

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
    data = request.get_json()
    message = data.get("message", "")
    tone = data.get("tone", "alex")

    if not message or not tone:
        return jsonify({"error": "Missing message or tone"}), 400

    clean_message = remove_emojis(message)
    audio_path = speak_text(clean_message, tone=tone)

    if not audio_path:
        return jsonify({"error": "Failed to generate speech"}), 500

    return jsonify({"status": "success", "audio_url": f"/{audio_path}"}), 200


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

    prev_title = (data.get("prev_title") or "").strip()
    prev_title_turn = data.get("prev_title_turn")

    TITLE_GAP = 6 

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
                data = snap.to_dict() or {}
                session_note = data.get("session_note", "")
                suggestions = data.get("suggestions", [])
                title = data.get("title", title)
                title_turn = data.get("title_turn", title_turn)
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

    sug_result = run_suggestions(messages)
    session_note = sug_result.get("session_note", "")
    suggestions = sug_result.get("suggestions", [])

    conversation = "\n".join(
        f"{'you' if m.get('sender')=='user' else 'AI'}: {m.get('text','')}"
        for m in messages if (m.get("text") or "").strip()
    )
    prompt = (
        "Give a 5–7 word title for this conversation. "
        "No quotes. Don't use 'user' or 'AI'.\n\n"
        f"{conversation}\n\nTitle:"
    )
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.7,
        )
        title = completion["choices"][0]["message"]["content"].strip().strip('"').strip("'")

        if title.lower().startswith("ai") or title.lower().startswith("/ai"):
            title = title.lstrip("/").split(":", 1)[-1].strip() or "New Chat"

        title_turn = cur_user_turns
    except Exception as e:
        print("[generate_title_and_suggestions] Title generation error:", e)
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

    prompt = f"Respond to: '{user_msg}'\n\n"

    prompt += "Directive.\n" if d >= 0.7 else "Low direction.\n"

    prompt += "Reflective.\n" if r >= 0.7 else "Little reflection.\n"

    prompt += "Empathetic.\n" if e >= 0.7 else "Neutral tone.\n"

    prompt += "Validate feelings.\n" if v >= 0.7 else "No validation.\n"

    return prompt

price_id_map = {
    "standard": "price_1RhgVqP5GIG16Xw0NLlekkEj",
    "premium": "price_1RhgX2P5GIG16Xw0oNAh4wnh",
}

lookup_table = price_id_map.copy()

@app.route("/create-checkout-session", methods=["POST"])
def create_checkout_session():
    try:
        data = request.get_json()
        print("[DEBUG] Incoming POST to /create-checkout-session with data:", data)

        plan = data.get("plan")
        price_id = lookup_table.get(plan)

        if not price_id:
            print("[ERROR] Invalid plan:", plan)
            return jsonify({"error": "Invalid plan ID"}), 400

        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{"price": price_id, "quantity": 1}],
            mode="subscription",
            success_url="http://localhost:3000/plans?status=success",
            cancel_url="http://localhost:3000/plans?status=cancel",
        )

        print("[SUCCESS] Stripe session created:", session.url)
        return jsonify({"url": session.url})

    except Exception as e:
        print("[ERROR] Stripe session creation failed:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/create-subscription", methods=["POST"])
def create_subscription():
    try:
        data = request.get_json()
        email = data.get("email")
        plan = data.get("plan")

        if not email or plan not in price_id_map:
            return jsonify({"error": "Invalid request data"}), 400

        customers = stripe.Customer.list(email=email).data
        customer = customers[0] if customers else stripe.Customer.create(email=email)

        subscription = stripe.Subscription.create(
            customer=customer.id,
            items=[{"price": price_id_map[plan]}],
            payment_behavior="default_incomplete",
            payment_settings={"save_default_payment_method": "on_subscription"},
            expand=["latest_invoice.payment_intent", "latest_invoice"]
        )

        db = firestore.client()
        user_ref = db.collection("users").document(email)
        user_ref.set({
            "plan": plan
        }, merge=True)

        payment_intent = subscription["latest_invoice"].get("payment_intent")
        if payment_intent:
            return jsonify({"clientSecret": payment_intent["client_secret"]})
        else:
            return jsonify({
                "message": "No immediate payment required",
                "subscriptionId": subscription.id,
                "invoiceId": subscription["latest_invoice"].id
            })

    except Exception as e:
        print(f"[create-subscription] Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/cancel-subscription", methods=["POST"])
def cancel_subscription():
    try:
        data = request.get_json()
        email = data.get("email")
        if not email:
            return jsonify({"error": "Missing email."}), 400

        db = firestore.client()
        user_ref = db.collection("users").document(email)

        customers = stripe.Customer.list(email=email).data
        if not customers:
            print("[INFO] No Stripe customer found — auto-downgrading.")
            user_ref.set({"plan": "basic"}, merge=True)
            return jsonify({
                "status": "downgraded_to_basic",
                "message": "No Stripe customer found, downgraded to Basic."
            })

        customer = customers[0]
        subscriptions = stripe.Subscription.list(customer=customer.id, limit=10).data
        if not subscriptions:
            print("[INFO] No active subscription found — auto-downgrading.")
            user_ref.set({"plan": "basic"}, merge=True)
            return jsonify({
                "status": "downgraded_to_basic",
                "message": "No Stripe subscription found, downgraded to Basic."
            })

        active_sub = next((sub for sub in subscriptions if sub.status in ['active', 'trialing', 'incomplete', 'past_due']), None)
        if active_sub:
            stripe.Subscription.delete(active_sub.id)
            print(f"[CANCEL] Subscription {active_sub.id} deleted.")
        else:
            print("[INFO] No valid subscription to cancel.")

        user_ref.set({"plan": "basic"}, merge=True)

        return jsonify({
            "status": "downgraded_to_basic",
            "message": "Subscription canceled and plan downgraded."
        })

    except Exception as e:
        print(f"[cancel-subscription] Error: {e}")
        return jsonify({"error": str(e)}), 500

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
    app.run(debug=True, port=8000)
