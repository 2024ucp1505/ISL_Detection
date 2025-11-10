from flask import Flask, request, jsonify
from flask_cors import CORS
import os, base64
import numpy as np
import cv2
from dotenv import load_dotenv
# backend/app.py (only the new parts shown)

import os, json, base64
from flask import Flask, request, jsonify
from models.isl_model import ISLModel

# --- LLM (Gemini) ---
try:
    import google.generativeai as genai
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        G_MODEL = genai.GenerativeModel("gemini-2.5-flash")
    else:
        G_MODEL = None
except Exception:
    G_MODEL = None

app = Flask(__name__)

# --- singleton ISL model ---
ISL = ISLModel(model_path="backend/models/model.h5")  # adjust path if needed

# ---------- NEW: /predict_char ----------
@app.route("/predict_char", methods=["POST"])
def predict_char():
    try:
        data = request.get_json(force=True)
        image_b64 = data.get("image", "")
        if not image_b64:
            return jsonify({"success": False, "error": "missing image"}), 400

        out = ISL.predict_char(image_b64)
        # normalize empty char to ""
        ch = out.get("char", "") or ""
        conf = float(out.get("confidence", 0.0))
        return jsonify({"success": True, "char": ch, "confidence": conf})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ---------- NEW: /finalize_word ----------
@app.route("/finalize_word", methods=["POST"])
def finalize_word():
    try:
        data = request.get_json(force=True)
        word = (data.get("word") or "").strip()
        language = (data.get("language") or "Hindi").strip()
        if not word:
            return jsonify({"success": False, "error": "missing word"}), 400

        corrected, translated = word, word
        if G_MODEL:
            # single prompt → corrected + translated in one go
            prompt = (
                "You will receive one word from Indian Sign Language recognition.\n"
                "1) Return the cleaned/corrected English word (spelling/casing only).\n"
                f"2) Return its translation into {language}.\n"
                "Respond as strict JSON with keys: corrected, translated.\n"
                f"Word: {word}"
            )
            try:
                resp = G_MODEL.generate_content(prompt)
                # parse model text to JSON safely
                import json, re
                txt = resp.text or ""
                m = re.search(r"\{.*\}", txt, re.DOTALL)
                if m:
                    j = json.loads(m.group(0))
                    corrected = j.get("corrected", corrected)
                    translated = j.get("translated", translated)
            except Exception:
                # keep the fallbacks
                pass

        return jsonify({"success": True, "corrected": corrected, "translated": translated, "language": language})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
# ---- env ----
load_dotenv()
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "5000"))
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "gemini").lower()

# ---- models ----
from models.dummy_model import DummyISLModel
isl_model = DummyISLModel()

# ---- optional YOLO swap (activated when file exists & env path is valid) ----
YOLO_PATH = os.getenv("YOLO_MODEL_PATH", "models/best.pt")
if os.path.exists(YOLO_PATH):
    try:
        from models.yolo_model import YOLOISLModel
        isl_model = YOLOISLModel(model_path=YOLO_PATH)
        print(f"[boot] YOLO model loaded: {YOLO_PATH}")
    except Exception as e:
        print(f"[boot] YOLO not loaded ({e}). Using DummyISLModel.")

# ---- LLM providers ----
gemini_model = None
gemma_pipeline = None

if MODEL_PROVIDER == "gemini":
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        gemini_model = genai.GenerativeModel("gemini-2.5-flash")
        print("[boot] LLM: Gemini 1.5 Flash ready")
    except Exception as e:
        print(f"[boot] Gemini init failed: {e}")
elif MODEL_PROVIDER == "gemma":
    # Placeholder wire-up so you can flip later if you host Gemma
    # For hackathon speed, keep MODEL_PROVIDER=gemini today.
    try:
        # EXAMPLE (pseudocode): from vllm / text-generation-inference endpoint
        # import requests
        # GEMMA_ENDPOINT = os.getenv("GEMMA_ENDPOINT")
        # def gemma_generate(prompt): return requests.post(GEMMA_ENDPOINT, json={"prompt": prompt}).json()["text"]
        gemma_pipeline = None
        print("[boot] LLM: Gemma selected (no runtime wired here)")
    except Exception as e:
        print(f"[boot] Gemma init failed: {e}")

# ---- app ----
app = Flask(__name__)
CORS(app)

# ---------- helpers ----------
def _b64_to_ndarray(data_url: str):
    # input: "data:image/jpeg;base64,<...>"
    image_b64 = data_url.split(",", 1)[1] if "," in data_url else data_url
    image_bytes = base64.b64decode(image_b64)
    nparr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def _llm_correct(text: str) -> str:
    prompt = (
        "You are an autocorrect system for ISL-to-English output. "
        "Fix casing, spacing, and simple grammar. Respond with ONLY the corrected text.\n\n"
        f"Text: {text}\nCorrected:"
    )
    if MODEL_PROVIDER == "gemini" and gemini_model:
        r = gemini_model.generate_content(prompt)
        return (r.text or "").strip()
    elif MODEL_PROVIDER == "gemma" and gemma_pipeline:
        # r = gemma_generate(prompt)  # if you add it
        # return r.strip()
        return text  # placeholder fallback
    return text

def _llm_translate(text: str, target_lang: str) -> str:
    prompt = (
        f"Translate this text to {target_lang}. Respond with ONLY the translation.\n\n"
        f"Text: {text}\nTranslation:"
    )
    if MODEL_PROVIDER == "gemini" and gemini_model:
        r = gemini_model.generate_content(prompt)
        return (r.text or "").strip()
    elif MODEL_PROVIDER == "gemma" and gemma_pipeline:
        # r = gemma_generate(prompt)
        # return r.strip()
        return text  # placeholder fallback
    return text

# ---------- routes ----------
@app.get("/health")
def health():
    return jsonify(status="ok", provider=MODEL_PROVIDER)

@app.post("/predict")
def predict_isl():
    try:
        frame = _b64_to_ndarray(request.json["image"])
        pred_text, conf = isl_model.predict(frame)  # (text, confidence 0..1)
        return jsonify(success=True, prediction=pred_text, confidence=conf)
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500

@app.post("/autocorrect")
def autocorrect_text():
    try:
        text = request.json["text"]
        corrected = _llm_correct(text)
        return jsonify(success=True, corrected_text=corrected)
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500

@app.post("/translate")
def translate_text():
    try:
        text = request.json["text"]
        lang = request.json.get("language", "Hindi")
        translation = _llm_translate(text, lang)
        return jsonify(success=True, translation=translation, language=lang)
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500

if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=True)


