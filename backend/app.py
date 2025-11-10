# backend/app.py
import os, re, json
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# -------- ENV & APP --------
load_dotenv()
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "5000"))
MODEL_H5_PATH = os.getenv("MODEL_H5_PATH", "/Users/bhav/Stuff/Misc/SPHINX/backend/models/model.h5")  # change if your .h5 lives elsewhere

app = Flask(__name__)
CORS(app)

# -------- Gemini (google-generativeai==0.4.0 recommended) --------
try:
    import google.generativeai as genai
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI = genai.GenerativeModel("gemini-2.5-flash")
    else:
        GEMINI = None
except Exception:
    GEMINI = None

# -------- ISL unified model (single+dual hand) --------
from models.isl_model import ISLModel
ISL = ISLModel(model_path=MODEL_H5_PATH)

# -------- ROUTES --------
@app.get("/health")
def health():
    ok = bool(ISL) and (GEMINI is not None or os.getenv("GEMINI_API_KEY", "") == "")
    return jsonify({"ok": ok, "model_path": MODEL_H5_PATH, "gemini": GEMINI is not None})

@app.post("/predict_char")
def predict_char():
    try:
        data = request.get_json(force=True) or {}
        img_b64 = data.get("image")  # data URL or raw base64
        if not img_b64:
            return jsonify({"success": False, "error": "missing image"}), 400
        out = ISL.predict_char(img_b64)
        return jsonify({
            "success": True,
            "char": out.get("char", "") or "",
            "confidence": float(out.get("confidence", 0.0))
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.post("/finalize_word")
def finalize_word():
    try:
        data = request.get_json(force=True) or {}
        word = (data.get("word") or "").strip()
        language = (data.get("language") or "Hindi").strip()
        if not word:
            return jsonify({"success": False, "error": "missing word"}), 400

        corrected, translated = word, word
        if GEMINI:
            prompt = (
                "You will receive one English word from an ISL recognizer.\n"
                "Return strict JSON with keys: corrected, translated.\n"
                f'Correct the spelling/casing of the word and translate into "{language}".\n'
                f"Word: {word}"
            )
            try:
                resp = GEMINI.generate_content(prompt)
                txt = (getattr(resp, "text", None) or "").strip()
                m = re.search(r"\{.*\}", txt, re.DOTALL)
                if m:
                    j = json.loads(m.group(0))
                    corrected = j.get("corrected", corrected)
                    translated = j.get("translated", translated)
            except Exception:
                # fallback to raw word if parsing fails
                pass

        return jsonify({"success": True, "corrected": corrected, "translated": translated, "language": language})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# -------- MAIN --------
if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=False)