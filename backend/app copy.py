from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from models.dummy_model import DummyISLModel
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Initialize dummy model
isl_model = DummyISLModel()

# Configure AI model based on choice
USE_GEMINI = True  # Set to False to use Gemma instead

if USE_GEMINI:
    import google.generativeai as genai
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("WARNING: GEMINI_API_KEY not found in .env file!")
    else:
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel('gemini-pro')
        print("✅ Gemini API configured successfully")
else:
    # Add Gemma configuration here when you switch
    print("⚠️ Using Gemma mode (not implemented yet)")
    pass

@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'message': 'Gestura Backend API',
        'version': '1.0',
        'endpoints': ['/predict', '/autocorrect', '/translate']
    })

@app.route('/health')
def health():
    """Check if API is working"""
    return jsonify({'status': 'healthy', 'ai_model': 'Gemini' if USE_GEMINI else 'Gemma'})

@app.route('/predict', methods=['POST'])
def predict_isl():
    """
    Receives video frame from frontend
    Returns: ISL text prediction
    """
    try:
        # Get image from frontend
        data = request.json
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'No image data provided'
            }), 400
        
        image_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64,
        
        # Convert to OpenCV image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({
                'success': False,
                'error': 'Invalid image data'
            }), 400
        
        # Get prediction from model
        prediction = isl_model.predict(frame)
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'confidence': 0.92  # Dummy confidence
        })
        
    except Exception as e:
        print(f"Error in /predict: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/autocorrect', methods=['POST'])
def autocorrect_text():
    """
    Uses AI to fix spelling/grammar mistakes in ISL output
    """
    try:
        data = request.json
        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'No text provided'
            }), 400
        
        text = data['text']
        
        if USE_GEMINI:
            # Use Gemini for autocorrection
            prompt = f"""You are an autocorrect system for Indian Sign Language translation.
Fix any spelling mistakes or grammar issues in this text.
Return ONLY the corrected text, nothing else. No explanations.

Text: {text}
Corrected:"""
            
            response = gemini_model.generate_content(prompt)
            corrected = response.text.strip()
        else:
            # Placeholder for Gemma
            corrected = text  # Just return original for now
        
        return jsonify({
            'success': True,
            'corrected_text': corrected,
            'original_text': text
        })
        
    except Exception as e:
        print(f"Error in /autocorrect: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/translate', methods=['POST'])
def translate_text():
    """
    Translates corrected English text to multiple Indian languages
    """
    try:
        data = request.json
        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'No text provided'
            }), 400
        
        text = data['text']
        target_lang = data.get('language', 'Hindi')
        
        if USE_GEMINI:
            # Use Gemini for translation
            prompt = f"""Translate this text to {target_lang}.
Return ONLY the translation in {target_lang} script, nothing else. No explanations.

Text: {text}
{target_lang} Translation:"""
            
            response = gemini_model.generate_content(prompt)
            translation = response.text.strip()
        else:
            # Placeholder for Gemma
            translation = f"[{target_lang} translation of: {text}]"
        
        return jsonify({
            'success': True,
            'translation': translation,
            'language': target_lang,
            'original_text': text
        })
        
    except Exception as e:
        print(f"Error in /translate: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("=" * 50)
    print("🚀 Gestura Backend Starting...")
    print("=" * 50)
    print(f"✅ Flask server: http://127.0.0.1:5000")
    print(f"✅ AI Model: {'Gemini' if USE_GEMINI else 'Gemma'}")
    print(f"✅ Endpoints: /predict, /autocorrect, /translate")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)