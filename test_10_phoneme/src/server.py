from flask import Flask, request, jsonify, send_file
import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from difflib import SequenceMatcher
import subprocess
import os

app = Flask(__name__, static_folder='static', static_url_path='')

# Load eSpeak wav2vec2 models
MODELS = {
    "wav2vec2_lv60": {
        "name": "Wav2Vec2 LV-60 eSpeak",
        "processor": None,
        "model": None
    },
    "wav2vec2_xlsr53": {
        "name": "Wav2Vec2 XLSR-53 eSpeak",
        "processor": None,
        "model": None
    }
}

try:
    MODELS["wav2vec2_lv60"]["processor"] = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
    MODELS["wav2vec2_lv60"]["model"] = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
    print("✓ Wav2Vec2 LV-60 eSpeak loaded")
except Exception as e:
    print(f"✗ Wav2Vec2 LV-60 load failed: {e}")

try:
    MODELS["wav2vec2_xlsr53"]["processor"] = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    MODELS["wav2vec2_xlsr53"]["model"] = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    print("✓ Wav2Vec2 XLSR-53 eSpeak loaded")
except Exception as e:
    print(f"✗ Wav2Vec2 XLSR-53 load failed: {e}")

# eSpeak to IPA mapping
ESPEAK_TO_IPA = {
    "eI": "eɪ",
    "aI": "aɪ",
    "aU": "aʊ",
    "OI": "ɔɪ",
    "O:": "ɔː",
    "o": "oʊ",
    "@": "ə",
    "3:": "ɜː",
    "I": "ɪ",
    "i:": "iː",
    "U": "ʊ",
    "u:": "uː",
    "A": "ʌ",
    "æ": "æ",
    "A:": "ɑː",
    "p": "p",
    "b": "b",
    "t": "t",
    "d": "d",
    "k": "k",
    "g": "ɡ",
    "m": "m",
    "n": "n",
    "N": "ŋ",
    "f": "f",
    "v": "v",
    "T": "θ",
    "D": "ð",
    "s": "s",
    "z": "z",
    "S": "ʃ",
    "Z": "ʒ",
    "tS": "tʃ",
    "dZ": "dʒ",
    "h": "h",
    "l": "l",
    "r": "ɹ",
    "j": "j",
    "w": "w",
}

# Phoneme mappings with eSpeak and IPA
WORDS = {
    "tomato": {
        "American": {"espeak": "t @ m 3: t @U", "ipa": "təˈmeɪtoʊ"},
        "British": {"espeak": "t @ m A: t @U", "ipa": "təˈmɑːtəʊ"}
    },
    "dance": {
        "American": {"espeak": "d æ n s", "ipa": "dæns"},
        "British": {"espeak": "d A: n s", "ipa": "dɑːns"}
    },
    "bath": {
        "American": {"espeak": "b æ T", "ipa": "bæθ"},
        "British": {"espeak": "b A: T", "ipa": "bɑːθ"}
    },
    "grass": {
        "American": {"espeak": "g r æ s", "ipa": "ɡræs"},
        "British": {"espeak": "g r A: s", "ipa": "ɡrɑːs"}
    },
    "lot": {
        "American": {"espeak": "l A t", "ipa": "lɑt"},
        "British": {"espeak": "l O t", "ipa": "lɒt"}
    },
    "cloth": {
        "American": {"espeak": "k l O T", "ipa": "klɔθ"},
        "British": {"espeak": "k l O T", "ipa": "klɒθ"}
    },
    "phone": {
        "American": {"espeak": "f @U n", "ipa": "foʊn"},
        "British": {"espeak": "f @U n", "ipa": "fəʊn"}
    },
    "schedule": {
        "American": {"espeak": "s k E dZ u: l", "ipa": "ˈskɛdʒuːl"},
        "British": {"espeak": "S E dZ u: l", "ipa": "ˈʃɛdʒuːl"}
    },
    "lever": {
        "American": {"espeak": "l E v @", "ipa": "ˈlɛvɚ"},
        "British": {"espeak": "l i: v @", "ipa": "ˈliːvə"}
    },
    "route": {
        "American": {"espeak": "r u: t", "ipa": "rut"},
        "British": {"espeak": "r aU t", "ipa": "raʊt"}
    },
    "go": {
        "American": {"espeak": "g @U", "ipa": "goʊ"},
        "British": {"espeak": "g @U", "ipa": "gəʊ"}
    },
    "home": {
        "American": {"espeak": "h @U m", "ipa": "hoʊm"},
        "British": {"espeak": "h @U m", "ipa": "həʊm"}
    },
    "boat": {
        "American": {"espeak": "b @U t", "ipa": "boʊt"},
        "British": {"espeak": "b @U t", "ipa": "bəʊt"}
    },
    "know": {
        "American": {"espeak": "n @U", "ipa": "noʊ"},
        "British": {"espeak": "n @U", "ipa": "nəʊ"}
    },
    "show": {
        "American": {"espeak": "S @U", "ipa": "ʃoʊ"},
        "British": {"espeak": "S @U", "ipa": "ʃəʊ"}
    },
    "low": {
        "American": {"espeak": "l @U", "ipa": "loʊ"},
        "British": {"espeak": "l @U", "ipa": "ləʊ"}
    },
    "road": {
        "American": {"espeak": "r @U d", "ipa": "roʊd"},
        "British": {"espeak": "r @U d", "ipa": "rəʊd"}
    },
    "coat": {
        "American": {"espeak": "k @U t", "ipa": "koʊt"},
        "British": {"espeak": "k @U t", "ipa": "kəʊt"}
    },
    "note": {
        "American": {"espeak": "n @U t", "ipa": "noʊt"},
        "British": {"espeak": "n @U t", "ipa": "nəʊt"}
    },
    "bone": {
        "American": {"espeak": "b @U n", "ipa": "boʊn"},
        "British": {"espeak": "b @U n", "ipa": "bəʊn"}
    },
    "tone": {
        "American": {"espeak": "t @U n", "ipa": "toʊn"},
        "British": {"espeak": "t @U n", "ipa": "təʊn"}
    },
}

def espeak_to_ipa(espeak_seq):
    """Convert eSpeak phonemes to IPA"""
    if not espeak_seq or espeak_seq == "N/A":
        return "N/A"
    
    tokens = espeak_seq.split()
    ipa_tokens = [ESPEAK_TO_IPA.get(t, t) for t in tokens]
    return "".join(ipa_tokens)

def calculate_score(detected, expected):
    """Calculate similarity score 0-100 using SequenceMatcher"""
    if expected == "N/A":
        return 0
    
    matcher = SequenceMatcher(None, detected.lower(), expected.lower())
    ratio = matcher.ratio()
    return int(ratio * 100)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/models')
def get_models():
    available = []
    for model_id, model_data in MODELS.items():
        if model_data["model"]:
            available.append({"id": model_id, "name": model_data["name"]})
    return jsonify(available)

@app.route('/tts', methods=['POST'])
def text_to_speech():
    """Generate speech audio using espeak"""
    data = request.get_json()
    text = data.get('text', '')
    accent = data.get('accent', 'American')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    # Map accent to espeak voice
    # American: en-us, British: en-gb
    voice = 'en-us' if accent == 'American' else 'en-gb'
    
    # Generate audio file
    audio_path = '/tmp/tts_output.wav'
    try:
        # Use espeak-ng to generate WAV file
        # -s: speed (words per minute), -g: gap between words (ms)
        # -v: voice, -w: output file
        subprocess.run([
            'espeak-ng',
            '-s', '150',  # Speed
            '-g', '5',    # Gap between words
            '-v', voice,
            '-w', audio_path,
            text
        ], check=True, capture_output=True)
        
        if not os.path.exists(audio_path):
            return jsonify({"error": "Audio generation failed"}), 500
        
        return send_file(audio_path, mimetype='audio/wav')
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"espeak error: {e.stderr.decode()}"}), 500
    except Exception as e:
        return jsonify({"error": f"TTS failed: {e}"}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file"}), 400
    
    audio_file = request.files['audio']
    accent = request.form.get('accent', 'American')
    word = request.form.get('word', '')
    model_id = request.form.get('model', 'wav2vec2_lv60')
    
    if model_id not in MODELS or not MODELS[model_id]["model"]:
        return jsonify({"error": "Model not available"}), 400
    
    audio_path = '/tmp/audio.wav'
    audio_file.save(audio_path)
    
    try:
        speech, sr = librosa.load(audio_path, sr=16000)
    except Exception as e:
        return jsonify({"error": f"Audio load failed: {e}"}), 400
    
    try:
        processor = MODELS[model_id]["processor"]
        model = MODELS[model_id]["model"]
        
        inputs = processor(speech, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
    except Exception as e:
        return jsonify({"error": f"Inference failed: {e}"}), 500
    
    phoneme_data = WORDS.get(word, {}).get(accent, {})
    expected_espeak = phoneme_data.get("espeak", "N/A")
    expected_ipa = phoneme_data.get("ipa", "N/A")
    detected_ipa = espeak_to_ipa(transcription)
    
    score = calculate_score(transcription, expected_espeak)
    
    return jsonify({
        "transcription": transcription,
        "detected_ipa": detected_ipa,
        "expected_espeak": expected_espeak,
        "expected_ipa": expected_ipa,
        "score": score,
        "match": score == 100
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
