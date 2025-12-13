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
    "E": "ɛ",
    "e": "e",
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
# 10 vowel difference patterns, 10 examples each
WORDS = {
    # Pattern 1: /oʊ/ vs /əʊ/ (GOAT vowel)
    "go": {
        "American": {"espeak": "g @U", "ipa": "goʊ"},
        "British": {"espeak": "g @U", "ipa": "gəʊ"}
    },
    "know": {
        "American": {"espeak": "n @U", "ipa": "noʊ"},
        "British": {"espeak": "n @U", "ipa": "nəʊ"}
    },
    "show": {
        "American": {"espeak": "S @U", "ipa": "ʃoʊ"},
        "British": {"espeak": "S @U", "ipa": "ʃəʊ"}
    },
    "home": {
        "American": {"espeak": "h @U m", "ipa": "hoʊm"},
        "British": {"espeak": "h @U m", "ipa": "həʊm"}
    },
    "boat": {
        "American": {"espeak": "b @U t", "ipa": "boʊt"},
        "British": {"espeak": "b @U t", "ipa": "bəʊt"}
    },
    "phone": {
        "American": {"espeak": "f @U n", "ipa": "foʊn"},
        "British": {"espeak": "f @U n", "ipa": "fəʊn"}
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
    "low": {
        "American": {"espeak": "l @U", "ipa": "loʊ"},
        "British": {"espeak": "l @U", "ipa": "ləʊ"}
    },
    # Pattern 2: /æ/ vs /ɑː/ (TRAP-BATH split)
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
    "class": {
        "American": {"espeak": "k l æ s", "ipa": "klæs"},
        "British": {"espeak": "k l A: s", "ipa": "klɑːs"}
    },
    "path": {
        "American": {"espeak": "p æ T", "ipa": "pæθ"},
        "British": {"espeak": "p A: T", "ipa": "pɑːθ"}
    },
    "fast": {
        "American": {"espeak": "f æ s t", "ipa": "fæst"},
        "British": {"espeak": "f A: s t", "ipa": "fɑːst"}
    },
    "ask": {
        "American": {"espeak": "æ s k", "ipa": "æsk"},
        "British": {"espeak": "A: s k", "ipa": "ɑːsk"}
    },
    "half": {
        "American": {"espeak": "h æ f", "ipa": "hæf"},
        "British": {"espeak": "h A: f", "ipa": "hɑːf"}
    },
    "laugh": {
        "American": {"espeak": "l æ f", "ipa": "læf"},
        "British": {"espeak": "l A: f", "ipa": "lɑːf"}
    },
    "after": {
        "American": {"espeak": "æ f t @", "ipa": "ˈæftɚ"},
        "British": {"espeak": "A: f t @", "ipa": "ˈɑːftə"}
    },
    # Pattern 3: /ɑr/ vs /ɑː/ (R-coloring)
    "car": {
        "American": {"espeak": "k A r", "ipa": "kɑr"},
        "British": {"espeak": "k A:", "ipa": "kɑː"}
    },
    "far": {
        "American": {"espeak": "f A r", "ipa": "fɑr"},
        "British": {"espeak": "f A:", "ipa": "fɑː"}
    },
    "bar": {
        "American": {"espeak": "b A r", "ipa": "bɑr"},
        "British": {"espeak": "b A:", "ipa": "bɑː"}
    },
    "star": {
        "American": {"espeak": "s t A r", "ipa": "stɑr"},
        "British": {"espeak": "s t A:", "ipa": "stɑː"}
    },
    "hard": {
        "American": {"espeak": "h A r d", "ipa": "hɑrd"},
        "British": {"espeak": "h A: d", "ipa": "hɑːd"}
    },
    "card": {
        "American": {"espeak": "k A r d", "ipa": "kɑrd"},
        "British": {"espeak": "k A: d", "ipa": "kɑːd"}
    },
    "park": {
        "American": {"espeak": "p A r k", "ipa": "pɑrk"},
        "British": {"espeak": "p A: k", "ipa": "pɑːk"}
    },
    "dark": {
        "American": {"espeak": "d A r k", "ipa": "dɑrk"},
        "British": {"espeak": "d A: k", "ipa": "dɑːk"}
    },
    "arm": {
        "American": {"espeak": "A r m", "ipa": "ɑrm"},
        "British": {"espeak": "A: m", "ipa": "ɑːm"}
    },
    "art": {
        "American": {"espeak": "A r t", "ipa": "ɑrt"},
        "British": {"espeak": "A: t", "ipa": "ɑːt"}
    },
    # Pattern 4: /ɔr/ vs /ɔː/ (R-coloring)
    "more": {
        "American": {"espeak": "m O r", "ipa": "mɔr"},
        "British": {"espeak": "m O:", "ipa": "mɔː"}
    },
    "door": {
        "American": {"espeak": "d O r", "ipa": "dɔr"},
        "British": {"espeak": "d O:", "ipa": "dɔː"}
    },
    "floor": {
        "American": {"espeak": "f l O r", "ipa": "flɔr"},
        "British": {"espeak": "f l O:", "ipa": "flɔː"}
    },
    "store": {
        "American": {"espeak": "s t O r", "ipa": "stɔr"},
        "British": {"espeak": "s t O:", "ipa": "stɔː"}
    },
    "four": {
        "American": {"espeak": "f O r", "ipa": "fɔr"},
        "British": {"espeak": "f O:", "ipa": "fɔː"}
    },
    "pour": {
        "American": {"espeak": "p O r", "ipa": "pɔr"},
        "British": {"espeak": "p O:", "ipa": "pɔː"}
    },
    "warm": {
        "American": {"espeak": "w O r m", "ipa": "wɔrm"},
        "British": {"espeak": "w O: m", "ipa": "wɔːm"}
    },
    "corn": {
        "American": {"espeak": "k O r n", "ipa": "kɔrn"},
        "British": {"espeak": "k O: n", "ipa": "kɔːn"}
    },
    "born": {
        "American": {"espeak": "b O r n", "ipa": "bɔrn"},
        "British": {"espeak": "b O: n", "ipa": "bɔːn"}
    },
    "worn": {
        "American": {"espeak": "w O r n", "ipa": "wɔrn"},
        "British": {"espeak": "w O: n", "ipa": "wɔːn"}
    },
    # Pattern 5: /ɑ/ vs /ɒ/ (LOT vowel)
    "lot": {
        "American": {"espeak": "l A t", "ipa": "lɑt"},
        "British": {"espeak": "l O t", "ipa": "lɒt"}
    },
    "hot": {
        "American": {"espeak": "h A t", "ipa": "hɑt"},
        "British": {"espeak": "h O t", "ipa": "hɒt"}
    },
    "not": {
        "American": {"espeak": "n A t", "ipa": "nɑt"},
        "British": {"espeak": "n O t", "ipa": "nɒt"}
    },
    "pot": {
        "American": {"espeak": "p A t", "ipa": "pɑt"},
        "British": {"espeak": "p O t", "ipa": "pɒt"}
    },
    "got": {
        "American": {"espeak": "g A t", "ipa": "ɡɑt"},
        "British": {"espeak": "g O t", "ipa": "ɡɒt"}
    },
    "cot": {
        "American": {"espeak": "k A t", "ipa": "kɑt"},
        "British": {"espeak": "k O t", "ipa": "kɒt"}
    },
    "nod": {
        "American": {"espeak": "n A d", "ipa": "nɑd"},
        "British": {"espeak": "n O d", "ipa": "nɒd"}
    },
    "rob": {
        "American": {"espeak": "r A b", "ipa": "rɑb"},
        "British": {"espeak": "r O b", "ipa": "rɒb"}
    },
    "stop": {
        "American": {"espeak": "s t A p", "ipa": "stɑp"},
        "British": {"espeak": "s t O p", "ipa": "stɒp"}
    },
    "top": {
        "American": {"espeak": "t A p", "ipa": "tɑp"},
        "British": {"espeak": "t O p", "ipa": "tɒp"}
    },
    # Pattern 6: /ɔ/ vs /ɒ/ (CLOTH vowel)
    "cloth": {
        "American": {"espeak": "k l O T", "ipa": "klɔθ"},
        "British": {"espeak": "k l O T", "ipa": "klɒθ"}
    },
    "off": {
        "American": {"espeak": "O f", "ipa": "ɔf"},
        "British": {"espeak": "O f", "ipa": "ɒf"}
    },
    "soft": {
        "American": {"espeak": "s O f t", "ipa": "sɔft"},
        "British": {"espeak": "s O f t", "ipa": "sɒft"}
    },
    "cross": {
        "American": {"espeak": "k r O s", "ipa": "krɔs"},
        "British": {"espeak": "k r O s", "ipa": "krɒs"}
    },
    "loss": {
        "American": {"espeak": "l O s", "ipa": "lɔs"},
        "British": {"espeak": "l O s", "ipa": "lɒs"}
    },
    "boss": {
        "American": {"espeak": "b O s", "ipa": "bɔs"},
        "British": {"espeak": "b O s", "ipa": "bɒs"}
    },
    "cost": {
        "American": {"espeak": "k O s t", "ipa": "kɔst"},
        "British": {"espeak": "k O s t", "ipa": "kɒst"}
    },
    "frost": {
        "American": {"espeak": "f r O s t", "ipa": "frɔst"},
        "British": {"espeak": "f r O s t", "ipa": "frɒst"}
    },
    "toss": {
        "American": {"espeak": "t O s", "ipa": "tɔs"},
        "British": {"espeak": "t O s", "ipa": "tɒs"}
    },
    "moss": {
        "American": {"espeak": "m O s", "ipa": "mɔs"},
        "British": {"espeak": "m O s", "ipa": "mɒs"}
    },
    # Pattern 7: /ɪr/ vs /ɪə/ (NEAR - R-coloring)
    "near": {
        "American": {"espeak": "n I r", "ipa": "nɪr"},
        "British": {"espeak": "n I @", "ipa": "nɪə"}
    },
    "here": {
        "American": {"espeak": "h I r", "ipa": "hɪr"},
        "British": {"espeak": "h I @", "ipa": "hɪə"}
    },
    "fear": {
        "American": {"espeak": "f I r", "ipa": "fɪr"},
        "British": {"espeak": "f I @", "ipa": "fɪə"}
    },
    "clear": {
        "American": {"espeak": "k l I r", "ipa": "klɪr"},
        "British": {"espeak": "k l I @", "ipa": "klɪə"}
    },
    "year": {
        "American": {"espeak": "j I r", "ipa": "jɪr"},
        "British": {"espeak": "j I @", "ipa": "jɪə"}
    },
    "ear": {
        "American": {"espeak": "I r", "ipa": "ɪr"},
        "British": {"espeak": "I @", "ipa": "ɪə"}
    },
    "beer": {
        "American": {"espeak": "b I r", "ipa": "bɪr"},
        "British": {"espeak": "b I @", "ipa": "bɪə"}
    },
    "dear": {
        "American": {"espeak": "d I r", "ipa": "dɪr"},
        "British": {"espeak": "d I @", "ipa": "dɪə"}
    },
    "tear": {
        "American": {"espeak": "t I r", "ipa": "tɪr"},
        "British": {"espeak": "t I @", "ipa": "tɪə"}
    },
    "sheer": {
        "American": {"espeak": "S I r", "ipa": "ʃɪr"},
        "British": {"espeak": "S I @", "ipa": "ʃɪə"}
    },
    # Pattern 8: /ɛr/ vs /eə/ (SQUARE - R-coloring)
    "air": {
        "American": {"espeak": "E r", "ipa": "ɛr"},
        "British": {"espeak": "e @", "ipa": "eə"}
    },
    "care": {
        "American": {"espeak": "k E r", "ipa": "kɛr"},
        "British": {"espeak": "k e @", "ipa": "keə"}
    },
    "share": {
        "American": {"espeak": "S E r", "ipa": "ʃɛr"},
        "British": {"espeak": "S e @", "ipa": "ʃeə"}
    },
    "where": {
        "American": {"espeak": "w E r", "ipa": "wɛr"},
        "British": {"espeak": "w e @", "ipa": "weə"}
    },
    "hair": {
        "American": {"espeak": "h E r", "ipa": "hɛr"},
        "British": {"espeak": "h e @", "ipa": "heə"}
    },
    "fair": {
        "American": {"espeak": "f E r", "ipa": "fɛr"},
        "British": {"espeak": "f e @", "ipa": "feə"}
    },
    "square": {
        "American": {"espeak": "s k w E r", "ipa": "skwɛr"},
        "British": {"espeak": "s k w e @", "ipa": "skweə"}
    },
    "stare": {
        "American": {"espeak": "s t E r", "ipa": "stɛr"},
        "British": {"espeak": "s t e @", "ipa": "steə"}
    },
    "rare": {
        "American": {"espeak": "r E r", "ipa": "rɛr"},
        "British": {"espeak": "r e @", "ipa": "reə"}
    },
    "bear": {
        "American": {"espeak": "b E r", "ipa": "bɛr"},
        "British": {"espeak": "b e @", "ipa": "beə"}
    },
    # Pattern 9: /ʊr/ vs /ʊə/ (CURE - R-coloring)
    "tour": {
        "American": {"espeak": "t U r", "ipa": "tʊr"},
        "British": {"espeak": "t U @", "ipa": "tʊə"}
    },
    "poor": {
        "American": {"espeak": "p U r", "ipa": "pʊr"},
        "British": {"espeak": "p U @", "ipa": "pʊə"}
    },
    "sure": {
        "American": {"espeak": "S U r", "ipa": "ʃʊr"},
        "British": {"espeak": "S U @", "ipa": "ʃʊə"}
    },
    "cure": {
        "American": {"espeak": "k j U r", "ipa": "kjʊr"},
        "British": {"espeak": "k j U @", "ipa": "kjʊə"}
    },
    "pure": {
        "American": {"espeak": "p j U r", "ipa": "pjʊr"},
        "British": {"espeak": "p j U @", "ipa": "pjʊə"}
    },
    "lure": {
        "American": {"espeak": "l U r", "ipa": "lʊr"},
        "British": {"espeak": "l U @", "ipa": "lʊə"}
    },
    "endure": {
        "American": {"espeak": "E n d j U r", "ipa": "ɛnˈdjʊr"},
        "British": {"espeak": "E n d j U @", "ipa": "ɛnˈdjʊə"}
    },
    "mature": {
        "American": {"espeak": "m @ tS U r", "ipa": "məˈtʃʊr"},
        "British": {"espeak": "m @ tS U @", "ipa": "məˈtʃʊə"}
    },
    "secure": {
        "American": {"espeak": "s I k j U r", "ipa": "sɪˈkjʊr"},
        "British": {"espeak": "s I k j U @", "ipa": "sɪˈkjʊə"}
    },
    "obscure": {
        "American": {"espeak": "@ b s k j U r", "ipa": "əbˈskjʊr"},
        "British": {"espeak": "@ b s k j U @", "ipa": "əbˈskjʊə"}
    },
    # Pattern 10: /ɜr/ vs /ɜː/ (NURSE - R-coloring)
    "nurse": {
        "American": {"espeak": "n 3: r s", "ipa": "nɜrs"},
        "British": {"espeak": "n 3: s", "ipa": "nɜːs"}
    },
    "bird": {
        "American": {"espeak": "b 3: r d", "ipa": "bɜrd"},
        "British": {"espeak": "b 3: d", "ipa": "bɜːd"}
    },
    "word": {
        "American": {"espeak": "w 3: r d", "ipa": "wɜrd"},
        "British": {"espeak": "w 3: d", "ipa": "wɜːd"}
    },
    "heard": {
        "American": {"espeak": "h 3: r d", "ipa": "hɜrd"},
        "British": {"espeak": "h 3: d", "ipa": "hɜːd"}
    },
    "turn": {
        "American": {"espeak": "t 3: r n", "ipa": "tɜrn"},
        "British": {"espeak": "t 3: n", "ipa": "tɜːn"}
    },
    "burn": {
        "American": {"espeak": "b 3: r n", "ipa": "bɜrn"},
        "British": {"espeak": "b 3: n", "ipa": "bɜːn"}
    },
    "curse": {
        "American": {"espeak": "k 3: r s", "ipa": "kɜrs"},
        "British": {"espeak": "k 3: s", "ipa": "kɜːs"}
    },
    "first": {
        "American": {"espeak": "f 3: r s t", "ipa": "fɜrst"},
        "British": {"espeak": "f 3: s t", "ipa": "fɜːst"}
    },
    "third": {
        "American": {"espeak": "T 3: r d", "ipa": "θɜrd"},
        "British": {"espeak": "T 3: d", "ipa": "θɜːd"}
    },
    "learn": {
        "American": {"espeak": "l 3: r n", "ipa": "lɜrn"},
        "British": {"espeak": "l 3: n", "ipa": "lɜːn"}
    },
}

# Pattern definitions for UI display
PATTERNS = [
    {"id": 1, "name": "American /oʊ/ vs British /əʊ/", "description": "GOAT vowel"},
    {"id": 2, "name": "American /æ/ vs British /ɑː/", "description": "TRAP-BATH split"},
    {"id": 3, "name": "American /ɑr/ vs British /ɑː/", "description": "R-coloring"},
    {"id": 4, "name": "American /ɔr/ vs British /ɔː/", "description": "R-coloring"},
    {"id": 5, "name": "American /ɑ/ vs British /ɒ/", "description": "LOT vowel"},
    {"id": 6, "name": "American /ɔ/ vs British /ɒ/", "description": "CLOTH vowel"},
    {"id": 7, "name": "American /ɪr/ vs British /ɪə/", "description": "NEAR - R-coloring"},
    {"id": 8, "name": "American /ɛr/ vs British /eə/", "description": "SQUARE - R-coloring"},
    {"id": 9, "name": "American /ʊr/ vs British /ʊə/", "description": "CURE - R-coloring"},
    {"id": 10, "name": "American /ɜr/ vs British /ɜː/", "description": "NURSE - R-coloring"},
]

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

@app.route('/patterns')
def get_patterns():
    """Get all patterns"""
    return jsonify(PATTERNS)

@app.route('/pattern/<int:pattern_id>/words')
def get_pattern_words(pattern_id):
    """Get words for a specific pattern"""
    pattern_words = {
        1: ["go", "know", "show", "home", "boat", "phone", "road", "coat", "note", "low"],
        2: ["dance", "bath", "grass", "class", "path", "fast", "ask", "half", "laugh", "after"],
        3: ["car", "far", "bar", "star", "hard", "card", "park", "dark", "arm", "art"],
        4: ["more", "door", "floor", "store", "four", "pour", "warm", "corn", "born", "worn"],
        5: ["lot", "hot", "not", "pot", "got", "cot", "nod", "rob", "stop", "top"],
        6: ["cloth", "off", "soft", "cross", "loss", "boss", "cost", "frost", "toss", "moss"],
        7: ["near", "here", "fear", "clear", "year", "ear", "beer", "dear", "tear", "sheer"],
        8: ["air", "care", "share", "where", "hair", "fair", "square", "stare", "rare", "bear"],
        9: ["tour", "poor", "sure", "cure", "pure", "lure", "endure", "mature", "secure", "obscure"],
        10: ["nurse", "bird", "word", "heard", "turn", "burn", "curse", "first", "third", "learn"],
    }
    
    if pattern_id not in pattern_words:
        return jsonify({"error": "Pattern not found"}), 404
    
    pattern = PATTERNS[pattern_id - 1]
    return jsonify({
        "pattern": {
            "id": pattern_id,
            "name": pattern["name"],
            "description": pattern["description"]
        },
        "words": pattern_words[pattern_id]
    })

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
