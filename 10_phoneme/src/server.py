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
    # Japanese patterns - American
    "light": {"American": {"espeak": "l aI t", "ipa": "laɪt"}, "British": {"espeak": "l aI t", "ipa": "laɪt"}},
    "right": {"American": {"espeak": "r aI t", "ipa": "raɪt"}, "British": {"espeak": "r aI t", "ipa": "raɪt"}},
    "lead": {"American": {"espeak": "l i: d", "ipa": "liːd"}, "British": {"espeak": "l i: d", "ipa": "liːd"}},
    "read": {"American": {"espeak": "r i: d", "ipa": "riːd"}, "British": {"espeak": "r i: d", "ipa": "riːd"}},
    "long": {"American": {"espeak": "l O N", "ipa": "lɔŋ"}, "British": {"espeak": "l O N", "ipa": "lɒŋ"}},
    "wrong": {"American": {"espeak": "r O N", "ipa": "rɔŋ"}, "British": {"espeak": "r O N", "ipa": "rɒŋ"}},
    "play": {"American": {"espeak": "p l eI", "ipa": "pleɪ"}, "British": {"espeak": "p l eI", "ipa": "pleɪ"}},
    "pray": {"American": {"espeak": "p r eI", "ipa": "preɪ"}, "British": {"espeak": "p r eI", "ipa": "preɪ"}},
    "fly": {"American": {"espeak": "f l aI", "ipa": "flaɪ"}, "British": {"espeak": "f l aI", "ipa": "flaɪ"}},
    "fry": {"American": {"espeak": "f r aI", "ipa": "fraɪ"}, "British": {"espeak": "f r aI", "ipa": "fraɪ"}},
    "think": {"American": {"espeak": "T I N k", "ipa": "θɪŋk"}, "British": {"espeak": "T I N k", "ipa": "θɪŋk"}},
    "sink": {"American": {"espeak": "s I N k", "ipa": "sɪŋk"}, "British": {"espeak": "s I N k", "ipa": "sɪŋk"}},
    "thick": {"American": {"espeak": "T I k", "ipa": "θɪk"}, "British": {"espeak": "T I k", "ipa": "θɪk"}},
    "sick": {"American": {"espeak": "s I k", "ipa": "sɪk"}, "British": {"espeak": "s I k", "ipa": "sɪk"}},
    "thin": {"American": {"espeak": "T I n", "ipa": "θɪn"}, "British": {"espeak": "T I n", "ipa": "θɪn"}},
    "sin": {"American": {"espeak": "s I n", "ipa": "sɪn"}, "British": {"espeak": "s I n", "ipa": "sɪn"}},
    "thought": {"American": {"espeak": "T O t", "ipa": "θɔt"}, "British": {"espeak": "T O: t", "ipa": "θɔːt"}},
    "sought": {"American": {"espeak": "s O t", "ipa": "sɔt"}, "British": {"espeak": "s O: t", "ipa": "sɔːt"}},
    "three": {"American": {"espeak": "T r i:", "ipa": "θri"}, "British": {"espeak": "T r i:", "ipa": "θriː"}},
    "tree": {"American": {"espeak": "t r i:", "ipa": "tri"}, "British": {"espeak": "t r i:", "ipa": "triː"}},
    "very": {"American": {"espeak": "v E r i:", "ipa": "ˈvɛri"}, "British": {"espeak": "v E r i:", "ipa": "ˈvɛri"}},
    "berry": {"American": {"espeak": "b E r i:", "ipa": "ˈbɛri"}, "British": {"espeak": "b E r i:", "ipa": "ˈbɛri"}},
    "vote": {"American": {"espeak": "v @U t", "ipa": "voʊt"}, "British": {"espeak": "v @U t", "ipa": "vəʊt"}},
    "boat": {"American": {"espeak": "b @U t", "ipa": "boʊt"}, "British": {"espeak": "b @U t", "ipa": "bəʊt"}},
    "vest": {"American": {"espeak": "v E s t", "ipa": "vɛst"}, "British": {"espeak": "v E s t", "ipa": "vɛst"}},
    "best": {"American": {"espeak": "b E s t", "ipa": "bɛst"}, "British": {"espeak": "b E s t", "ipa": "bɛst"}},
    "vine": {"American": {"espeak": "v aI n", "ipa": "vaɪn"}, "British": {"espeak": "v aI n", "ipa": "vaɪn"}},
    "bine": {"American": {"espeak": "b aI n", "ipa": "baɪn"}, "British": {"espeak": "b aI n", "ipa": "baɪn"}},
    "veal": {"American": {"espeak": "v i: l", "ipa": "viːl"}, "British": {"espeak": "v i: l", "ipa": "viːl"}},
    "beal": {"American": {"espeak": "b i: l", "ipa": "biːl"}, "British": {"espeak": "b i: l", "ipa": "biːl"}},
    "cat": {"American": {"espeak": "k æ t", "ipa": "kæt"}, "British": {"espeak": "k æ t", "ipa": "kæt"}},
    "cot": {"American": {"espeak": "k A t", "ipa": "kɑt"}, "British": {"espeak": "k O t", "ipa": "kɒt"}},
    "hat": {"American": {"espeak": "h æ t", "ipa": "hæt"}, "British": {"espeak": "h æ t", "ipa": "hæt"}},
    "hot": {"American": {"espeak": "h A t", "ipa": "hɑt"}, "British": {"espeak": "h O t", "ipa": "hɒt"}},
    "bat": {"American": {"espeak": "b æ t", "ipa": "bæt"}, "British": {"espeak": "b æ t", "ipa": "bæt"}},
    "bot": {"American": {"espeak": "b A t", "ipa": "bɑt"}, "British": {"espeak": "b O t", "ipa": "bɒt"}},
    "sad": {"American": {"espeak": "s æ d", "ipa": "sæd"}, "British": {"espeak": "s æ d", "ipa": "sæd"}},
    "sod": {"American": {"espeak": "s A d", "ipa": "sɑd"}, "British": {"espeak": "s O d", "ipa": "sɒd"}},
    "bad": {"American": {"espeak": "b æ d", "ipa": "bæd"}, "British": {"espeak": "b æ d", "ipa": "bæd"}},
    "bod": {"American": {"espeak": "b A d", "ipa": "bɑd"}, "British": {"espeak": "b O d", "ipa": "bɒd"}},
    "bit": {"American": {"espeak": "b I t", "ipa": "bɪt"}, "British": {"espeak": "b I t", "ipa": "bɪt"}},
    "beat": {"American": {"espeak": "b i: t", "ipa": "biːt"}, "British": {"espeak": "b i: t", "ipa": "biːt"}},
    "sit": {"American": {"espeak": "s I t", "ipa": "sɪt"}, "British": {"espeak": "s I t", "ipa": "sɪt"}},
    "seat": {"American": {"espeak": "s i: t", "ipa": "siːt"}, "British": {"espeak": "s i: t", "ipa": "siːt"}},
    "ship": {"American": {"espeak": "S I p", "ipa": "ʃɪp"}, "British": {"espeak": "S I p", "ipa": "ʃɪp"}},
    "sheep": {"American": {"espeak": "S i: p", "ipa": "ʃiːp"}, "British": {"espeak": "S i: p", "ipa": "ʃiːp"}},
    "fit": {"American": {"espeak": "f I t", "ipa": "fɪt"}, "British": {"espeak": "f I t", "ipa": "fɪt"}},
    "feet": {"American": {"espeak": "f i: t", "ipa": "fiːt"}, "British": {"espeak": "f i: t", "ipa": "fiːt"}},
    "lip": {"American": {"espeak": "l I p", "ipa": "lɪp"}, "British": {"espeak": "l I p", "ipa": "lɪp"}},
    "leap": {"American": {"espeak": "l i: p", "ipa": "liːp"}, "British": {"espeak": "l i: p", "ipa": "liːp"}},
    "full": {"American": {"espeak": "f U l", "ipa": "fʊl"}, "British": {"espeak": "f U l", "ipa": "fʊl"}},
    "fool": {"American": {"espeak": "f u: l", "ipa": "fuːl"}, "British": {"espeak": "f u: l", "ipa": "fuːl"}},
    "pull": {"American": {"espeak": "p U l", "ipa": "pʊl"}, "British": {"espeak": "p U l", "ipa": "pʊl"}},
    "pool": {"American": {"espeak": "p u: l", "ipa": "puːl"}, "British": {"espeak": "p u: l", "ipa": "puːl"}},
    "wood": {"American": {"espeak": "w U d", "ipa": "wʊd"}, "British": {"espeak": "w U d", "ipa": "wʊd"}},
    "wooed": {"American": {"espeak": "w u: d", "ipa": "wuːd"}, "British": {"espeak": "w u: d", "ipa": "wuːd"}},
    "could": {"American": {"espeak": "k U d", "ipa": "kʊd"}, "British": {"espeak": "k U d", "ipa": "kʊd"}},
    "cooed": {"American": {"espeak": "k u: d", "ipa": "kuːd"}, "British": {"espeak": "k u: d", "ipa": "kuːd"}},
    "should": {"American": {"espeak": "S U d", "ipa": "ʃʊd"}, "British": {"espeak": "S U d", "ipa": "ʃʊd"}},
    "shoed": {"American": {"espeak": "S u: d", "ipa": "ʃuːd"}, "British": {"espeak": "S u: d", "ipa": "ʃuːd"}},
    "wait": {"American": {"espeak": "w eI t", "ipa": "weɪt"}, "British": {"espeak": "w eI t", "ipa": "weɪt"}},
    "wet": {"American": {"espeak": "w E t", "ipa": "wɛt"}, "British": {"espeak": "w E t", "ipa": "wɛt"}},
    "late": {"American": {"espeak": "l eI t", "ipa": "leɪt"}, "British": {"espeak": "l eI t", "ipa": "leɪt"}},
    "let": {"American": {"espeak": "l E t", "ipa": "lɛt"}, "British": {"espeak": "l E t", "ipa": "lɛt"}},
    "pain": {"American": {"espeak": "p eI n", "ipa": "peɪn"}, "British": {"espeak": "p eI n", "ipa": "peɪn"}},
    "pen": {"American": {"espeak": "p E n", "ipa": "pɛn"}, "British": {"espeak": "p E n", "ipa": "pɛn"}},
    "main": {"American": {"espeak": "m eI n", "ipa": "meɪn"}, "British": {"espeak": "m eI n", "ipa": "meɪn"}},
    "men": {"American": {"espeak": "m E n", "ipa": "mɛn"}, "British": {"espeak": "m E n", "ipa": "mɛn"}},
    "sail": {"American": {"espeak": "s eI l", "ipa": "seɪl"}, "British": {"espeak": "s eI l", "ipa": "seɪl"}},
    "sell": {"American": {"espeak": "s E l", "ipa": "sɛl"}, "British": {"espeak": "s E l", "ipa": "sɛl"}},
    "lit": {"American": {"espeak": "l I t", "ipa": "lɪt"}, "British": {"espeak": "l I t", "ipa": "lɪt"}},
    "rit": {"American": {"espeak": "r I t", "ipa": "rɪt"}, "British": {"espeak": "r I t", "ipa": "rɪt"}},
    "bite": {"American": {"espeak": "b aI t", "ipa": "baɪt"}, "British": {"espeak": "b aI t", "ipa": "baɪt"}},
    "sight": {"American": {"espeak": "s aI t", "ipa": "saɪt"}, "British": {"espeak": "s aI t", "ipa": "saɪt"}},
    "night": {"American": {"espeak": "n aI t", "ipa": "naɪt"}, "British": {"espeak": "n aI t", "ipa": "naɪt"}},
    "nit": {"American": {"espeak": "n I t", "ipa": "nɪt"}, "British": {"espeak": "n I t", "ipa": "nɪt"}},
    "now": {"American": {"espeak": "n aU", "ipa": "naʊ"}, "British": {"espeak": "n aU", "ipa": "naʊ"}},
    "no": {"American": {"espeak": "n @U", "ipa": "noʊ"}, "British": {"espeak": "n @U", "ipa": "nəʊ"}},
    "how": {"American": {"espeak": "h aU", "ipa": "haʊ"}, "British": {"espeak": "h aU", "ipa": "haʊ"}},
    "ho": {"American": {"espeak": "h @U", "ipa": "hoʊ"}, "British": {"espeak": "h @U", "ipa": "həʊ"}},
    "cow": {"American": {"espeak": "k aU", "ipa": "kaʊ"}, "British": {"espeak": "k aU", "ipa": "kaʊ"}},
    "co": {"American": {"espeak": "k @U", "ipa": "koʊ"}, "British": {"espeak": "k @U", "ipa": "kəʊ"}},
    "out": {"American": {"espeak": "@U t", "ipa": "aʊt"}, "British": {"espeak": "@U t", "ipa": "aʊt"}},
    "oat": {"American": {"espeak": "@U t", "ipa": "oʊt"}, "British": {"espeak": "@U t", "ipa": "əʊt"}},
    "loud": {"American": {"espeak": "l aU d", "ipa": "laʊd"}, "British": {"espeak": "l aU d", "ipa": "laʊd"}},
    "load": {"American": {"espeak": "l @U d", "ipa": "loʊd"}, "British": {"espeak": "l @U d", "ipa": "ləʊd"}},
    "water": {"American": {"espeak": "w O t @ r", "ipa": "ˈwɔtɚ"}, "British": {"espeak": "w O: t @", "ipa": "ˈwɔːtə"}},
    "wetter": {"American": {"espeak": "w E t @ r", "ipa": "ˈwɛtɚ"}, "British": {"espeak": "w E t @", "ipa": "ˈwɛtə"}},
    "better": {"American": {"espeak": "b E t @ r", "ipa": "ˈbɛtɚ"}, "British": {"espeak": "b E t @", "ipa": "ˈbɛtə"}},
    "betta": {"American": {"espeak": "b E t @", "ipa": "ˈbɛtə"}, "British": {"espeak": "b E t @", "ipa": "ˈbɛtə"}},
    "letter": {"American": {"espeak": "l E t @ r", "ipa": "ˈlɛtɚ"}, "British": {"espeak": "l E t @", "ipa": "ˈlɛtə"}},
    "letta": {"American": {"espeak": "l E t @", "ipa": "ˈlɛtə"}, "British": {"espeak": "l E t @", "ipa": "ˈlɛtə"}},
    "matter": {"American": {"espeak": "m æ t @ r", "ipa": "ˈmætɚ"}, "British": {"espeak": "m æ t @", "ipa": "ˈmætə"}},
    "matta": {"American": {"espeak": "m æ t @", "ipa": "ˈmætə"}, "British": {"espeak": "m æ t @", "ipa": "ˈmætə"}},
    "butter": {"American": {"espeak": "b A t @ r", "ipa": "ˈbʌtɚ"}, "British": {"espeak": "b A t @", "ipa": "ˈbʌtə"}},
    "butta": {"American": {"espeak": "b A t @", "ipa": "ˈbʌtə"}, "British": {"espeak": "b A t @", "ipa": "ˈbʌtə"}},
    "cart": {"American": {"espeak": "k A r t", "ipa": "kɑrt"}, "British": {"espeak": "k A: t", "ipa": "kɑːt"}},
    "heart": {"American": {"espeak": "h A r t", "ipa": "hɑrt"}, "British": {"espeak": "h A: t", "ipa": "hɑːt"}},
    "bart": {"American": {"espeak": "b A r t", "ipa": "bɑrt"}, "British": {"espeak": "b A: t", "ipa": "bɑːt"}},
    "sard": {"American": {"espeak": "s A r d", "ipa": "sɑrd"}, "British": {"espeak": "s A: d", "ipa": "sɑːd"}},
    "bard": {"American": {"espeak": "b A r d", "ipa": "bɑrd"}, "British": {"espeak": "b A: d", "ipa": "bɑːd"}},
    "hoe": {"American": {"espeak": "h @U", "ipa": "hoʊ"}, "British": {"espeak": "h @U", "ipa": "həʊ"}},
    "about": {"American": {"espeak": "@ b aU t", "ipa": "əˈbaʊt"}, "British": {"espeak": "@ b aU t", "ipa": "əˈbaʊt"}},
    "above": {"American": {"espeak": "@ b A v", "ipa": "əˈbʌv"}, "British": {"espeak": "@ b A v", "ipa": "əˈbʌv"}},
    "again": {"American": {"espeak": "@ g E n", "ipa": "əˈɡɛn"}, "British": {"espeak": "@ g E n", "ipa": "əˈɡɛn"}},
    "ago": {"American": {"espeak": "@ g @U", "ipa": "əˈɡoʊ"}, "British": {"espeak": "@ g @U", "ipa": "əˈɡəʊ"}},
    "away": {"American": {"espeak": "@ w eI", "ipa": "əˈweɪ"}, "British": {"espeak": "@ w eI", "ipa": "əˈweɪ"}},
    "banana": {"American": {"espeak": "b @ n æ n @", "ipa": "bəˈnænə"}, "British": {"espeak": "b @ n A: n @", "ipa": "bəˈnɑːnə"}},
    "camera": {"American": {"espeak": "k æ m @ r @", "ipa": "ˈkæmərə"}, "British": {"espeak": "k æ m @ r @", "ipa": "ˈkæmərə"}},
    "sofa": {"American": {"espeak": "s @U f @", "ipa": "ˈsoʊfə"}, "British": {"espeak": "s @U f @", "ipa": "ˈsəʊfə"}},
    "panda": {"American": {"espeak": "p æ n d @", "ipa": "ˈpændə"}, "British": {"espeak": "p æ n d @", "ipa": "ˈpændə"}},
    "zebra": {"American": {"espeak": "z i: b r @", "ipa": "ˈziːbrə"}, "British": {"espeak": "z E b r @", "ipa": "ˈzɛbrə"}},
    "help": {"American": {"espeak": "h E l p", "ipa": "hɛlp"}, "British": {"espeak": "h E l p", "ipa": "hɛlp"}},
    "hope": {"American": {"espeak": "h @U p", "ipa": "hoʊp"}, "British": {"espeak": "h @U p", "ipa": "həʊp"}},
    "hand": {"American": {"espeak": "h æ n d", "ipa": "hænd"}, "British": {"espeak": "h æ n d", "ipa": "hænd"}},
    "hear": {"American": {"espeak": "h I r", "ipa": "hɪr"}, "British": {"espeak": "h I @", "ipa": "hɪə"}},
    "high": {"American": {"espeak": "h aI", "ipa": "haɪ"}, "British": {"espeak": "h aI", "ipa": "haɪ"}},
    "huge": {"American": {"espeak": "h j u: dZ", "ipa": "hjuːdʒ"}, "British": {"espeak": "h j u: dZ", "ipa": "hjuːdʒ"}},
    "red": {"American": {"espeak": "r E d", "ipa": "rɛd"}, "British": {"espeak": "r E d", "ipa": "rɛd"}},
    "write": {"American": {"espeak": "r aI t", "ipa": "raɪt"}, "British": {"espeak": "r aI t", "ipa": "raɪt"}},
    "run": {"American": {"espeak": "r A n", "ipa": "rʌn"}, "British": {"espeak": "r A n", "ipa": "rʌn"}},
    "ran": {"American": {"espeak": "r æ n", "ipa": "ræn"}, "British": {"espeak": "r æ n", "ipa": "ræn"}},
    "rod": {"American": {"espeak": "r A d", "ipa": "rɑd"}, "British": {"espeak": "r O d", "ipa": "rɒd"}},
    "rain": {"American": {"espeak": "r eI n", "ipa": "reɪn"}, "British": {"espeak": "r eI n", "ipa": "reɪn"}},
    "record": {"American": {"espeak": "r E k @ r d", "ipa": "ˈrɛkɚd"}, "British": {"espeak": "r E k O: d", "ipa": "ˈrɛkɔːd"}},
    "present": {"American": {"espeak": "p r E z @ n t", "ipa": "ˈprɛzənt"}, "British": {"espeak": "p r E z @ n t", "ipa": "ˈprɛzənt"}},
    "object": {"American": {"espeak": "A b dZ E k t", "ipa": "ˈɑbdʒɛkt"}, "British": {"espeak": "O b dZ E k t", "ipa": "ˈɒbdʒɛkt"}},
    "project": {"American": {"espeak": "p r A dZ E k t", "ipa": "ˈprɑdʒɛkt"}, "British": {"espeak": "p r O dZ E k t", "ipa": "ˈprɒdʒɛkt"}},
    "permit": {"American": {"espeak": "p @ r m I t", "ipa": "pərˈmɪt"}, "British": {"espeak": "p @ m I t", "ipa": "pəˈmɪt"}},
    "produce": {"American": {"espeak": "p r @ d u: s", "ipa": "prəˈduːs"}, "British": {"espeak": "p r @ d j u: s", "ipa": "prəˈdjuːs"}},
    "import": {"American": {"espeak": "I m p O r t", "ipa": "ˈɪmpɔrt"}, "British": {"espeak": "I m p O: t", "ipa": "ˈɪmpɔːt"}},
    "export": {"American": {"espeak": "E k s p O r t", "ipa": "ˈɛkspɔrt"}, "British": {"espeak": "E k s p O: t", "ipa": "ˈɛkspɔːt"}},
    "contract": {"American": {"espeak": "k A n t r æ k t", "ipa": "ˈkɑntrækt"}, "British": {"espeak": "k O n t r æ k t", "ipa": "ˈkɒntrækt"}},
    "contest": {"American": {"espeak": "k A n t E s t", "ipa": "ˈkɑntɛst"}, "British": {"espeak": "k O n t E s t", "ipa": "ˈkɒntɛst"}},
    "sing": {"American": {"espeak": "s I N", "ipa": "sɪŋ"}, "British": {"espeak": "s I N", "ipa": "sɪŋ"}},
    "song": {"American": {"espeak": "s O N", "ipa": "sɔŋ"}, "British": {"espeak": "s O N", "ipa": "sɒŋ"}},
    "ring": {"American": {"espeak": "r I N", "ipa": "rɪŋ"}, "British": {"espeak": "r I N", "ipa": "rɪŋ"}},
    "strong": {"American": {"espeak": "s t r O N", "ipa": "strɔŋ"}, "British": {"espeak": "s t r O N", "ipa": "strɒŋ"}},
    "thing": {"American": {"espeak": "T I N", "ipa": "θɪŋ"}, "British": {"espeak": "T I N", "ipa": "θɪŋ"}},
    "bring": {"American": {"espeak": "b r I N", "ipa": "brɪŋ"}, "British": {"espeak": "b r I N", "ipa": "brɪŋ"}},
    "spring": {"American": {"espeak": "s p r I N", "ipa": "sprɪŋ"}, "British": {"espeak": "s p r I N", "ipa": "sprɪŋ"}},
    "string": {"American": {"espeak": "s t r I N", "ipa": "strɪŋ"}, "British": {"espeak": "s t r I N", "ipa": "strɪŋ"}},
    "judge": {"American": {"espeak": "dZ A dZ", "ipa": "dʒʌdʒ"}, "British": {"espeak": "dZ A dZ", "ipa": "dʒʌdʒ"}},
    "garage": {"American": {"espeak": "g @ r A: Z", "ipa": "ɡəˈrɑːʒ"}, "British": {"espeak": "g æ r A: Z", "ipa": "ɡæˈrɑːʒ"}},
    "age": {"American": {"espeak": "eI dZ", "ipa": "eɪdʒ"}, "British": {"espeak": "eI dZ", "ipa": "eɪdʒ"}},
    "beige": {"American": {"espeak": "b eI Z", "ipa": "beɪʒ"}, "British": {"espeak": "b eI Z", "ipa": "beɪʒ"}},
    "cage": {"American": {"espeak": "k eI dZ", "ipa": "keɪdʒ"}, "British": {"espeak": "k eI dZ", "ipa": "keɪdʒ"}},
    "massage": {"American": {"espeak": "m @ s A: Z", "ipa": "məˈsɑːʒ"}, "British": {"espeak": "m æ s A: Z", "ipa": "mæˈsɑːʒ"}},
    "page": {"American": {"espeak": "p eI dZ", "ipa": "peɪdʒ"}, "British": {"espeak": "p eI dZ", "ipa": "peɪdʒ"}},
    "rouge": {"American": {"espeak": "r u: Z", "ipa": "ruːʒ"}, "British": {"espeak": "r u: Z", "ipa": "ruːʒ"}},
    "stage": {"American": {"espeak": "s t eI dZ", "ipa": "steɪdʒ"}, "British": {"espeak": "s t eI dZ", "ipa": "steɪdʒ"}},
    "prestige": {"American": {"espeak": "p r E s t i: Z", "ipa": "prɛˈstiːʒ"}, "British": {"espeak": "p r E s t i: Z", "ipa": "prɛˈstiːʒ"}},
    "west": {"American": {"espeak": "w E s t", "ipa": "wɛst"}, "British": {"espeak": "w E s t", "ipa": "wɛst"}},
    "wine": {"American": {"espeak": "w aI n", "ipa": "waɪn"}, "British": {"espeak": "w aI n", "ipa": "waɪn"}},
    "worse": {"American": {"espeak": "w 3: r s", "ipa": "wɜrs"}, "British": {"espeak": "w 3: s", "ipa": "wɜːs"}},
    "verse": {"American": {"espeak": "v 3: r s", "ipa": "vɜrs"}, "British": {"espeak": "v 3: s", "ipa": "vɜːs"}},
    "wary": {"American": {"espeak": "w E r i:", "ipa": "ˈwɛri"}, "British": {"espeak": "w E@ r i:", "ipa": "ˈweəri"}},
    # Additional words for Japanese patterns
    "ket": {"American": {"espeak": "k E t", "ipa": "kɛt"}, "British": {"espeak": "k E t", "ipa": "kɛt"}},
    "het": {"American": {"espeak": "h E t", "ipa": "hɛt"}, "British": {"espeak": "h E t", "ipa": "hɛt"}},
    "bet": {"American": {"espeak": "b E t", "ipa": "bɛt"}, "British": {"espeak": "b E t", "ipa": "bɛt"}},
    "sed": {"American": {"espeak": "s E d", "ipa": "sɛd"}, "British": {"espeak": "s E d", "ipa": "sɛd"}},
    "bed": {"American": {"espeak": "b E d", "ipa": "bɛd"}, "British": {"espeak": "b E d", "ipa": "bɛd"}},
    "cut": {"American": {"espeak": "k A t", "ipa": "kʌt"}, "British": {"espeak": "k A t", "ipa": "kʌt"}},
    "hut": {"American": {"espeak": "h A t", "ipa": "hʌt"}, "British": {"espeak": "h A t", "ipa": "hʌt"}},
    "but": {"American": {"espeak": "b A t", "ipa": "bʌt"}, "British": {"espeak": "b A t", "ipa": "bʌt"}},
    "cup": {"American": {"espeak": "k A p", "ipa": "kʌp"}, "British": {"espeak": "k A p", "ipa": "kʌp"}},
    "cop": {"American": {"espeak": "k A p", "ipa": "kɑp"}, "British": {"espeak": "k O p", "ipa": "kɒp"}},
    "luck": {"American": {"espeak": "l A k", "ipa": "lʌk"}, "British": {"espeak": "l A k", "ipa": "lʌk"}},
    "lock": {"American": {"espeak": "l A k", "ipa": "lɑk"}, "British": {"espeak": "l O k", "ipa": "lɒk"}},
    "boy": {"American": {"espeak": "b OI", "ipa": "bɔɪ"}, "British": {"espeak": "b OI", "ipa": "bɔɪ"}},
    "toy": {"American": {"espeak": "t OI", "ipa": "tɔɪ"}, "British": {"espeak": "t OI", "ipa": "tɔɪ"}},
    "coin": {"American": {"espeak": "k OI n", "ipa": "kɔɪn"}, "British": {"espeak": "k OI n", "ipa": "kɔɪn"}},
    "join": {"American": {"espeak": "dZ OI n", "ipa": "dʒɔɪn"}, "British": {"espeak": "dZ OI n", "ipa": "dʒɔɪn"}},
    "voice": {"American": {"espeak": "v OI s", "ipa": "vɔɪs"}, "British": {"espeak": "v OI s", "ipa": "vɔɪs"}},
    "choice": {"American": {"espeak": "tS OI s", "ipa": "tʃɔɪs"}, "British": {"espeak": "tS OI s", "ipa": "tʃɔɪs"}},
    "noise": {"American": {"espeak": "n OI z", "ipa": "nɔɪz"}, "British": {"espeak": "n OI z", "ipa": "nɔɪz"}},
    "poise": {"American": {"espeak": "p OI z", "ipa": "pɔɪz"}, "British": {"espeak": "p OI z", "ipa": "pɔɪz"}},
    "joy": {"American": {"espeak": "dZ OI", "ipa": "dʒɔɪ"}, "British": {"espeak": "dZ OI", "ipa": "dʒɔɪ"}},
    "roy": {"American": {"espeak": "r OI", "ipa": "rɔɪ"}, "British": {"espeak": "r OI", "ipa": "rɔɪ"}},
    "bade": {"American": {"espeak": "b eI d", "ipa": "beɪd"}, "British": {"espeak": "b eI d", "ipa": "beɪd"}},
    "raid": {"American": {"espeak": "r eI d", "ipa": "reɪd"}, "British": {"espeak": "r eI d", "ipa": "reɪd"}},
    "mate": {"American": {"espeak": "m eI t", "ipa": "meɪt"}, "British": {"espeak": "m eI t", "ipa": "meɪt"}},
    "sate": {"American": {"espeak": "s eI t", "ipa": "seɪt"}, "British": {"espeak": "s eI t", "ipa": "seɪt"}},
    "pate": {"American": {"espeak": "p eI t", "ipa": "peɪt"}, "British": {"espeak": "p eI t", "ipa": "peɪt"}},
    "need": {"American": {"espeak": "n i: d", "ipa": "niːd"}, "British": {"espeak": "n i: d", "ipa": "niːd"}},
    "meet": {"American": {"espeak": "m i: t", "ipa": "miːt"}, "British": {"espeak": "m i: t", "ipa": "miːt"}},
    "see": {"American": {"espeak": "s i:", "ipa": "siː"}, "British": {"espeak": "s i:", "ipa": "siː"}},
    "put": {"American": {"espeak": "p U t", "ipa": "pʊt"}, "British": {"espeak": "p U t", "ipa": "pʊt"}},
    "book": {"American": {"espeak": "b U k", "ipa": "bʊk"}, "British": {"espeak": "b U k", "ipa": "bʊk"}},
    "look": {"American": {"espeak": "l U k", "ipa": "lʊk"}, "British": {"espeak": "l U k", "ipa": "lʊk"}},
    "took": {"American": {"espeak": "t U k", "ipa": "tʊk"}, "British": {"espeak": "t U k", "ipa": "tʊk"}},
    "good": {"American": {"espeak": "g U d", "ipa": "gʊd"}, "British": {"espeak": "g U d", "ipa": "gʊd"}},
    "food": {"American": {"espeak": "f u: d", "ipa": "fuːd"}, "British": {"espeak": "f u: d", "ipa": "fuːd"}},
    "mood": {"American": {"espeak": "m u: d", "ipa": "muːd"}, "British": {"espeak": "m u: d", "ipa": "muːd"}},
    "cool": {"American": {"espeak": "k u: l", "ipa": "kuːl"}, "British": {"espeak": "k u: l", "ipa": "kuːl"}},
    "tool": {"American": {"espeak": "t u: l", "ipa": "tuːl"}, "British": {"espeak": "t u: l", "ipa": "tuːl"}},
    "rule": {"American": {"espeak": "r u: l", "ipa": "ruːl"}, "British": {"espeak": "r u: l", "ipa": "ruːl"}},
    "day": {"American": {"espeak": "d eI", "ipa": "deɪ"}, "British": {"espeak": "d eI", "ipa": "deɪ"}},
    "say": {"American": {"espeak": "s eI", "ipa": "seɪ"}, "British": {"espeak": "s eI", "ipa": "seɪ"}},
    "way": {"American": {"espeak": "w eI", "ipa": "weɪ"}, "British": {"espeak": "w eI", "ipa": "weɪ"}},
    "stay": {"American": {"espeak": "s t eI", "ipa": "steɪ"}, "British": {"espeak": "s t eI", "ipa": "steɪ"}},
    "time": {"American": {"espeak": "t aI m", "ipa": "taɪm"}, "British": {"espeak": "t aI m", "ipa": "taɪm"}},
    "like": {"American": {"espeak": "l aI k", "ipa": "laɪk"}, "British": {"espeak": "l aI k", "ipa": "laɪk"}},
    "fine": {"American": {"espeak": "f aI n", "ipa": "faɪn"}, "British": {"espeak": "f aI n", "ipa": "faɪn"}},
    "line": {"American": {"espeak": "l aI n", "ipa": "laɪn"}, "British": {"espeak": "l aI n", "ipa": "laɪn"}},
    "mine": {"American": {"espeak": "m aI n", "ipa": "maɪn"}, "British": {"espeak": "m aI n", "ipa": "maɪn"}},
    "house": {"American": {"espeak": "h aU s", "ipa": "haʊs"}, "British": {"espeak": "h aU s", "ipa": "haʊs"}},
    "mouse": {"American": {"espeak": "m aU s", "ipa": "maʊs"}, "British": {"espeak": "m aU s", "ipa": "maʊs"}},
    "down": {"American": {"espeak": "d aU n", "ipa": "daʊn"}, "British": {"espeak": "d aU n", "ipa": "daʊn"}},
    "town": {"American": {"espeak": "t aU n", "ipa": "taʊn"}, "British": {"espeak": "t aU n", "ipa": "taʊn"}},
    "round": {"American": {"espeak": "r aU n d", "ipa": "raʊnd"}, "British": {"espeak": "r aU n d", "ipa": "raʊnd"}},
    "hit": {"American": {"espeak": "h I t", "ipa": "hɪt"}, "British": {"espeak": "h I t", "ipa": "hɪt"}},
    "win": {"American": {"espeak": "w I n", "ipa": "wɪn"}, "British": {"espeak": "w I n", "ipa": "wɪn"}},
    "pin": {"American": {"espeak": "p I n", "ipa": "pɪn"}, "British": {"espeak": "p I n", "ipa": "pɪn"}},
    "tin": {"American": {"espeak": "t I n", "ipa": "tɪn"}, "British": {"espeak": "t I n", "ipa": "tɪn"}},
    "get": {"American": {"espeak": "g E t", "ipa": "gɛt"}, "British": {"espeak": "g E t", "ipa": "gɛt"}},
    "net": {"American": {"espeak": "n E t", "ipa": "nɛt"}, "British": {"espeak": "n E t", "ipa": "nɛt"}},
    "bridge": {"American": {"espeak": "b r I dZ", "ipa": "brɪdʒ"}, "British": {"espeak": "b r I dZ", "ipa": "brɪdʒ"}},
    "edge": {"American": {"espeak": "E dZ", "ipa": "ɛdʒ"}, "British": {"espeak": "E dZ", "ipa": "ɛdʒ"}},
    "badge": {"American": {"espeak": "b { dZ", "ipa": "bædʒ"}, "British": {"espeak": "b { dZ", "ipa": "bædʒ"}},
    "fridge": {"American": {"espeak": "f r I dZ", "ipa": "frɪdʒ"}, "British": {"espeak": "f r I dZ", "ipa": "frɪdʒ"}},
    "hedge": {"American": {"espeak": "h E dZ", "ipa": "hɛdʒ"}, "British": {"espeak": "h E dZ", "ipa": "hɛdʒ"}},
    "web": {"American": {"espeak": "w E b", "ipa": "wɛb"}, "British": {"espeak": "w E b", "ipa": "wɛb"}},
    "wax": {"American": {"espeak": "w { k s", "ipa": "wæks"}, "British": {"espeak": "w { k s", "ipa": "wæks"}},
    "wish": {"American": {"espeak": "w I S", "ipa": "wɪʃ"}, "British": {"espeak": "w I S", "ipa": "wɪʃ"}},
    "wave": {"American": {"espeak": "w eI v", "ipa": "weɪv"}, "British": {"espeak": "w eI v", "ipa": "weɪv"}},
    "through": {"American": {"espeak": "T r u:", "ipa": "θruː"}, "British": {"espeak": "T r u:", "ipa": "θruː"}},
    "throw": {"American": {"espeak": "T r @U", "ipa": "θroʊ"}, "British": {"espeak": "T r @U", "ipa": "θrəʊ"}},
    "throat": {"American": {"espeak": "T r @U t", "ipa": "θroʊt"}, "British": {"espeak": "T r @U t", "ipa": "θrəʊt"}},
    "thrust": {"American": {"espeak": "T r A s t", "ipa": "θrʌst"}, "British": {"espeak": "T r A s t", "ipa": "θrʌst"}},
    "threat": {"American": {"espeak": "T r E t", "ipa": "θrɛt"}, "British": {"espeak": "T r E t", "ipa": "θrɛt"}},
}

# Pattern definitions organized by user_mode and accent
# Structure: PATTERN_SETS[user_mode][accent][pattern_id] = {name, description, words}

PATTERN_SETS = {
    "Native": {
        "American": {
            1: {"name": "/oʊ/ sound", "description": "GOAT vowel", "words": ["go", "know", "show", "home", "boat", "phone", "road", "coat", "note", "low"]},
            2: {"name": "/æ/ sound", "description": "TRAP vowel", "words": ["dance", "bath", "grass", "class", "path", "fast", "ask", "half", "laugh", "after"]},
            3: {"name": "/ɑ/ sound", "description": "LOT vowel", "words": ["lot", "hot", "not", "pot", "got", "cot", "nod", "rob", "stop", "top"]},
            4: {"name": "/ɔ/ sound", "description": "CLOTH vowel", "words": ["cloth", "off", "soft", "cross", "loss", "boss", "cost", "frost", "toss", "moss"]},
            5: {"name": "/ɑr/ sound", "description": "R-coloring - START", "words": ["car", "far", "bar", "star", "hard", "card", "park", "dark", "arm", "art"]},
            6: {"name": "/ɔr/ sound", "description": "R-coloring - NORTH", "words": ["more", "door", "floor", "store", "four", "pour", "warm", "corn", "born", "worn"]},
            7: {"name": "/ɪr/ sound", "description": "R-coloring - NEAR", "words": ["near", "here", "fear", "clear", "year", "ear", "beer", "dear", "tear", "sheer"]},
            8: {"name": "/ɛr/ sound", "description": "R-coloring - SQUARE", "words": ["air", "care", "share", "where", "hair", "fair", "square", "stare", "rare", "bear"]},
            9: {"name": "/ʊr/ sound", "description": "R-coloring - CURE", "words": ["tour", "poor", "sure", "cure", "pure", "lure", "endure", "mature", "secure", "obscure"]},
            10: {"name": "/ɜr/ sound", "description": "R-coloring - NURSE", "words": ["nurse", "bird", "word", "heard", "turn", "burn", "curse", "first", "third", "learn"]},
        },
        "British": {
            1: {"name": "/əʊ/ sound", "description": "GOAT vowel", "words": ["go", "know", "show", "home", "boat", "phone", "road", "coat", "note", "low"]},
            2: {"name": "/ɑː/ sound", "description": "BATH vowel", "words": ["dance", "bath", "grass", "class", "path", "fast", "ask", "half", "laugh", "after"]},
            3: {"name": "/ɒ/ sound", "description": "LOT vowel", "words": ["lot", "hot", "not", "pot", "got", "cot", "nod", "rob", "stop", "top"]},
            4: {"name": "/ɒ/ sound", "description": "CLOTH vowel", "words": ["cloth", "off", "soft", "cross", "loss", "boss", "cost", "frost", "toss", "moss"]},
            5: {"name": "/ɑː/ sound", "description": "START vowel", "words": ["car", "far", "bar", "star", "hard", "card", "park", "dark", "arm", "art"]},
            6: {"name": "/ɔː/ sound", "description": "NORTH vowel", "words": ["more", "door", "floor", "store", "four", "pour", "warm", "corn", "born", "worn"]},
            7: {"name": "/ɪə/ sound", "description": "NEAR vowel", "words": ["near", "here", "fear", "clear", "year", "ear", "beer", "dear", "tear", "sheer"]},
            8: {"name": "/eə/ sound", "description": "SQUARE vowel", "words": ["air", "care", "share", "where", "hair", "fair", "square", "stare", "rare", "bear"]},
            9: {"name": "/ʊə/ sound", "description": "CURE vowel", "words": ["tour", "poor", "sure", "cure", "pure", "lure", "endure", "mature", "secure", "obscure"]},
            10: {"name": "/ɜː/ sound", "description": "NURSE vowel", "words": ["nurse", "bird", "word", "heard", "turn", "burn", "curse", "first", "third", "learn"]},
        }
    },
    "Japanese": {
        "American": {
            1: {"name": "/æ/ sound", "description": "TRAP vowel", "words": ["cat", "hat", "bat", "sad", "bad", "dance", "fast", "ask", "half", "laugh"]},
            2: {"name": "/ɑ/ sound", "description": "LOT vowel", "words": ["lot", "hot", "not", "pot", "got", "cot", "nod", "rob", "stop", "top"]},
            3: {"name": "/ɪ/ sound", "description": "KIT vowel", "words": ["bit", "sit", "fit", "hit", "ship", "lip", "win", "pin", "tin", "sin"]},
            4: {"name": "/iː/ sound", "description": "FLEECE vowel", "words": ["beat", "seat", "feet", "meet", "sheep", "leap", "read", "lead", "need", "see"]},
            5: {"name": "/ʊ/ sound", "description": "FOOT vowel", "words": ["wood", "book", "look", "took", "good", "full", "pull", "could", "should", "put"]},
            6: {"name": "/uː/ sound", "description": "GOOSE vowel", "words": ["food", "mood", "cool", "tool", "rule", "fool", "pool", "wooed", "cooed", "shoed"]},
            7: {"name": "/eɪ/ sound", "description": "FACE diphthong", "words": ["day", "say", "way", "play", "stay", "wait", "late", "pain", "main", "sail"]},
            8: {"name": "/aɪ/ sound", "description": "PRICE diphthong", "words": ["light", "right", "sight", "night", "bite", "time", "like", "fine", "line", "mine"]},
            9: {"name": "/aʊ/ sound", "description": "MOUTH diphthong", "words": ["now", "how", "cow", "out", "loud", "house", "mouse", "down", "town", "round"]},
            10: {"name": "/ə/ sound", "description": "Schwa vowel", "words": ["about", "above", "again", "ago", "away", "banana", "camera", "sofa", "panda", "zebra"]},
        },
        "British": {
            1: {"name": "/æ/ sound", "description": "TRAP vowel", "words": ["cat", "hat", "bat", "sad", "bad", "dance", "fast", "ask", "half", "laugh"]},
            2: {"name": "/ɑː/ sound", "description": "BATH vowel", "words": ["dance", "bath", "grass", "class", "path", "fast", "ask", "half", "laugh", "after"]},
            3: {"name": "/ɪ/ sound", "description": "KIT vowel", "words": ["bit", "sit", "fit", "hit", "ship", "lip", "win", "pin", "tin", "sin"]},
            4: {"name": "/iː/ sound", "description": "FLEECE vowel", "words": ["beat", "seat", "feet", "meet", "sheep", "leap", "read", "lead", "need", "see"]},
            5: {"name": "/ʊ/ sound", "description": "FOOT vowel", "words": ["wood", "book", "look", "took", "good", "full", "pull", "could", "should", "put"]},
            6: {"name": "/uː/ sound", "description": "GOOSE vowel", "words": ["food", "mood", "cool", "tool", "rule", "fool", "pool", "wooed", "cooed", "shoed"]},
            7: {"name": "/eɪ/ sound", "description": "FACE diphthong", "words": ["day", "say", "way", "play", "stay", "wait", "late", "pain", "main", "sail"]},
            8: {"name": "/aɪ/ sound", "description": "PRICE diphthong", "words": ["light", "right", "sight", "night", "bite", "time", "like", "fine", "line", "mine"]},
            9: {"name": "/aʊ/ sound", "description": "MOUTH diphthong", "words": ["now", "how", "cow", "out", "loud", "house", "mouse", "down", "town", "round"]},
            10: {"name": "/ə/ sound", "description": "Schwa vowel", "words": ["about", "above", "again", "ago", "away", "banana", "camera", "sofa", "panda", "zebra"]},
        }
    },
    "French": {
        "American": {
            1: {"name": "/θ/ sound", "description": "TH pronunciation", "words": ["think", "thick", "thin", "thought", "three", "through", "throw", "throat", "thrust", "threat"]},
            2: {"name": "/h/ sound", "description": "H pronunciation", "words": ["hat", "hot", "help", "hope", "hand", "hard", "here", "hear", "high", "huge"]},
            3: {"name": "/r/ sound", "description": "R pronunciation", "words": ["red", "read", "right", "write", "run", "ran", "road", "rod", "rain", "ran"]},
            4: {"name": "/ɪ/ sound", "description": "KIT vowel", "words": ["bit", "sit", "fit", "hit", "ship", "lip", "win", "pin", "tin", "sin"]},
            5: {"name": "/iː/ sound", "description": "FLEECE vowel", "words": ["beat", "seat", "feet", "meet", "sheep", "leap", "read", "lead", "need", "see"]},
            6: {"name": "/ə/ sound", "description": "Schwa vowel", "words": ["about", "above", "again", "ago", "away", "banana", "camera", "sofa", "panda", "zebra"]},
            7: {"name": "/ɛ/ sound", "description": "DRESS vowel", "words": ["bed", "red", "met", "set", "pet", "wet", "let", "get", "bet", "net"]},
            8: {"name": "/ŋ/ sound", "description": "NG ending", "words": ["sing", "song", "ring", "wrong", "long", "strong", "thing", "bring", "spring", "string"]},
            9: {"name": "/dʒ/ sound", "description": "J pronunciation", "words": ["judge", "age", "cage", "page", "stage", "bridge", "edge", "badge", "fridge", "hedge"]},
            10: {"name": "/w/ sound", "description": "W pronunciation", "words": ["west", "wine", "wet", "worse", "wary", "win", "web", "wax", "wish", "wave"]},
        },
        "British": {
            1: {"name": "/θ/ sound", "description": "TH pronunciation", "words": ["think", "thick", "thin", "thought", "three", "through", "throw", "throat", "thrust", "threat"]},
            2: {"name": "/h/ sound", "description": "H pronunciation", "words": ["hat", "hot", "help", "hope", "hand", "hard", "here", "hear", "high", "huge"]},
            3: {"name": "/r/ sound", "description": "R pronunciation", "words": ["red", "read", "right", "write", "run", "ran", "road", "rod", "rain", "ran"]},
            4: {"name": "/ɪ/ sound", "description": "KIT vowel", "words": ["bit", "sit", "fit", "hit", "ship", "lip", "win", "pin", "tin", "sin"]},
            5: {"name": "/iː/ sound", "description": "FLEECE vowel", "words": ["beat", "seat", "feet", "meet", "sheep", "leap", "read", "lead", "need", "see"]},
            6: {"name": "/ə/ sound", "description": "Schwa vowel", "words": ["about", "above", "again", "ago", "away", "banana", "camera", "sofa", "panda", "zebra"]},
            7: {"name": "/ɛ/ sound", "description": "DRESS vowel", "words": ["bed", "red", "met", "set", "pet", "wet", "let", "get", "bet", "net"]},
            8: {"name": "/ŋ/ sound", "description": "NG ending", "words": ["sing", "song", "ring", "wrong", "long", "strong", "thing", "bring", "spring", "string"]},
            9: {"name": "/dʒ/ sound", "description": "J pronunciation", "words": ["judge", "age", "cage", "page", "stage", "bridge", "edge", "badge", "fridge", "hedge"]},
            10: {"name": "/w/ sound", "description": "W pronunciation", "words": ["west", "wine", "wet", "worse", "wary", "win", "web", "wax", "wish", "wave"]},
        }
    }
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

@app.route('/user-modes')
def get_user_modes():
    """Get available user modes"""
    return jsonify(["Native", "Japanese", "French"])

@app.route('/patterns')
def get_patterns():
    """Get patterns for a specific user mode and accent"""
    user_mode = request.args.get('user_mode', 'Native')
    accent = request.args.get('accent', 'American')
    
    if user_mode not in PATTERN_SETS or accent not in PATTERN_SETS[user_mode]:
        return jsonify({"error": "Invalid user_mode or accent"}), 400
    
    patterns = []
    for pattern_id in sorted(PATTERN_SETS[user_mode][accent].keys()):
        pattern_data = PATTERN_SETS[user_mode][accent][pattern_id]
        patterns.append({
            "id": pattern_id,
            "name": pattern_data["name"],
            "description": pattern_data["description"]
        })
    
    return jsonify(patterns)

@app.route('/pattern/<int:pattern_id>/words')
def get_pattern_words(pattern_id):
    """Get words for a specific pattern"""
    user_mode = request.args.get('user_mode', 'Native')
    accent = request.args.get('accent', 'American')
    
    if user_mode not in PATTERN_SETS or accent not in PATTERN_SETS[user_mode]:
        return jsonify({"error": "Invalid user_mode or accent"}), 400
    
    if pattern_id not in PATTERN_SETS[user_mode][accent]:
        return jsonify({"error": "Pattern not found"}), 404
    
    pattern_data = PATTERN_SETS[user_mode][accent][pattern_id]
    return jsonify({
        "pattern": {
            "id": pattern_id,
            "name": pattern_data["name"],
            "description": pattern_data["description"]
        },
        "words": pattern_data["words"]
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
