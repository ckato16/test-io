from flask import Flask, request, jsonify, send_file
import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from difflib import SequenceMatcher
import subprocess
import os
import re

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

# Phonics data structure matching frontend
# Structure: PHONICS_DATA[level][category][sound_index] = {sound, ipa, es, words}
PHONICS_DATA = {
    "Basic": {
        "Single Consonants": [
            {"sound": "b", "ipa": "/b/", "es": {"en-GB": "b", "en-US": "b", "en-AU": "b", "en-IE": "b", "en-IN": "b", "en-CA": "b"}, "words": ["bat", "ball", "cab", "bed", "box"]},
            {"sound": "c", "ipa": "/k/", "es": {"en-GB": "k", "en-US": "k", "en-AU": "k", "en-IE": "k", "en-IN": "k", "en-CA": "k"}, "words": ["cat", "cup", "can", "car", "cut"]},
            {"sound": "d", "ipa": "/d/", "es": {"en-GB": "d", "en-US": "d", "en-AU": "d", "en-IE": "d", "en-IN": "d", "en-CA": "d"}, "words": ["dog", "red", "hand", "dad", "duck"]},
            {"sound": "f", "ipa": "/f/", "es": {"en-GB": "f", "en-US": "f", "en-AU": "f", "en-IE": "f", "en-IN": "f", "en-CA": "f"}, "words": ["fish", "fun", "leaf", "fan", "fox"]},
            {"sound": "g", "ipa": "/ɡ/", "es": {"en-GB": "g", "en-US": "g", "en-AU": "g", "en-IE": "g", "en-IN": "g", "en-CA": "g"}, "words": ["go", "big", "bag", "gun", "gum"]},
            {"sound": "h", "ipa": "/h/", "es": {"en-GB": "h", "en-US": "h", "en-AU": "h", "en-IE": "h", "en-IN": "h", "en-CA": "h"}, "words": ["hat", "hot", "hen", "hop", "hug"]},
            {"sound": "j", "ipa": "/dʒ/", "es": {"en-GB": "dZ", "en-US": "dZ", "en-AU": "dZ", "en-IE": "dZ", "en-IN": "dZ", "en-CA": "dZ"}, "words": ["jam", "jet", "jar", "jump", "joy"]},
            {"sound": "k", "ipa": "/k/", "es": {"en-GB": "k", "en-US": "k", "en-AU": "k", "en-IE": "k", "en-IN": "k", "en-CA": "k"}, "words": ["kite", "key", "book", "kid", "kit"]},
            {"sound": "l", "ipa": "/l/", "es": {"en-GB": "l", "en-US": "l", "en-AU": "l", "en-IE": "l", "en-IN": "l", "en-CA": "l"}, "words": ["lip", "leg", "ball", "log", "lap"]},
            {"sound": "m", "ipa": "/m/", "es": {"en-GB": "m", "en-US": "m", "en-AU": "m", "en-IE": "m", "en-IN": "m", "en-CA": "m"}, "words": ["man", "mum", "moon", "map", "mat"]},
            {"sound": "n", "ipa": "/n/", "es": {"en-GB": "n", "en-US": "n", "en-AU": "n", "en-IE": "n", "en-IN": "n", "en-CA": "n"}, "words": ["net", "sun", "pen", "nut", "nap"]},
            {"sound": "p", "ipa": "/p/", "es": {"en-GB": "p", "en-US": "p", "en-AU": "p", "en-IE": "p", "en-IN": "p", "en-CA": "p"}, "words": ["pen", "cap", "pig", "pan", "pot"]},
            {"sound": "r", "ipa": {"en-GB": "/ɑː/", "en-US": "/ɹ/", "en-AU": "/ɑː/", "en-IE": "/ɑː/", "en-IN": "/ɹ/", "en-CA": "/ɹ/"}, "es": {"en-GB": "A:", "en-US": "r", "en-AU": "A:", "en-IE": "A:", "en-IN": "r", "en-CA": "r"}, "words": ["red", "run", "car", "rat", "rug"]},
            {"sound": "s", "ipa": "/s/", "es": {"en-GB": "s", "en-US": "s", "en-AU": "s", "en-IE": "s", "en-IN": "s", "en-CA": "s"}, "words": ["sun", "sock", "bus", "sit", "sat"]},
            {"sound": "t", "ipa": "/t/", "es": {"en-GB": "t", "en-US": "t", "en-AU": "t", "en-IE": "t", "en-IN": "t", "en-CA": "t"}, "words": ["top", "cat", "ten", "tap", "tub"]},
            {"sound": "v", "ipa": "/v/", "es": {"en-GB": "v", "en-US": "v", "en-AU": "v", "en-IE": "v", "en-IN": "v", "en-CA": "v"}, "words": ["van", "vet", "five", "vine", "vote"]},
            {"sound": "w", "ipa": "/w/", "es": {"en-GB": "w", "en-US": "w", "en-AU": "w", "en-IE": "w", "en-IN": "w", "en-CA": "w"}, "words": ["wet", "win", "cow", "web", "wig"]},
            {"sound": "x", "ipa": "/ks/", "es": {"en-GB": "ks", "en-US": "ks", "en-AU": "ks", "en-IE": "ks", "en-IN": "ks", "en-CA": "ks"}, "words": ["box", "fox", "six", "ax", "mix"]},
            {"sound": "y", "ipa": "/j/", "es": {"en-GB": "j", "en-US": "j", "en-AU": "j", "en-IE": "j", "en-IN": "j", "en-CA": "j"}, "words": ["yes", "yellow", "yum", "yam", "yak"]},
            {"sound": "z", "ipa": "/z/", "es": {"en-GB": "z", "en-US": "z", "en-AU": "z", "en-IE": "z", "en-IN": "z", "en-CA": "z"}, "words": ["zip", "zoo", "buzz", "zap", "zen"]}
        ],
        "Consonant Digraphs": [
            {"sound": "ch", "ipa": "/tʃ/", "es": {"en-GB": "tS", "en-US": "tS", "en-AU": "tS", "en-IE": "tS", "en-IN": "tS", "en-CA": "tS"}, "words": ["chip", "chair", "lunch", "chat", "chin"]},
            {"sound": "ck", "ipa": "/k/", "es": {"en-GB": "k", "en-US": "k", "en-AU": "k", "en-IE": "k", "en-IN": "k", "en-CA": "k"}, "words": ["duck", "clock", "sock", "back", "pack"]},
            {"sound": "ng", "ipa": "/ŋ/", "es": {"en-GB": "N", "en-US": "N", "en-AU": "N", "en-IE": "N", "en-IN": "N", "en-CA": "N"}, "words": ["ring", "sing", "long", "song", "wing"]},
            {"sound": "nk", "ipa": "/ŋk/", "es": {"en-GB": "Nk", "en-US": "Nk", "en-AU": "Nk", "en-IE": "Nk", "en-IN": "Nk", "en-CA": "Nk"}, "words": ["bank", "sink", "think", "tank", "wink"]},
            {"sound": "qu", "ipa": "/kw/", "es": {"en-GB": "kw", "en-US": "kw", "en-AU": "kw", "en-IE": "kw", "en-IN": "kw", "en-CA": "kw"}, "words": ["queen", "quick", "quiz", "quilt", "quit"]},
            {"sound": "sh", "ipa": "/ʃ/", "es": {"en-GB": "S", "en-US": "S", "en-AU": "S", "en-IE": "S", "en-IN": "S", "en-CA": "S"}, "words": ["ship", "fish", "shop", "shell", "shut"]},
            {"sound": "th", "ipa": "/θ/ /ð/", "es": {"en-GB": "T @", "en-US": "T @", "en-AU": "T @", "en-IE": "T @", "en-IN": "T @", "en-CA": "T @"}, "words": ["thin", "this", "bath", "that", "then"]}
        ],
        "Single Vowels": [
            {"sound": "a", "ipa": "/æ/", "es": {"en-GB": "a", "en-US": "a", "en-AU": "a", "en-IE": "a", "en-IN": "a", "en-CA": "a"}, "words": ["ant", "apple", "cat", "hat", "map"]},
            {"sound": "e", "ipa": "/ɛ/", "es": {"en-GB": "E", "en-US": "E", "en-AU": "E", "en-IE": "E", "en-IN": "E", "en-CA": "E"}, "words": ["egg", "bed", "pen", "red", "net"]},
            {"sound": "i", "ipa": "/ɪ/", "es": {"en-GB": "I", "en-US": "I", "en-AU": "I", "en-IE": "I", "en-IN": "I", "en-CA": "I"}, "words": ["in", "sit", "pig", "pin", "hit"]},
            {"sound": "o", "ipa": "/ɒ/", "es": {"en-GB": "O", "en-US": "O", "en-AU": "O", "en-IE": "O", "en-IN": "O", "en-CA": "O"}, "words": ["on", "hot", "dog", "pot", "top"]},
            {"sound": "u", "ipa": "/ʌ/", "es": {"en-GB": "U", "en-US": "U", "en-AU": "U", "en-IE": "U", "en-IN": "U", "en-CA": "U"}, "words": ["up", "sun", "cup", "bug", "rug"]}
        ],
        "Vowel Digraphs": [
            {"sound": "ai", "ipa": "/eɪ/", "es": {"en-GB": "eI", "en-US": "eI", "en-AU": "eI", "en-IE": "eI", "en-IN": "eI", "en-CA": "eI"}, "words": ["snail", "rain", "train", "paint", "tail"]},
            {"sound": "au", "ipa": "/ɔː/", "es": {"en-GB": "O:", "en-US": "O:", "en-AU": "O:", "en-IE": "O:", "en-IN": "O:", "en-CA": "O:"}, "words": ["haul", "cause", "pause", "fault", "launch"]},
            {"sound": "aw", "ipa": {"en-GB": "/ɔː/", "en-US": "/ɔ/", "en-AU": "/ɔː/", "en-IE": "/ɔː/", "en-IN": "/ɔ/", "en-CA": "/ɔ/"}, "es": {"en-GB": "O:", "en-US": "O", "en-AU": "O:", "en-IE": "O:", "en-IN": "O", "en-CA": "O"}, "words": ["yawn", "paw", "claw", "draw", "straw"]},
            {"sound": "ay", "ipa": "/eɪ/", "es": {"en-GB": "eI", "en-US": "eI", "en-AU": "eI", "en-IE": "eI", "en-IN": "eI", "en-CA": "eI"}, "words": ["day", "play", "stay", "say", "way"]},
            {"sound": "ea", "ipa": "/iː/", "es": {"en-GB": "i:", "en-US": "i:", "en-AU": "i:", "en-IE": "i:", "en-IN": "i:", "en-CA": "i:"}, "words": ["tea", "meat", "seat", "beach", "leaf"]},
            {"sound": "ea", "ipa": "/ɛ/", "es": {"en-GB": "E", "en-US": "E", "en-AU": "E", "en-IE": "E", "en-IN": "E", "en-CA": "E"}, "words": ["bread", "weather", "head", "dead", "spread"]},
            {"sound": "ee", "ipa": "/iː/", "es": {"en-GB": "i:", "en-US": "i:", "en-AU": "i:", "en-IE": "i:", "en-IN": "i:", "en-CA": "i:"}, "words": ["see", "tree", "green", "bee", "seed"]},
            {"sound": "ew", "ipa": "/uː/", "es": {"en-GB": "u:", "en-US": "u:", "en-AU": "u:", "en-IE": "u:", "en-IN": "u:", "en-CA": "u:"}, "words": ["chew", "few", "new", "dew", "flew"]},
            {"sound": "ie", "ipa": "/aɪ/", "es": {"en-GB": "aI", "en-US": "aI", "en-AU": "aI", "en-IE": "aI", "en-IN": "aI", "en-CA": "aI"}, "words": ["tie", "pie", "die", "lie", "cried"]},
            {"sound": "oa", "ipa": "/əʊ/", "es": {"en-GB": "@U", "en-US": "@U", "en-AU": "@U", "en-IE": "@U", "en-IN": "@U", "en-CA": "@U"}, "words": ["goat", "road", "boat", "coat", "load"]},
            {"sound": "oi", "ipa": "/ɔɪ/", "es": {"en-GB": "OI", "en-US": "OI", "en-AU": "OI", "en-IE": "OI", "en-IN": "OI", "en-CA": "OI"}, "words": ["spoil", "coin", "noise", "join", "point"]},
            {"sound": "oo", "ipa": "/uː/", "es": {"en-GB": "u:", "en-US": "u:", "en-AU": "u:", "en-IE": "u:", "en-IN": "u:", "en-CA": "u:"}, "words": ["zoo", "moon", "food", "room", "cool"]},
            {"sound": "oo", "ipa": "/ʊ/", "es": {"en-GB": "U", "en-US": "U", "en-AU": "U", "en-IE": "U", "en-IN": "U", "en-CA": "U"}, "words": ["book", "look", "foot", "good", "wood"]},
            {"sound": "ou", "ipa": "/aʊ/", "es": {"en-GB": "aU", "en-US": "aU", "en-AU": "aU", "en-IE": "aU", "en-IN": "aU", "en-CA": "aU"}, "words": ["out", "shout", "cloud", "house", "mouse"]},
            {"sound": "ow", "ipa": {"en-GB": "/əʊ/", "en-US": "/aʊ/", "en-AU": "/əʊ/", "en-IE": "/əʊ/", "en-IN": "/aʊ/", "en-CA": "/aʊ/"}, "es": {"en-GB": "@U", "en-US": "aU", "en-AU": "@U", "en-IE": "@U", "en-IN": "aU", "en-CA": "aU"}, "words": ["brown", "cow", "down", "town", "how"]},
            {"sound": "oy", "ipa": "/ɔɪ/", "es": {"en-GB": "OI", "en-US": "OI", "en-AU": "OI", "en-IE": "OI", "en-IN": "OI", "en-CA": "OI"}, "words": ["boy", "toy", "coin", "joy", "soy"]}
        ],
        "Trigraphs": [
            {"sound": "igh", "ipa": "/aɪ/", "es": {"en-GB": "aI", "en-US": "aI", "en-AU": "aI", "en-IE": "aI", "en-IN": "aI", "en-CA": "aI"}, "words": ["high", "night", "light", "right", "bright"]}
        ],
        "R-controlled Vowels": [
            {"sound": "air", "ipa": {"en-GB": "/ɛə/", "en-US": "/ɛɹ/", "en-AU": "/ɛə/", "en-IE": "/ɛə/", "en-IN": "/ɛɹ/", "en-CA": "/ɛɹ/"}, "es": {"en-GB": "e@", "en-US": "e r", "en-AU": "e@", "en-IE": "e@", "en-IN": "e r", "en-CA": "e r"}, "words": ["hair", "fair", "chair", "pair", "stair"]},
            {"sound": "ar", "ipa": {"en-GB": "/ɑː/", "en-US": "/ɑɹ/", "en-AU": "/ɑː/", "en-IE": "/ɑː/", "en-IN": "/ɑɹ/", "en-CA": "/ɑɹ/"}, "es": {"en-GB": "A:", "en-US": "A r", "en-AU": "A:", "en-IE": "A:", "en-IN": "A r", "en-CA": "A r"}, "words": ["car", "star", "farm", "park", "hard"]},
            {"sound": "are", "ipa": {"en-GB": "/ɛə/", "en-US": "/ɛɹ/", "en-AU": "/ɛə/", "en-IE": "/ɛə/", "en-IN": "/ɛɹ/", "en-CA": "/ɛɹ/"}, "es": {"en-GB": "e@", "en-US": "e r", "en-AU": "e@", "en-IE": "e@", "en-IN": "e r", "en-CA": "e r"}, "words": ["care", "share", "bare", "dare", "scare"]},
            {"sound": "ear", "ipa": {"en-GB": "/ɪə/", "en-US": "/ɪɹ/", "en-AU": "/ɪə/", "en-IE": "/ɪə/", "en-IN": "/ɪɹ/", "en-CA": "/ɪɹ/"}, "es": {"en-GB": "i@", "en-US": "i r", "en-AU": "i@", "en-IE": "i@", "en-IN": "i r", "en-CA": "i r"}, "words": ["hear", "near", "dear", "fear", "clear"]},
            {"sound": "er", "ipa": {"en-GB": "/ə/", "en-US": "/ɹ/", "en-AU": "/ə/", "en-IE": "/ə/", "en-IN": "/ɹ/", "en-CA": "/ɹ/"}, "es": {"en-GB": "@", "en-US": "r", "en-AU": "@", "en-IE": "@", "en-IN": "r", "en-CA": "r"}, "words": ["letter", "better", "summer", "water", "matter"]},
            {"sound": "ir", "ipa": {"en-GB": "/ɜː/", "en-US": "/ɜɹ/", "en-AU": "/ɜː/", "en-IE": "/ɜː/", "en-IN": "/ɜɹ/", "en-CA": "/ɜɹ/"}, "es": {"en-GB": "3:", "en-US": "3 r", "en-AU": "3:", "en-IE": "3:", "en-IN": "3 r", "en-CA": "3 r"}, "words": ["bird", "girl", "shirt", "dirt", "third"]},
            {"sound": "ire", "ipa": {"en-GB": "/aɪə/", "en-US": "/aɪɹ/", "en-AU": "/aɪə/", "en-IE": "/aɪə/", "en-IN": "/aɪɹ/", "en-CA": "/aɪɹ/"}, "es": {"en-GB": "aI@", "en-US": "aI r", "en-AU": "aI@", "en-IE": "aI@", "en-IN": "aI r", "en-CA": "aI r"}, "words": ["fire", "wire", "tire", "hire", "sire"]},
            {"sound": "or", "ipa": {"en-GB": "/ɔː/", "en-US": "/ɔɹ/", "en-AU": "/ɔː/", "en-IE": "/ɔː/", "en-IN": "/ɔɹ/", "en-CA": "/ɔɹ/"}, "es": {"en-GB": "O:", "en-US": "O r", "en-AU": "O:", "en-IE": "O:", "en-IN": "O r", "en-CA": "O r"}, "words": ["fork", "corn", "short", "born", "storm"]},
            {"sound": "ur", "ipa": {"en-GB": "/ɜː/", "en-US": "/ɜɹ/", "en-AU": "/ɜː/", "en-IE": "/ɜː/", "en-IN": "/ɜɹ/", "en-CA": "/ɜɹ/"}, "es": {"en-GB": "3:", "en-US": "3 r", "en-AU": "3:", "en-IE": "3:", "en-IN": "3 r", "en-CA": "3 r"}, "words": ["nurse", "turn", "hurt", "burn", "curb"]},
            {"sound": "ure", "ipa": {"en-GB": "/ɔə/", "en-US": "/ɔɹ/", "en-AU": "/ɔə/", "en-IE": "/ɔə/", "en-IN": "/ɔɹ/", "en-CA": "/ɔɹ/"}, "es": {"en-GB": "O@", "en-US": "O r", "en-AU": "O@", "en-IE": "O@", "en-IN": "O r", "en-CA": "O r"}, "words": ["pure", "cure", "sure", "lure", "endure"]}
        ],
        "Silent E Patterns": [
            {"sound": "a_e", "ipa": "/eɪ/", "es": {"en-GB": "eI", "en-US": "eI", "en-AU": "eI", "en-IE": "eI", "en-IN": "eI", "en-CA": "eI"}, "words": ["cake", "name", "gate", "lake", "make"]},
            {"sound": "i_e", "ipa": "/aɪ/", "es": {"en-GB": "aI", "en-US": "aI", "en-AU": "aI", "en-IE": "aI", "en-IN": "aI", "en-CA": "aI"}, "words": ["smile", "time", "bike", "like", "five"]},
            {"sound": "o_e", "ipa": "/oʊ/", "es": {"en-GB": "oU", "en-US": "oU", "en-AU": "oU", "en-IE": "oU", "en-IN": "oU", "en-CA": "oU"}, "words": ["home", "bone", "rose", "hope", "note"]},
            {"sound": "u_e", "ipa": "/uː/", "es": {"en-GB": "u:", "en-US": "u:", "en-AU": "u:", "en-IE": "u:", "en-IN": "u:", "en-CA": "u:"}, "words": ["huge", "cube", "tune", "mule", "fuse"]}
        ]
    },
    "Advanced": {
        "Soft C & G": [
            {"sound": "c", "ipa": "/s/", "es": {"en-GB": "s", "en-US": "s", "en-AU": "s", "en-IE": "s", "en-IN": "s", "en-CA": "s"}, "words": ["city", "cent", "circle", "cease", "cycle"]},
            {"sound": "g", "ipa": "/dʒ/", "es": {"en-GB": "dZ", "en-US": "dZ", "en-AU": "dZ", "en-IE": "dZ", "en-IN": "dZ", "en-CA": "dZ"}, "words": ["gym", "gem", "giant", "giraffe", "gist"]}
        ],
        "Y as Vowel": [
            {"sound": "y", "ipa": "/aɪ/", "es": {"en-GB": "aI", "en-US": "aI", "en-AU": "aI", "en-IE": "aI", "en-IN": "aI", "en-CA": "aI"}, "words": ["my", "cry", "sky", "fly", "dry"]},
            {"sound": "y", "ipa": "/ɪ/", "es": {"en-GB": "I", "en-US": "I", "en-AU": "I", "en-IE": "I", "en-IN": "I", "en-CA": "I"}, "words": ["happy", "baby", "lady", "candy", "funny"]}
        ],
        "Silent Letters": [
            {"sound": "gn", "ipa": "/n/", "es": {"en-GB": "n", "en-US": "n", "en-AU": "n", "en-IE": "n", "en-IN": "n", "en-CA": "n"}, "words": ["gnat", "gnome", "sign", "design", "align"]},
            {"sound": "kn", "ipa": "/n/", "es": {"en-GB": "n", "en-US": "n", "en-AU": "n", "en-IE": "n", "en-IN": "n", "en-CA": "n"}, "words": ["knee", "know", "knife", "knock", "knight"]},
            {"sound": "ph", "ipa": "/f/", "es": {"en-GB": "f", "en-US": "f", "en-AU": "f", "en-IE": "f", "en-IN": "f", "en-CA": "f"}, "words": ["phone", "photo", "graph", "elephant", "phrase"]},
            {"sound": "wr", "ipa": "/r/", "es": {"en-GB": "r", "en-US": "r", "en-AU": "r", "en-IE": "r", "en-IN": "r", "en-CA": "r"}, "words": ["write", "wrong", "wrap", "wrist", "wreck"]}
        ],
        "Quadgraphs": [
            {"sound": "augh", "ipa": "/ɔ/", "es": {"en-GB": "O", "en-US": "O", "en-AU": "O", "en-IE": "O", "en-IN": "O", "en-CA": "O"}, "words": ["caught", "taught", "daughter", "naughty", "fraught"]},
            {"sound": "eigh", "ipa": "/eɪ/", "es": {"en-GB": "eI", "en-US": "eI", "en-AU": "eI", "en-IE": "eI", "en-IN": "eI", "en-CA": "eI"}, "words": ["eight", "weight", "neighbor", "sleigh", "weigh"]},
            {"sound": "ough", "ipa": "/ʌ/", "es": {"en-GB": "U", "en-US": "U", "en-AU": "U", "en-IE": "U", "en-IN": "U", "en-CA": "U"}, "words": ["tough", "rough", "enough", "trough", "slough"]},
            {"sound": "ough", "ipa": "/uː/", "es": {"en-GB": "u:", "en-US": "u:", "en-AU": "u:", "en-IE": "u:", "en-IN": "u:", "en-CA": "u:"}, "words": ["through", "who", "threw", "flew", "grew"]},
            {"sound": "ough", "ipa": "/oʊ/", "es": {"en-GB": "oU", "en-US": "oU", "en-AU": "oU", "en-IE": "oU", "en-IN": "oU", "en-CA": "oU"}, "words": ["though", "dough", "although", "thorough", "borough"]},
            {"sound": "ough", "ipa": "/aʊ/", "es": {"en-GB": "aU", "en-US": "aU", "en-AU": "aU", "en-IE": "aU", "en-IN": "aU", "en-CA": "aU"}, "words": ["bough", "plough", "drought", "slough", "brought"]}
        ],
        "W/Qu-modified Vowels": [
            {"sound": "war", "ipa": {"en-GB": "/ɔː/", "en-US": "/ɔɹ/", "en-AU": "/ɔː/", "en-IE": "/ɔː/", "en-IN": "/ɔɹ/", "en-CA": "/ɔɹ/"}, "es": {"en-GB": "O:", "en-US": "O r", "en-AU": "O:", "en-IE": "O:", "en-IN": "O r", "en-CA": "O r"}, "words": ["war", "warm", "warn", "ward", "wart"]},
            {"sound": "quar", "ipa": {"en-GB": "/ɔː/", "en-US": "/ɔɹ/", "en-AU": "/ɔː/", "en-IE": "/ɔː/", "en-IN": "/ɔɹ/", "en-CA": "/ɔɹ/"}, "es": {"en-GB": "O:", "en-US": "O r", "en-AU": "O:", "en-IE": "O:", "en-IN": "O r", "en-CA": "O r"}, "words": ["quarter", "quartz", "quart", "quarry", "squash"]},
            {"sound": "wor", "ipa": {"en-GB": "/ɜː/", "en-US": "/ɜɹ/", "en-AU": "/ɜː/", "en-IE": "/ɜː/", "en-IN": "/ɜɹ/", "en-CA": "/ɜɹ/"}, "es": {"en-GB": "3:", "en-US": "3 r", "en-AU": "3:", "en-IE": "3:", "en-IN": "3 r", "en-CA": "3 r"}, "words": ["word", "work", "worm", "worth", "world"]},
            {"sound": "wa", "ipa": {"en-GB": "/ɒ/", "en-US": "/ɑ/", "en-AU": "/ɒ/", "en-IE": "/ɒ/", "en-IN": "/ɑ/", "en-CA": "/ɑ/"}, "es": {"en-GB": "O", "en-US": "A", "en-AU": "O", "en-IE": "O", "en-IN": "A", "en-CA": "A"}, "words": ["want", "wash", "watch", "water", "was"]}
        ]
    }
}

# Map frontend accent codes to backend accent names
ACCENT_MAP = {
    "en-GB": "British",
    "en-US": "American",
    "en-AU": "Australian",
    "en-IE": "Irish",
    "en-IN": "Indian",
    "en-CA": "Canadian"
}

def get_value_for_accent(data, accent_code):
    """Get value for accent, with fallback"""
    if isinstance(data, dict):
        return data.get(accent_code) or data.get("en-US") or list(data.values())[0]
    return data

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

def get_espeak_phonemes_for_word(word, accent_code):
    """Get eSpeak phonemes for a word using espeak-ng"""
    accent_map = {
        "en-GB": "en-gb",
        "en-US": "en-us",
        "en-AU": "en-au",
        "en-IE": "en-ie",
        "en-IN": "en-in",
        "en-CA": "en-ca"
    }
    voice = accent_map.get(accent_code, "en-us")
    
    try:
        # Use espeak-ng with phoneme output (-x flag)
        result = subprocess.run(
            ['espeak-ng', '-x', '-v', voice, word],
            capture_output=True,
            text=True,
            check=True,
            timeout=5
        )
        # Extract phonemes (remove brackets and clean)
        phonemes = result.stdout.strip()
        # Remove brackets if present
        phonemes = re.sub(r'[\[\]]', '', phonemes)
        return phonemes
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        return None

# Build WORDS dictionary lazily - phonemes will be fetched on-demand
WORDS = {}

def get_word_phonemes_lazy(word, accent_code):
    """Get phonemes for a word on-demand (lazy loading)"""
    accent_name = ACCENT_MAP.get(accent_code, "American")
    
    # Check cache first
    if word in WORDS and accent_name in WORDS[word]:
        return WORDS[word][accent_name]
    
    # Get phonemes from espeak-ng
    espeak_phonemes = get_espeak_phonemes_for_word(word, accent_code)
    if espeak_phonemes:
        ipa_value = espeak_to_ipa(espeak_phonemes)
        if word not in WORDS:
            WORDS[word] = {}
        WORDS[word][accent_name] = {
            "espeak": espeak_phonemes,
            "ipa": ipa_value
        }
        return WORDS[word][accent_name]
    return None

print("✓ Server ready (words will be loaded on-demand)")

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

@app.route('/levels')
def get_levels():
    """Get available levels"""
    return jsonify(list(PHONICS_DATA.keys()))

@app.route('/categories')
def get_categories():
    """Get categories for a level"""
    level = request.args.get('level', 'Basic')
    if level not in PHONICS_DATA:
        return jsonify({"error": "Invalid level"}), 400
    
    return jsonify(list(PHONICS_DATA[level].keys()))

@app.route('/sounds')
def get_sounds():
    """Get sounds for a level and category"""
    level = request.args.get('level', 'Basic')
    category = request.args.get('category', '')
    
    if level not in PHONICS_DATA or category not in PHONICS_DATA[level]:
        return jsonify({"error": "Invalid level or category"}), 400
    
    sounds = []
    for idx, sound_data in enumerate(PHONICS_DATA[level][category]):
        sounds.append({
            "id": idx,
            "sound": sound_data["sound"],
            "ipa": get_value_for_accent(sound_data["ipa"], "en-US"),
            "es": get_value_for_accent(sound_data["es"], "en-US")
        })
    
    return jsonify(sounds)

@app.route('/sound/<int:sound_id>/words')
def get_sound_words(sound_id):
    """Get words for a specific sound"""
    level = request.args.get('level', 'Basic')
    category = request.args.get('category', '')
    accent_code = request.args.get('accent', 'en-US')
    
    if level not in PHONICS_DATA or category not in PHONICS_DATA[level]:
        return jsonify({"error": "Invalid level or category"}), 400
    
    if sound_id >= len(PHONICS_DATA[level][category]):
        return jsonify({"error": "Sound not found"}), 404
    
    sound_data = PHONICS_DATA[level][category][sound_id]
    sound = sound_data["sound"]
    es_value = get_value_for_accent(sound_data["es"], accent_code)
    ipa_value = get_value_for_accent(sound_data["ipa"], accent_code)
    
    # Include the sound itself as the first word for practice
    words = [sound] + sound_data["words"]
    
    return jsonify({
        "sound": sound,
        "ipa": ipa_value,
        "es": es_value,
        "words": words,
        "reference_es": es_value  # eSpeak letters for reference sound
    })

@app.route('/tts', methods=['POST'])
def text_to_speech():
    """Generate speech audio using espeak"""
    data = request.get_json()
    text = data.get('text', '')
    accent_code = data.get('accent', 'en-US')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    # Map accent code to espeak voice
    accent_map = {
        "en-GB": "en-gb",
        "en-US": "en-us",
        "en-AU": "en-au",
        "en-IE": "en-ie",
        "en-IN": "en-in",
        "en-CA": "en-ca"
    }
    voice = accent_map.get(accent_code, "en-us")
    
    # Generate audio file
    audio_path = '/tmp/tts_output.wav'
    try:
        subprocess.run([
            'espeak-ng',
            '-s', '150',
            '-g', '5',
            '-v', voice,
            '-w', audio_path,
            text
        ], check=True, capture_output=True, timeout=10)
        
        if not os.path.exists(audio_path):
            return jsonify({"error": "Audio generation failed"}), 500
        
        return send_file(audio_path, mimetype='audio/wav')
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"espeak error: {e.stderr.decode()}"}), 500
    except Exception as e:
        return jsonify({"error": f"TTS failed: {e}"}), 500

@app.route('/tts-espeak', methods=['POST'])
def text_to_speech_espeak():
    """Generate speech audio using espeak phonemes directly"""
    data = request.get_json()
    espeak_phonemes = data.get('espeak', '')
    accent_code = data.get('accent', 'en-US')
    
    if not espeak_phonemes:
        return jsonify({"error": "No espeak phonemes provided"}), 400
    
    accent_map = {
        "en-GB": "en-gb",
        "en-US": "en-us",
        "en-AU": "en-au",
        "en-IE": "en-ie",
        "en-IN": "en-in",
        "en-CA": "en-ca"
    }
    voice = accent_map.get(accent_code, "en-us")
    
    audio_path = '/tmp/tts_espeak_output.wav'
    try:
        # Use espeak with phoneme input ([[phonemes]])
        phoneme_text = f"[[{espeak_phonemes}]]"
        subprocess.run([
            'espeak-ng',
            '-s', '150',
            '-g', '5',
            '-v', voice,
            '-w', audio_path,
            phoneme_text
        ], check=True, capture_output=True, timeout=10)
        
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
    accent_code = request.form.get('accent', 'en-US')
    word = request.form.get('word', '')
    model_id = request.form.get('model', 'wav2vec2_lv60')
    
    accent_name = ACCENT_MAP.get(accent_code, "American")
    
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
    
    # Get phonemes lazily
    phoneme_data = get_word_phonemes_lazy(word, accent_code)
    if not phoneme_data:
        phoneme_data = WORDS.get(word, {}).get(accent_name, {})
    
    expected_espeak = phoneme_data.get("espeak", "N/A") if phoneme_data else "N/A"
    expected_ipa = phoneme_data.get("ipa", "N/A") if phoneme_data else "N/A"
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
