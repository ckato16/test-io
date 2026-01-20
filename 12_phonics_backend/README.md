# Phonics Pronunciation Trainer Backend

Backend service for phonics pronunciation training with structured phonics content. Uses Wav2Vec2 models to analyze pronunciation accuracy.

## Usage

```bash
docker build -t phonics-app .
docker run -p 5000:5000 phonics-app
```

Visit http://localhost:5000

Features:
- Select level (Basic/Advanced) → Category → Sound → Word
- Multiple accent support (British, American, Australian, Irish, Indian, Canadian)
- Record pronunciation and get accuracy scores
- Listen to reference sounds using eSpeak phonemes
- Uses eSpeak-NG for TTS and Wav2Vec2 for analysis
