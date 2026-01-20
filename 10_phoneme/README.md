# Phoneme Pronunciation Trainer

Backend service for phoneme pronunciation training with accent variation support. Uses Wav2Vec2 models to analyze pronunciation accuracy.

## Usage

```bash
docker build -t phoneme-app .
docker run -p 5000:5000 phoneme-app
```

Visit http://localhost:5000

Features:
- Select user mode (Native, Japanese, French)
- Choose accent patterns
- Record pronunciation and get accuracy scores
- Uses eSpeak-NG for TTS and Wav2Vec2 for analysis
