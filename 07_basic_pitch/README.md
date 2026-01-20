# Pitch Transcription with Basic Pitch

Transcribes audio to MIDI using Spotify's Basic Pitch model.

## Usage

```bash
docker build -t basic-pitch-app .
docker run -p 5000:5000 basic-pitch-app
```

Visit http://localhost:5000, upload an audio file, adjust parameters:
- **Merge**: Merge short notes (0.0-1.0)
- **Threshold**: Confidence threshold (0.0-1.0)
- **Min Length**: Minimum note length in seconds

Downloads MIDI file with transcribed notes.

