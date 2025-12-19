FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Download eSpeak models during build
RUN python -c "from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC; Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-lv-60-espeak-cv-ft'); Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-lv-60-espeak-cv-ft')"

RUN python -c "from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC; Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-xlsr-53-espeak-cv-ft'); Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-xlsr-53-espeak-cv-ft')"

COPY src/ ./src/

EXPOSE 5000

CMD ["python", "src/server.py"]
