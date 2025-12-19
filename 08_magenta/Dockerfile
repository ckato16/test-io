FROM python:3.10-slim
# docker buildx build --platform linux/arm64 -t magenta-notebook .
# docker run -it --rm -p 8888:8888 magenta-notebook

WORKDIR /app

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    curl \
    build-essential \
    libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

# ARM-compatible TensorFlow
RUN pip install --upgrade pip
RUN pip install tensorflow==2.12.0
RUN pip install tensorflow-probability==0.20.0
RUN pip install tf_slim
RUN pip install pyfluidsynth

# Install Magenta without broken ARM dependencies
RUN pip install --no-deps magenta

# Install ARM-safe dependencies manually
RUN pip install \
    absl-py \
    pretty_midi \
    librosa \
    numpy \
    scipy \
    six \
    protobuf \
    mido \
    note-seq

# Install Jupyter
RUN pip install jupyter

RUN mkdir -p /app/notebooks

RUN mkdir -p /app/models

# Download ARM-safe models
RUN mkdir -p /app/models/drum_kit_rnn && \
    curl -L -o /app/models/drum_kit_rnn/drum_kit_rnn.mag \
    http://download.magenta.tensorflow.org/models/drum_kit_rnn.mag

RUN mkdir -p /app/models/basic_rnn && \
    curl -L -o /app/models/basic_rnn/basic_rnn.mag \
    http://download.magenta.tensorflow.org/models/basic_rnn.mag

EXPOSE 8888

CMD ["jupyter","notebook","--ip=0.0.0.0","--port=8888","--no-browser","--allow-root","--NotebookApp.token=''"]
