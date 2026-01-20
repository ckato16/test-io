# Third-Party Software Notices

This repository uses third-party software components. Required attributions:

## Apache License 2.0 Components

**TensorFlow** (used in `07_basic_pitch`, `08_magenta`)  
License: Apache 2.0 | Source: https://github.com/tensorflow/tensorflow

**Transformers/HuggingFace** (used in `10_phoneme`, `12_phonics_backend`)  
License: Apache 2.0 | Source: https://github.com/huggingface/transformers

**Facebook Wav2Vec2 Models** (used in `10_phoneme`, `12_phonics_backend`)  
- `facebook/wav2vec2-lv-60-espeak-cv-ft`
- `facebook/wav2vec2-xlsr-53-espeak-cv-ft`  
License: Apache 2.0 | Source: https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft

**Magenta** (used in `08_magenta`)  
License: Apache 2.0 | Source: https://github.com/magenta/magenta

**basic-pitch** (used in `07_basic_pitch`)  
License: Apache 2.0 | Source: https://github.com/spotify/basic-pitch

## GPL-3.0 Components

**eSpeak-NG** (used in `10_phoneme`, `12_phonics_backend`)  
License: GPL-3.0-or-later | Source: https://github.com/espeak-ng/espeak-ng  
Note: Used as external subprocess, not statically linked.

**phonemizer** (used in `10_phoneme`, `12_phonics_backend`)  
License: GPL-3.0-or-later | Source: https://github.com/bootphon/phonemizer

## Other Dependencies

All Python packages (Flask, PyTorch, librosa, numpy, etc.) and system dependencies (ffmpeg, libsndfile1) are permissively licensed (MIT, BSD, Apache 2.0) and allow commercial use.

For full license texts, see: https://www.apache.org/licenses/LICENSE-2.0 and https://www.gnu.org/licenses/gpl-3.0.html
