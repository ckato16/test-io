## Setup Instructions

```
docker build -t phoneme-app .
docker build --no-cache -t phoneme-app .
docker run -p 5000:5000 phoneme-app
visit http://localhost:5000
docker run -it phoneme-app bash
ls -lh /root/.cache/huggingface/hub/
ls -lh /root/.cache/torch/hub/
```