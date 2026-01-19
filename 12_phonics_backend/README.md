## Setup Instructions

```
docker build -t phonics-app .
docker build --no-cache -t phonics-app .
docker run -p 5000:5000 phonics-app
visit http://localhost:5000
docker run -it phonics-app bash
ls -lh /root/.cache/huggingface/hub/
ls -lh /root/.cache/torch/hub/
```