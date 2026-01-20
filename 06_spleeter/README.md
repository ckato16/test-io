# Audio Source Separation with Spleeter

Separates audio into stems (vocals, drums, bass, etc.) using Deezer's Spleeter.

## Usage

```bash
docker build -t spleeter-app .
docker run -it -v $(pwd):/data spleeter-app
```

Inside container:
```bash
spleeter separate audio.mp3 -p spleeter:2stems-16kHz -o output/
```

See https://github.com/deezer/spleeter for more options.

