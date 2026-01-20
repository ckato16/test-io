# Music Generation with Magenta

Jupyter notebook environment for Google Magenta music generation models.

## Usage

```bash
docker build -t magenta-notebook .
docker run -p 8888:8888 magenta-notebook
```

Visit http://localhost:8888 and open `notebooks/test.ipynb` to use Magenta models for music generation.

Includes pre-downloaded models:
- Drum Kit RNN
- Basic RNN

