# Hello World - Backend API

Simple Flask backend with CORS enabled to test Docker container communication.

## Usage

```bash
docker build -t hello-app .
docker run -p 5000:5000 hello-app
```

Visit http://localhost:5000/hello to test the API endpoint.

