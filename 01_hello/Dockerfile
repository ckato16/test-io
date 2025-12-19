FROM python:3.10-slim
WORKDIR /app
COPY server.py .
RUN pip install flask flask-cors
EXPOSE 5000
CMD ["python", "server.py"]
