# Revisar el siguiente link para mas información sobre la creación automatica de Dockerfiles 
# en Visual studio code https://aka.ms/vscode-docker-python
FROM python:3.9-slim

EXPOSE 5002

WORKDIR /app
COPY . /appdo

# Install pip requirements
COPY requirements.txt .
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r requirements.txt


# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["gunicorn", "--bind", "0.0.0.0:5002", "app:app"]