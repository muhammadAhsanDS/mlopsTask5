# Makefile

setup:
    pip install -r requirements.txt

run:
    python app.py

docker-build:
    docker build -t your-image-name .

docker-run:
    docker run -p 4000:80 your-image-name
