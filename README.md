pip install -e .
uvicorn app:app --reload --host 0.0.0.0 --port 8000
docker build -t bit-admit-ai .
docker run -p 8000:8000 bit-admit-ai
