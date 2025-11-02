Hey, sorry for the less professional documentation.

Currently very busy with the dev, documentation will follow later, I promise :)

How to run?

1. pip install -r requirements.txt
2. pip install -e .
3. uvicorn app:app --reload --host 0.0.0.0 --port 8000
4. docker build -t bit-admit-ai .
5. docker run -p 8000:8000 bit-admit-ai

Its in between ML and MLOPS thing, to be fully MLOPS I need to deploy it on some cloud may be EC2 or go with Arzue and also have to do CICD. Other than thats its complete. Anyways if you find issues ping me.
