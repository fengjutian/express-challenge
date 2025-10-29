cd backend

pip install fastapi uvicorn torch transformers Pillow

uvicorn app:app --host 0.0.0.0 --port 8000
