FROM python:3.11

WORKDIR /app

COPY service/frontend/ /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["streamlit", "run",  "Start.py"]
