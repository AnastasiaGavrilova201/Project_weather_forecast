FROM python:3.11

WORKDIR /app

COPY service/backend/ /app

RUN mkdir -p ~/.postgresql && \
    wget --progress=dot:giga "https://storage.yandexcloud.net/cloud-certs/CA.pem" \
    --output-document ~/.postgresql/root.crt && \
    chmod 0600 ~/.postgresql/root.crt

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "main.py"]
