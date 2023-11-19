FROM nvcr.io/nvidia/pytorch:22.12-py3

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "server.py"]
