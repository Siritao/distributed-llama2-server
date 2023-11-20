FROM nvcr.io/nvidia/pytorch:23.10-py3

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

EXPOSE 5000

# CMD ["python", "server.py"]
