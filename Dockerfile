FROM nvcr.io/nvidia/pytorch:23.10-py3

RUN mkdir -p /app/logs/
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY test_infer.py /app/test_infer.py
COPY server.py /app/server.py
