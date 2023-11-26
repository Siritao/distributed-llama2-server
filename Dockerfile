FROM nvcr.io/nvidia/pytorch:23.10-py3

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY server.py /app/server.py
