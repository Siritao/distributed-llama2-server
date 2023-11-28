# distributed-llama2-server
In this repo, we demonstrate how to serve a `Llama-2-7b-hf` on a node with 4x16G GPUs though GPU mem is partially occupied (supported by __model parallel__ and __CPU off-loading__) via __token streaming__.

## Install
### Prepare pre-trained model
Due to HuggingFace's policy, you need to apply for usage of `Llama-2-7b-hf` and download it before starting.

### Create docker containers
You can build images from source or by hand, plz refer to the `Dockerfile`. For instance, you can create server container by command like this:

```
docker run --gpus all --name=llmserver --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -p 5000:5000 -it -v $(pwd):/app nvcr.io/nvidia/pytorch:23.10-py3
```

Since it is hard to directly download the huge model checkpoints at creation, we recommend to pre-prepare and place it under `pwd`.

## Test inference pipeline

```
python test_infer.py
```

## Start server
You can use `watch -n 1 nvidia-smi` to decide the memory allocation and modify the `memory_bound` arg in `server.py`. Including more devices won't promote efficiency and may bring extra communication, so it is suggested to adjust `memory_bound` according to your environment.

```
python server.py
```

Using `Gunicorn` to deploy the server:

```
gunicorn server:app -c gunicorn.conf.py
```

## Query

```
python client.py 'text to be completed'
```

Note that multi-client query is supported by multi-thread serving (at the expense of latency, the total throughput may not increase).