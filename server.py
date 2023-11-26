from flask import Flask, request, Response
import flask_cors

from threading import Thread

from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer, TextIteratorStreamer
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model, dispatch_model

import torch


app = Flask(__name__)
flask_cors.CORS(app)

model_path = './Llama-2-7b-hf'
memory_bound = {0: '16GiB', 1: '16GiB', 2: '16GiB', 3: '16GiB', 'cpu': '16GiB'}

print('init model...')
tokenizer = LlamaTokenizer.from_pretrained(model_path)
config = LlamaConfig.from_pretrained(model_path)

with init_empty_weights():
    model = LlamaForCausalLM._from_config(config, torch_dtype=torch.float16)

device_map = infer_auto_device_map(model,
                                   max_memory=memory_bound,
                                   no_split_module_classes=LlamaForCausalLM._no_split_modules)

load_checkpoint_in_model(model, model_path, device_map=device_map)
model = dispatch_model(model,device_map=device_map)
print('model init done!')


def generate(input_text, max_new_tokens=200, temperature=1.0, top_k=0, top_p=0.9):
    input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)

    decode_kwargs = dict(skip_special_tokens=True)
    streamer = TextIteratorStreamer(tokenizer, decode_kwargs=decode_kwargs)

    generation_kwargs = dict(input_ids,
                             streamer=streamer,
                             max_new_tokens=max_new_tokens,
                             temperature=temperature,
                             top_k=top_k,
                             top_p=top_p,
                             do_sample=True)
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    for new_text in streamer:
        yield new_text + '\n'


@app.route('/llmserver', methods=['POST'])
def infer():
    data = request.json
    input_text = data.get('input_text', 'hello i am llama')

    return Response(generate(input_text), mimetype='text/event-stream')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
