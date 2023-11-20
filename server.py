from flask import Flask, request, jsonify
from transformers import LlamaConfig,LlamaForCausalLM,LlamaTokenizer
from accelerate import init_empty_weights,infer_auto_device_map,load_checkpoint_in_model,dispatch_model
import torch
import flask_cors

app = Flask(__name__)
flask_cors.CORS(app)

model_path = '../Llama-2-7b-hf'
memory_bound = {0: '16GiB', 1: '16GiB', 2: '16GiB', 3: '16GiB'}

tokenizer = LlamaTokenizer.from_pretrained(model_path)
config = LlamaConfig.from_pretrained(model_path)

with init_empty_weights():
    model = LlamaForCausalLM._from_config(config, torch_dtype=torch.float16)

device_map = infer_auto_device_map(model, max_memory=memory_bound, no_split_module_classes=LlamaForCausalLM._no_split_modules)
load_checkpoint_in_model(model, model_path, device_map=device_map)
model = dispatch_model(model,device_map=device_map)

@app.route('/llmserver', methods=['POST'])
def infer():
    data = request.json
    input_text = data.get('input_text', '')
    result = generate(input_text)
    return jsonify({'result': result})

def generate(input_text, max_new_tokens=200, temperature=1.0, top_k=0, top_p=0.9):
    input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(
    **input_ids,
    max_new_tokens=max_new_tokens,
    temperature=temperature,
    top_k=top_k,
    top_p=top_p,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
    )
    output_texts = [tokenizer.decode(output_sequence, skip_special_tokens=True).replace('\n', ' ')
                    for output_sequence in outputs]
    return output_texts
	
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
