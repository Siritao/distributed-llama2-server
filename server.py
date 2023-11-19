import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("../Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("../Llama-2-7b-hf")

@app.route('/infer', methods=['POST'])
def infer():
    data = request.json
    input_text = data.get('input_text', '')
    result = generate(input_text)
    return jsonify({'result': result})

def generate(input_text, max_length=50, num_return_sequences=1, temperature=1.0, top_k=0, top_p=0.9):
  input_ids = tokenizer.encode(input_text, return_tensors="pt")
  output_sequences = model.generate(
    input_ids=input_ids,
    max_length=max_length,
    num_return_sequences=num_return_sequences,
    temperature=temperature,
    top_k=top_k,
    top_p=top_p,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
  )
  output_texts = []
  for output_sequence in output_sequences:
    output_text = tokenizer.decode(output_sequence, skip_special_tokens=True)
    output_texts.append(output_text)
  return output_texts
	
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
