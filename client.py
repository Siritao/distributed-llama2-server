import requests

def query(input_text):
    url = 'http://localhost:5000/llmserver'
    prompt = {'input_text': input_text}
    response = requests.post(url, json=prompt)
    
    if response.status_code == 200:  # success
        result = response.json()
        print(result['result'])
    else:
        print('Error code:', response.status_code)

if __name__ == '__main__':
    input_text = "The defination of cloud computing is"
    query(input_text)
