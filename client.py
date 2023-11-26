import requests
import sys


def run_client(input_text):
    url = 'http://localhost:5000/llmserver'
    Headers = {'Accept': 'text/event-stream'}
    data = {'input_text': input_text}

    response = requests.post(url, json=data, stream=True, headers=Headers)
    for res in response.iter_lines():
        print(res.decode(), end='')
        sys.stdout.flush()
    print()


if __name__ == '__main__':
    input_text = sys.argv[1]
    run_client(input_text)
