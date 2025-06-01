#!/usr/bin/env python3
import requests
import json

# Test the SGLang server with a simple completion
url = 'http://localhost:30000/generate'
data = {
    'text': 'The future of artificial intelligence is',
    'sampling_params': {
        'max_new_tokens': 100,
        'temperature': 0.7
    }
}

try:
    print("🔄 Testing SGLang server...")
    response = requests.post(url, json=data)
    if response.status_code == 200:
        result = response.json()
        print('✅ SGLang server is working!')
        print('Prompt:', data['text'])
        print('Completion:', result.get('text', ''))
    else:
        print(f'❌ Error: {response.status_code}')
        print(response.text)
except Exception as e:
    print(f'❌ Connection error: {e}')
    print('Make sure the SGLang server is running on port 30000') 