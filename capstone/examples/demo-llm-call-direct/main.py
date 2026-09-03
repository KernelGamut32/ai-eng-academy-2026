import requests
import json
import http.client
import httpx
import sys
import os
from dotenv import load_dotenv

load_dotenv()

# Validate required environment variables early with helpful messages
HAWKEYE_TOKEN_URL = os.getenv("HAWKEYE_TOKEN_URL")
HAWKEYE_HOST= os.getenv("HAWKEYE_HOST")
HAWKEYE_CLIENT_ID = os.getenv("HAWKEYE_CLIENT_ID")
HAWKEYE_CLIENT_SECRET = os.getenv("HAWKEYE_CLIENT_SECRET")
OPERATION_ENDPOINT = os.getenv("OPERATION_ENDPOINT")

def get_token(auth_server_url, client_id, client_secret):
    token_req_payload = {'grant_type': 'client_credentials'}

    token_response = requests.post(
        auth_server_url,
        data=token_req_payload,
        verify=False,
        allow_redirects=False,
        auth=(client_id, client_secret)
    )

    if token_response.status_code != 200:
        print(
            f"Failed to obtain token from the OAuth 2.0 server. Status: {token_response.status_code}. Body: {token_response.text}",
            file=sys.stderr,
        )
        sys.exit(1)

    tokens = json.loads(token_response.text)
    token = tokens['access_token']
    return token

token = get_token(
    HAWKEYE_TOKEN_URL,
    client_id=HAWKEYE_CLIENT_ID,
    client_secret=HAWKEYE_CLIENT_SECRET,
)

print(f"Obtained token: {token}\n\n")

conn = http.client.HTTPSConnection(HAWKEYE_HOST)
payload = json.dumps({
  "model": "openai/gpt-oss-120b",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful, pattern-following assistant."
    },
    {
      "role": "user",
      "content": "Help me translate the following corporate jargon into plain English."
    },
    {
      "role": "assistant",
      "content": "Sure, I'd be happy to!"
    },
    {
      "role": "user",
      "content": "New synergies will help drive top-line growth."
    },
    {
      "role": "assistant",
      "content": "Things working well together will increase revenue."
    }
  ],
  "top_p": 1,
  "temperature": 1,
  "stream": False,
  "max_tokens": 400
})
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {token}'
}
conn.request("POST", OPERATION_ENDPOINT, payload, headers)
res = conn.getresponse()
data = res.read()
print("Response using HTTP client")
print(data.decode("utf-8"))
