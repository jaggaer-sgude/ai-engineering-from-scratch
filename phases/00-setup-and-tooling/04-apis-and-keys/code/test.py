import os
import json
import urllib.request

try:
    from dotenv import load_dotenv
    loaded = load_dotenv()
except ImportError:
        loaded = False

print("Loaded:", loaded)
print("API key:", os.getenv("GEMINI_API_KEY"))

def call_with_sdk():
    try:
        from google import genai
    except ImportError:
        print("Install the SDK: pip install google-genai")
        return
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY environment variable first")
        return
    
    client = genai.Client(
        api_key=api_key
    )
    response = client.models.generate_content(
        model = 'gemini-3.5-flash-lite',
        contents= 'What is a neural network in one sentence?'
    )
    print(response.text)


def call_raw_http():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("set the key first")
        return

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/gemini-3.5-flash:generateContent?key={api_key}"
    )

    body = {
        "contents": [
            {
                "parts": [
                    {
                        "text": "What is neural network?"
                    }
                ]
            }
        ]
    }

    request = urllib.request.Request(
        url = url,
        data = json.dumps(body).encode("utf-8"),
        headers = {
            "Content-Type": "application/json"
        },
        method= "POST",
    )

    with urllib.request.urlopen(request) as response:
        result = json.loads(response.read().decode("utf-8"))

    print(result["candidates"][0]["content"]["parts"][0]["text"])

if __name__ == "__main__":
    call_with_sdk()
    call_raw_http()