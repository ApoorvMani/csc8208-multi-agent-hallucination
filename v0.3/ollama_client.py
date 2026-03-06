"""
ollama_client.py - sends prompts and returns responses from local ollama model
"""

import requests
import time

OLLAMA_URL = "http://localhost:11434/api/generate"

def query_agent(model, prompt, system_prompt="", temperature=0.5):
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "options": {"temperature": temperature}
    }

    start = time.time() # start a timer to know how long model takes

    try:
    # calling and sending payload to ollama
        response = requests.post(
        OLLAMA_URL,         # address
        json=payload,       # data sent as json
        timeout=120         # timeout is no response 
        )

        duration = round(time.time() - start, 2) # stop the timer 
        response.raise_for_status()   # check for errors (like model not found)
        data = response.json()        # parse json from ollama to dict
        answer = data.get("response", "").strip()  # grab the answer text and clean it

    # check for no answer:
        if not answer:
            print(f"[WARNING] No response from ollama for {model}")
            return {"text": "", "duration": duration, "success": False}
        return {"text": answer, "duration": duration, "success": True}
    # --- error handling ---
    # each except catches a different thing that can go wrong

    except requests.exceptions.ConnectionError:
        # ollama isn't running
        duration = round(time.time() - start, 2)
        print(f"  [ERROR] Can't connect to Ollama. run ollama serve")
        return {"text": "", "duration": duration, "success": False}

    except requests.exceptions.Timeout:
        # model took longer than 120 seconds — probably too big
        duration = round(time.time() - start, 2)
        print(f"  [ERROR] Ollama timed out after 120s for {model}")
        return {"text": "", "duration": duration, "success": False}

    except Exception as e:
        # anything else we didn't expect
        duration = round(time.time() - start, 2)
        print(f"  [ERROR] Something went wrong: {e}")
        return {"text": "", "duration": duration, "success": False}