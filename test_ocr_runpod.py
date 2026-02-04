import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

url = "https://api.runpod.ai/v2/qapobo6yo6o9rf/run"

payload = {
    "input": {
        "pdf_url": "https://www.accessdata.fda.gov/drugsatfda_docs/label/2002/21108s1lbl.pdf",
        "save_to_s3": True,
        "filename_prefix": "RENOVO_20020714"
    }
}

headers = {
    "Content-Type": "application/json"
}

# Check for API key in env, though user didn't provide one in prompt
api_key = os.environ.get("RUNPOD_API_KEY")
if api_key:
    headers["Authorization"] = f"Bearer {api_key}"
else:
    print("Warning: No RUNPOD_API_KEY found in environment. Request may fail if auth is required.")

print(f"Sending request to {url}...")
print(f"Payload: {json.dumps(payload, indent=2)}")

import concurrent.futures
import time

def run_test(idx):
    print(f"[{idx}] Sending request...")
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        print(f"[{idx}] Initial Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"[{idx}] Failed initial request: {response.text}")
            return

        data = response.json()
        task_id = data.get("id")
        status = data.get("status")
        print(f"[{idx}] Job ID: {task_id}, Status: {status}")

        if status in ["IN_QUEUE", "IN_PROGRESS"] and task_id:
            base_url = url.replace("/run", "")
            status_url = f"{base_url}/status/{task_id}"
            
            while status in ["IN_QUEUE", "IN_PROGRESS"]:
                time.sleep(3)
                try:
                    status_res = requests.get(status_url, headers=headers, timeout=30)
                    status_data = status_res.json()
                    status = status_data.get("status")
                    print(f"[{idx}] Poll Status: {status}")
                    
                    if status == "COMPLETED":
                        print(f"[{idx}] Job Completed!")
                        output = status_data.get("output", {})
                        if isinstance(output, dict):
                            print(f"[{idx}] Result Length: {len(str(output))}")
                        else:
                            print(f"[{idx}] Result: {str(output)[:100]}...")
                        break
                    elif status == "FAILED":
                        print(f"[{idx}] Job Failed: {status_data}")
                        break
                except Exception as e:
                    print(f"[{idx}] Poll Error: {e}")
                    time.sleep(2)
        elif status == "COMPLETED":
             print(f"[{idx}] Immediate Success!")

    except Exception as e:
        print(f"[{idx}] Error: {e}")

# Run 1 request first (Control)
print("\n--- Running Single Request Control ---")
run_test("Control")

# Run 3 concurrent requests
print("\n--- Running 3 Concurrent Requests ---")
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(run_test, i) for i in range(3)]
    concurrent.futures.wait(futures) 

