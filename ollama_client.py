import time
import requests
import json

SYSTEM_PROMPT = """You are a financial calculator assistant. Your task is to output a mathematical program using the FinQA Domain-Specific Language (DSL) to answer the user's question based on the provided text and table.

Allowed operations:
- add(arg1, arg2)
- subtract(arg1, arg2)
- multiply(arg1, arg2)
- divide(arg1, arg2)
- exp(arg1, arg2)
- greater(arg1, arg2)
- table_max(row_name)
- table_min(row_name)
- table_sum(row_name)
- table_average(row_name)

Refer to the results of previous steps using #0, #1, etc.
Format your output exactly as a comma-separated sequence of operations, ending with EOF. Do not include any other explanations, markdown formatting, or conversational filler.

Example 1:
Context:
Table:
| Year | Revenue |
| 2020 | 50000 |
| 2021 | 65000 |
Question: What was the rate of increase of revenue from 2020 to 2021?
Output: subtract(65000, 50000), divide(#0, 50000), EOF

CRITICAL: You must extract the exact literal numbers from the table. Do not use placeholders. Each problem will be unique, and require basic arithmetic decision making. Count digits carefully."""

class OllamaClient:
    """
    Client for interacting with a local Ollama instance.
    Includes helpers to request completions and measure MLOps performance metrics (TTFT, TPS).
    Install "llama3.2:3b-instruct-q4_K_M" and run "ollama create finqa-llama3.2 -f Modelfile" first.
    """
    def __init__(self, base_url="http://127.0.0.1:11434", model="finqa-llama3.2"):
        self.base_url = base_url
        self.model = model

    def generate_mlx(self, prompt, system_prompt=None):
        """
        Sends a generation request to the MLX-LM server and measures performance metrics.
        """
        sys_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
        url = "http://127.0.0.1:11435/v1/chat/completions"
        payload = {
            "model": "fused_model",
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": True,
            "temperature": 0.0
        }

        start_time = time.time()
        first_token_time = None
        response_text = ""
        token_count = 0

        response = requests.post(url, json=payload, stream=True, timeout=60)
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8').strip()
                if decoded_line.startswith("data: "):
                    data_str = decoded_line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        if first_token_time is None:
                            first_token_time = time.time()
                        
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            response_text += content
                            token_count += 1
                    except Exception:
                        pass

        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        ttft = (first_token_time - start_time) if first_token_time else total_time
        generation_time = total_time - ttft
        tps = token_count / generation_time if generation_time > 0 else 0

        return {
            "text": response_text.strip(),
            "ttft": ttft,
            "tps": tps,
            "total_time": total_time,
            "token_count": token_count
        }

    def generate(self, prompt, system_prompt=None):
        """
        Sends a generation request to Ollama or MLX-LM and measures performance metrics.
        Returns:
            dict with 'text' (the generated string), 'ttft' (Time to First Token in seconds),
            'tps' (Tokens Per Second), and 'total_time' (total duration in seconds).
        """
        if self.model == "finqa-llama3.2-tuned":
            return self.generate_mlx(prompt, system_prompt=system_prompt)

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True
        }
        if system_prompt:
            payload["system"] = system_prompt


        start_time = time.time()
        first_token_time = None
        response_text = ""
        token_count = 0

        # Send request and stream the response
        response = requests.post(url, json=payload, stream=True, timeout=60)
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode('utf-8'))
                
                # Check for first token (or rather, the first response chunk)
                if first_token_time is None:
                    first_token_time = time.time()
                
                # Add to response
                response_text += chunk.get("response", "")
                
                # If we're done, we might get metadata
                if chunk.get("done", False):
                    # Ollama provides token counts if available
                    token_count = chunk.get("eval_count", 0)

        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        ttft = (first_token_time - start_time) if first_token_time else total_time
        
        # If token count is not provided by Ollama, fallback to character/word-based approximation
        # if token_count == 0:
        #     # simple fallback: roughly 1 token per 4 characters
        #     token_count = len(response_text) // 4
        
        generation_time = total_time - ttft
        tps = token_count / generation_time if generation_time > 0 else 0

        return {
            "text": response_text.strip(),
            "ttft": ttft,
            "tps": tps,
            "total_time": total_time,
            "token_count": token_count
        }

if __name__ == "__main__":
    client = OllamaClient()
    print("Testing connection to Ollama with custom model 'finqa-llama3.2'...")
    
    # Mock FinQA prompt combining a table and a text fact
    mock_prompt = """
    Context:
    Table:
    company | payments volume ( billions ) | total volume ( billions ) | total transactions ( billions ) | cards ( millions )
    american express | $ 637 | $ 647 | 5.0 | 86
    Question:
    what is the average payment volume per transaction for american express?
    """
    
    try:
        res = client.generate(mock_prompt)
        print(f"\nPrompt Sent:\n{mock_prompt.strip()}\n")
        print(f"Response (should be FinQA program DSL): '{res['text']}'")
        print(f"TTFT: {res['ttft']:.4f}s")
        print(f"TPS: {res['tps']:.2f}")
        print(f"Total time: {res['total_time']:.4f}s")
        print(f"Token count: {res['token_count']}")
    except Exception as e:
        print(f"Error: {e}")
        print("Please verify that Ollama is running and the 'finqa-llama3.2' model has been built.")
