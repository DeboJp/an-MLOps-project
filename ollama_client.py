import time
import requests
import json

class OllamaClient:
    """
    Client for interacting with a local Ollama instance.
    Includes helpers to request completions and measure MLOps performance metrics (TTFT, TPS).
    Install "llama3.2:3b-instruct-q4_K_M" and run "ollama create finqa-llama3.2 -f Modelfile" first.
    """
    def __init__(self, base_url="http://127.0.0.1:11434", model="finqa-llama3.2"):
        self.base_url = base_url
        self.model = model

    def generate(self, prompt):
        """
        Sends a generation request to Ollama and measures performance metrics.
        Returns:
            dict with 'text' (the generated string), 'ttft' (Time to First Token in seconds),
            'tps' (Tokens Per Second), and 'total_time' (total duration in seconds).
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True
        }

        start_time = time.time()
        first_token_time = None
        response_text = ""
        token_count = 0

        # Send request and stream the response
        response = requests.post(url, json=payload, stream=True)
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
