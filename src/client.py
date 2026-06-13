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
    Client for interacting with local Ollama, MLX-LM, or AWS SageMaker instances.
    Includes helpers to request completions and measure performance metrics (TTFT, TPS).
    """
    def __init__(self, base_url="http://127.0.0.1:11434", model="finqa-llama3.2", aws_region="us-east-1"):
        self.base_url = base_url
        self.model = model
        self.aws_region = aws_region

    def generate_mlx(self, prompt, system_prompt=None):
        """
        Sends a generation request to local MLX-LM server (port 11435) and measures metrics.
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

    def generate_sagemaker(self, prompt, system_prompt=None, endpoint_name="finqa-llama3-2-tuned-endpoint"):
        """
        Sends a generation request to the deployed AWS SageMaker endpoint and measures latency.
        Requires boto3 to be configured in the environment.
        """
        import boto3
        
        sys_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
        
        # SageMaker TGI Container prompt layout
        formatted_prompt = f"<|system|>\n{sys_prompt}\n<|user|>\n{prompt}\n<|assistant|>\n"
        
        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": 128,
                "temperature": 0.01, # a tiny non-zero value, HGF requires it.
                "do_sample": False, # ignoes temp and picks highest prob token.
                "stop": ["EOF", "\n"]
            }
        }
        
        client = boto3.client("sagemaker-runtime", region_name=self.aws_region)
        
        start_time = time.time()
        response = client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload)
        )
        end_time = time.time()
        
        result = json.loads(response['Body'].read().decode('utf-8'))
        response_text = result[0]['generated_text']
        
        if response_text.startswith(formatted_prompt):
            response_text = response_text[len(formatted_prompt):]
            
        total_time = end_time - start_time
        token_count = len(response_text) // 4  # Standard token count approximation
        tps = token_count / total_time if total_time > 0 else 0
        ttft = total_time * 0.1  # Approximation for blocking endpoint
        
        return {
            "text": response_text.strip(),
            "ttft": ttft,
            "tps": tps,
            "total_time": total_time,
            "token_count": token_count
        }

    def generate(self, prompt, system_prompt=None):
        """
        Routes the request to Ollama, MLX-LM, or SageMaker depending on the model name.
        """
        if self.model == "finqa-llama3.2-tuned":
            return self.generate_mlx(prompt, system_prompt=system_prompt)
        elif "sagemaker" in self.model:
            return self.generate_sagemaker(prompt, system_prompt=system_prompt)

        # Local Ollama default route
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.0
            }
        }
        if system_prompt:
            payload["system"] = system_prompt

        start_time = time.time()
        first_token_time = None
        response_text = ""
        token_count = 0

        response = requests.post(url, json=payload, stream=True, timeout=60)
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode('utf-8'))
                
                if first_token_time is None:
                    first_token_time = time.time()
                
                response_text += chunk.get("response", "")
                
                if chunk.get("done", False):
                    token_count = chunk.get("eval_count", 0)

        end_time = time.time()
        
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
