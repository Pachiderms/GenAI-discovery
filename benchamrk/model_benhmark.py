import time
from ollama import generate

def test_speed(model_name, prompt, num_iterations=10):
    start_time = time.time()
    tokens = 0
    
    for _ in range(num_iterations):
        response = generate(model=model_name, prompt=prompt)
        tokens += response['eval_count']

    end_time = time.time()
    total_time = end_time - start_time
    avg_tokens = tokens / num_iterations
    tps = avg_tokens / (total_time / num_iterations)
    average_time = total_time / num_iterations
    
    print(f"Model: {model_name}")
    print(f"Total time for {num_iterations} iterations: {total_time:.2f} seconds")
    print(f"Average time per iteration: {average_time:.2f} seconds")
    print(f"Tokens per second (avg based): {tps:.2f}")

if __name__ == "__main__":
    mistral = "mistral:7b"
    phi = "phi3"
    prompt = "Talk about minimax algorithm in 200 words maximum."
    test_speed(mistral, prompt)
    test_speed(phi, prompt)