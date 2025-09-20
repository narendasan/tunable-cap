# scripts/smoke_openai_controller.py
from generative_policy_proposals._ControllerGeneratorOpenAI import ControllerGenerator

actor = ControllerGenerator(
    model="Qwen/Qwen2-VL-2B-Instruct",
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",
)
print(actor.generate_gameplay_code([], "Write a Python function predict_next_action(obs, weights, memory) in ```python``` fences."))
