# scripts/test_smoke_openai_controller.py
from pathlib import Path
import sys, logging
import os

# Ensure ../src is importable no matter where you run from
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from generative_policy_proposals._ControllerGeneratorOpenAI import ControllerGenerator

def main():
    logging.basicConfig(level=logging.INFO)

    actor = ControllerGenerator(
        model="google/gemma-3-12b-it:free",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("GEMMA3_12B_API_KEY"),
    )

    # Quick sanity check (optional; helps catch server/model issues fast)
    try:
        models = [m.id for m in actor.client.models.list().data]
        logging.info("Server models: %s", models)
    except Exception as e:
        logging.info("Model list not available (ok): %s", e)

    prompt = (
        "Write a Python function predict_next_action(obs, weights, memory) that returns an int. "
        "Return 0 by default. Put the code inside ```python``` fences."
    )
    reply = actor.generate_gameplay_code([], prompt)
    print("\n=== MODEL REPLY ===\n")
    print(reply)

if __name__ == "__main__":
    main()


