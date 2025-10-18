# OpenAI / OpenRouter Integration
This branch replaces the HuggingFace Transformer calls in `src/generative_policy_proposals/_ControllerGenerator.py` with an OpenAI-compatible client `src/generative_policy_proposals/_ControllerGeneratorOpenAI.py` that can talk to OpenRouter (e.g. Gemma 3 12B)

# Requirements

* Python 3.11+
* A virtual environment (.venv or similar)
* An OpenRouter API key (free tier available at https://openrouter.ai
)

# API Key Setup

VLM I used: Google: Gemma 3 12B (free) https://openrouter.ai/google/gemma-3-12b-it:free

1. Create your own API key by clicking the API tab
2. Save it as an environment variable in your shell. Using the name I provided below `(GEMMA3_12B_API_KEY)` makes sure that the key exists is identified wherever it's used. If you want to use a different api key name, then make sure to match all the references in the code to the new name.
```
export GEMMA3_12B_API_KEY="sk-xxxxxxxxxxxxxxxxxxx"
```
3. The code will automatically read this key when you run the scripts
```
echo $GEMMA3_12B_API_KEY
```