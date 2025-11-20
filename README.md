
# Prompt Engineering

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/tanayrastogi-repo/PromptEng/blob/main/PromptTemplates.py)

Interactive Marimo app demonstrating foundational and advanced prompt-engineering techniques—zero-shot, one-shot, few-shot, and self-consistency. The lab follows the IBM's Skill build course [Develop Generative AI Applications: Get Started](https://www.coursera.org/learn/develop-generative-ai-applications-get-started?specialization=ibm-rag-and-agentic-ai). 

The notebook uses OpenRouter’s free GPT-OSS model in place of IBM Watson and is useful for anyone experimenting with prompt design patterns.

## Prerequisites
- Python 3.13+
- [uv](https://docs.astral.sh/uv/) installed (`curl -LsSf https://astral.sh/uv/install.sh | sh` or `pip install uv`)
- OpenRouter API key set as `OPENROUTER_API`

## Quickstart
1. Clone the repo and enter it:
   ```bash
   git clone https://github.com/tanayrastogi-repo/PromptEng.git
   cd PromptEng
   ```
2. Install dependencies:
   ```bash
   uv sync
   ```
3. (Optional) Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```


## Run the Marimo App
```bash
uv run marimo run PromptTemplates.py
```
- Open the URL printed by Marimo (defaults to `http://127.0.0.1:27180/`).
- Supply your OpenRouter API key in the UI if you prefer not to use `.env`.
- Each section mixes markdown explanations with chat widgets; click the chat button to execute the prompt template.


## Dependency Management
- Use `uv add <package>` / `uv remove <package>` to update dependencies and lockfile.
- `uv run <cmd>` executes commands inside the managed environment, ensuring consistent tooling.
- Keep `.env`, `.venv`, and other generated artifacts listed in `.gitignore`.


## License
MIT Licence
