# Project Name - Lonn-San's Magnificent Model Manipulator

This project is designed to fine-tune language models using QLoRA, manage datasets, and interact with the Hugging Face Hub.

## Setup

This project uses `uv` for Python environment and package management. It's a fast Python installer and resolver.

### 1. Install `uv` (if you haven't already)

Follow the official installation instructions for `uv` from [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv).

For example, on macOS and Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Or using pip:
```bash
pip install uv
```

### 2. Create and Activate the Virtual Environment

Once `uv` is installed, navigate to the project root directory and run the following commands:

```bash
# Create a virtual environment named .venv
uv venv
# Activate the virtual environment
# On macOS and Linux:
source .venv/bin/activate
# On Windows (PowerShell):
# .venv\Scripts\Activate.ps1
# On Windows (CMD):
# .venv\Scripts\activate.bat
```

### 3. Install Dependencies

With the virtual environment activated, install the project dependencies (once `pyproject.toml` or `requirements.txt` is populated):
```bash
uv pip install -r requirements.txt  # Or uv pip install -e . if using pyproject.toml with dependencies
```

---
*This README is being crafted by your humble Padawan.*
