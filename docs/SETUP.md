# Setup Instructions

## 1. Install Python 3.10+

**Windows:**
- Download from https://www.python.org/downloads/
- Check "Add Python to PATH" during installation
- Verify: `python --version`

**macOS:**
```bash
brew install python@3.10  # Install Homebrew first: https://brew.sh
python3 --version
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip
python3 --version
```

## 2. Clone Repository
```bash
git clone https://github.com/ExistentialRiskBenchmark
cd ExistentialRiskBenchmark
```

## 3. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

## 4. Install Dependencies
```bash
pip install -r requirements.txt
pip install pre-commit
pre-commit install
```

## 5. Configure API Keys
Create `.env` file:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

## 6. Verify Installation
```bash
python -c "import anthropic, openai, google.generativeai"
pytest tests/
```

## Troubleshooting
- **"python not found"**: Use `python3` on macOS/Linux
- **Permission denied**: Use `pip install --user` or check venv activation
- **Import errors**: Ensure venv activated (`which python` should show venv path)
