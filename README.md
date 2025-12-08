<div align="center">

# 🃏 Knowledge Cards

**Transform research papers into structured insights with AI**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Powered by Gemini](https://img.shields.io/badge/Powered%20by-Gemini%20AI-purple.svg)](https://ai.google.dev/)

<img src="assets/cover.png" alt="Knowledge Cards" width="600"/>

*A schema-agnostic LLM pipeline that reifies unstructured documents into structured JSON knowledge cards.*

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Usage](#-usage) • [API](#-api) • [Contributing](#-contributing)

</div>

---

## ✨ Features

- **🎯 Schema-Agnostic** — Design your own extraction templates. Define exactly what information you want to pull from documents.
- **⚡ AI-Powered Extraction** — Leverage Google Gemini to intelligently extract and structure information from your PDFs.
- **📊 Beautiful Reports** — Generate professional PDF reports with structured cards for each paper, plus synthesized meta-cards.
- **📦 Multiple Outputs** — Export to JSONL, CSV, and PDF formats. Perfect for further analysis or sharing with your team.
- **🧩 Meta-Card Synthesis** — Automatically generate a synthesized overview card that combines insights from all processed papers.
- **📈 Real-time Progress** — Watch your cards being built with live progress updates as each paper is processed.
- **🎨 Modern UI** — Beautiful retro-futurist interface with glassmorphism design.

## 🖼️ Screenshots

<details>
<summary>Click to expand</summary>

### Welcome Screen
The landing page introduces the tool and its capabilities.

### Template Designer
Ask a research question and let AI generate a structured extraction schema.

### Runs Dashboard
Monitor your card-building runs and download results in multiple formats.

</details>

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- A Google Gemini API key ([Get one here](https://aistudio.google.com/apikey))

### Setup

```bash
# Clone the repository
git clone https://github.com/AntoineBellemare/knowledge-cards.git
cd knowledge-cards

# Install dependencies
pip install -r requirements.txt

# Create environment file
cd src
echo "GOOGLE_API_KEY=your_gemini_api_key_here" > .env
```

## 🚀 Quick Start

### 1. Start the Server

```bash
cd src
uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Open the App

Open `src/index.html` in your browser. You'll be greeted by the Welcome screen.

### 3. Create Your First Template

1. Navigate to **"New template from question"**
2. Enter a research question like: *"How does network dynamics change across biological lifeforms?"*
3. Click **"Generate template"** — AI will create a structured schema for your question

### 4. Build Your Cards

1. Go to **"Runs / reports"**
2. Select your template and paper folder
3. Click **"Build Cards"** and watch the progress
4. Download your results as JSONL, CSV, or PDF

## 📖 Usage

### Project Structure

```
knowledge-cards/
├── src/                    # Main application code
│   ├── api.py              # FastAPI backend
│   ├── index.html          # Web interface
│   ├── gemini_template_creation.py  # Core pipeline
│   └── cards_to_pdf.py     # PDF report generator
├── cards/                  # Your template schemas (JSON)
├── papers/                 # Your PDF papers organized by topic
│   └── plant_cognition/
│       ├── paper1.pdf
│       └── paper2.pdf
├── results/                # Generated outputs
└── assets/                 # Images and static files
```

### Templates

Templates are JSON schemas that define what information to extract. Example:

```json
{
  "name": "network_dynamics",
  "description": "Analyzing network dynamics across biological systems",
  "schema": {
    "metadata": {
      "title": "string",
      "authors": ["string"],
      "year": "number"
    },
    "findings": {
      "network_type": "string",
      "key_metrics": ["string"],
      "conclusions": "string"
    }
  }
}
```

### Paper Organization

Organize your PDFs in folders under `papers/`:

```
papers/
├── plant_cognition/
│   ├── baluska2006.pdf
│   └── trewavas2014.pdf
├── neuroscience/
│   └── friston2010.pdf
└── complexity/
    └── kauffman1993.pdf
```

### Output Formats

| Format | Description |
|--------|-------------|
| `.jsonl` | One JSON object per line, each representing a paper's extracted card |
| `.csv` | Flattened summary for spreadsheet analysis |
| `.pdf` | Beautiful formatted report with all cards and meta-synthesis |

## 🔌 API

The backend exposes a REST API for programmatic access:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/schema_from_question` | POST | Generate a template from a research question |
| `/run_gemini_stream` | GET | Run extraction with SSE progress updates |
| `/templates` | GET | List all available templates |
| `/templates` | POST | Save a new template |
| `/paper_folders` | GET | List available paper folders |
| `/results` | GET | List generated result files |
| `/download/{type}/{filename}` | GET | Download result files |

### Example: Generate Schema

```bash
curl -X POST http://localhost:8000/schema_from_question \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the mechanisms of plant memory?",
    "pointers": ["conceptual", "empirical"]
  }'
```

## 🛠️ Development

### Running Locally

```bash
# Backend
cd src
uvicorn api:app --reload --port 8000

# Frontend
# Simply open src/index.html in your browser
```

### Tech Stack

- **Backend**: FastAPI, Python
- **AI**: Google Gemini (gemini-2.5-flash-lite)
- **PDF Processing**: PyMuPDF (fitz), ReportLab
- **Frontend**: Vanilla HTML/JS, Tailwind CSS

## 🚢 Deployment

### Railway (Backend)

The included `Procfile` is configured for Railway deployment:

```
web: cd src && uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}
```

### Cloudflare Pages (Frontend)

Deploy the `src/` folder as a static site. The frontend auto-detects local vs production API URLs.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Google Gemini](https://ai.google.dev/) for AI-powered extraction
- UI inspired by retro-futurist and cyberpunk aesthetics
- Thanks to all contributors and users!

---

<div align="center">

**Built with 💜 by [Antoine Bellemare](https://github.com/AntoineBellemare)**

</div>
