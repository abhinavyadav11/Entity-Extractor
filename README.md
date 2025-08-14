# Info Extractor

This project extracts entities and relations from a PDF using Python, with optional help from an open‑source LLM (via local Ollama running Mistral).

Entities: Organisation, Name, PAN
Relation: PAN_Of

## Quick start

1) Create and activate venv (already configured by the workspace tools). If doing locally:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies:

```bash
pip install -r requirements.txt
```

3) Run extraction on your PDF (heuristics only):

```bash
python src/extract_entities.py "PDF for Python LLM.pdf" --out result.csv
```

This generates:
- result_entities.csv
- result_relations.csv
- result_combined.csv

4) Optional: Improve recall/quality using an open‑source LLM via Ollama.

- Install Ollama: https://ollama.com/download
- Pull a model (e.g., mistral):

```bash
ollama pull mistral
```

- Run with LLM assistance:

```bash
python src/extract_entities.py "PDF for Python LLM.pdf" --use-llm --llm-model mistral --out result.csv
```

The script merges regex/heuristics with LLM outputs and produces the same CSVs as above.

## Output format

- Entities CSV: columns [type, value]
- Relations CSV: columns [subject, predicate, object] with predicate fixed to PAN_Of
- Combined CSV: a single file including both entities and relations for convenience

## Notes for best quality

- The PAN regex is precise: [A-Z]{5}[0-9]{4}[A-Z]
- Heuristics look for lines containing organization keywords and name hints.
- The LLM prompt enforces JSON, deduplication and the allowed predicate.
- If text extraction is poor, consider adding OCR pre-processing (e.g., `ocrmypdf`).
