# Sample Documents (Demo Dataset)

This folder is populated by running:

```powershell
cd "C:\Project 1\backend"
venv\Scripts\activate
python ..\scripts\generate_sample_documents.py
```

It generates five files used throughout the README's demo procedure and
referenced by `backend/evaluation/benchmark_dataset.json`:

- `Operating_Systems.pdf` — 5 pages (intro, process management, deadlocks,
  deadlock prevention, memory management)
- `Java_Notes.pdf` — 4 pages (fundamentals, OOP principles, polymorphism,
  exception handling)
- `Machine_Learning.pdf` — 4 pages (intro, supervised learning,
  overfitting, evaluation metrics)
- `DBMS.pptx` — 4 slides (intro, normalization, transaction management,
  indexing)
- `handwritten_notes.jpg` — a rendered "handwritten-style" note on
  deadlock prevention, used to demonstrate OCR-based image search

The generator script is deterministic and the exact page/slide numbers it
produces are documented in its source (`scripts/generate_sample_documents.py`)
and matched by the benchmark dataset. If you edit the generator's content,
update `backend/evaluation/benchmark_dataset.json` to match.

Generated files are not committed to version control (see `.gitignore`);
regenerate them locally after cloning.
