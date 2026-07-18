# HARDDX — Intelligent AI Chatbot for Computer Hardware Diagnosis and Troubleshooting

A working implementation of the system described in Chapter 3 of the project report:
a hybrid **rule-based + machine learning** chatbot that diagnoses common computer
hardware faults from a natural-language description and returns likely causes and
step-by-step troubleshooting instructions.

## Architecture

```
User message
    │
    ▼
NLP Preprocessing (utils/nlp_preprocessor.py)
    - clean → tokenize → stop-word removal → lemmatization
    - lightweight entity extraction (hardware component nouns)
    │
    ├──────────────────────────────┬───────────────────────────────┐
    ▼                               ▼
ML Classifier                  Rule-Based Engine
(TF-IDF + Linear SVM,           (keyword matching against
 trained on hardware_faults_     data/knowledge_base.json)
 dataset.csv)
    │                               │
    └──────────────┬────────────────┘
                    ▼
        Confidence Fusion (inference_engine.py)
        - agreement between both layers → high confidence
        - strong rule evidence overrides weak ML guess
        - falls back to whichever layer has signal
                    │
                    ▼
        Diagnosis: fault label, causes, solution steps,
        severity, confidence score
                    │
                    ▼
        Flask API (app.py) → logged to database (database.py)
                    │
                    ▼
        Web Chat UI (templates/, static/)
```

## Project structure

```
hardware_chatbot/
├── app.py                       # Flask application & API routes
├── dialogue_manager.py          # Multi-turn state machine: clarify -> diagnose -> guide
├── inference_engine.py          # Hybrid rule-based + ML diagnosis engine (single-turn core)
├── train_model.py               # Trains and saves the ML classifier
├── ingest_external_datasets.py  # Maps external Kaggle CSVs into the training schema
├── database.py                  # Conversation + dialogue-state logging (SQLite by default, MySQL-ready)
├── requirements.txt
├── data/
│   ├── hardware_faults_dataset.csv   # Training data for the ML classifier
│   ├── knowledge_base.json           # Rule-based fault → causes/clarifying Qs/steps
│   ├── external/                     # Place extracted Kaggle dataset folders here
│   │   ├── computer_hardware/
│   │   ├── electronic_components/
│   │   ├── electronic_parts/
│   │   ├── software_reliability/
│   │   └── software_defects/
│   └── chatbot.db                    # SQLite database (auto-created)
├── models/
│   ├── intent_classifier.pkl    # Trained sklearn pipeline (auto-created by train_model.py)
│   ├── label_encoder.pkl
│   └── model_name.pkl
├── utils/
│   └── nlp_preprocessor.py      # NLP pipeline: tokenize, lemmatize, entity extraction
├── templates/
│   └── index.html               # Chat UI (text + image upload + settings modal)
└── static/
    ├── css/style.css
    └── js/
        ├── chat.js              # Drives the multi-turn conversation UI
        └── vision.js            # Client-side Gemini Vision call for image diagnosis
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download NLTK data (one-time):
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('punkt_tab')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('omw-1.4')
   ```

3. Train the ML classifier (creates the `models/` folder):
   ```bash
   python3 train_model.py
   ```

4. Initialize the database:
   ```bash
   py database.py
   ```

5. Run the web app:
   ```bash
   py app.py
   ```
   Then open **http://127.0.0.1:5000** in your browser.

## Using MySQL instead of SQLite

By default the app uses a local SQLite file at `data/chatbot.db` so it runs without
any extra setup, matching the schema design in Chapter 3.6. To use MySQL instead
(as named in the report), install `mysql-connector-python` and set environment
variables before running:

````bash
$env:DB_ENGINE="mysql"
$env:DB_HOST="localhost"
$env:DB_USER="root"
$env:DB_PASSWORD=""
$env:DB_NAME="hardware_chatbot"
```

```bash
py -m pip install mysql-connector-python
export DB_ENGINE=mysql
export DB_HOST=localhost
export DB_USER=root
export DB_PASSWORD=
export DB_NAME=hardware_chatbot
py database.py   # creates the tables
py app.py
```

## Multi-turn conversation flow

The chatbot no longer answers in a single shot. Each session moves through:

1. **Describe** — user types a problem and/or attaches a photo.
2. **Clarify** — the bot asks up to 2 short follow-up questions (defined per
   fault category in `data/knowledge_base.json` under `clarifying_questions`)
   to disambiguate before committing to a diagnosis.
3. **Diagnose** — likely causes + severity + confidence are shown once.
4. **Guide** — solution steps are presented one at a time. After each step the
   bot asks whether it worked; a "no" advances to the next step, a "yes"
   resolves the conversation, and exhausting all steps without success
   escalates to "see a technician."
5. Saying something like *"new issue"* or *"something else"* at any point
   restarts the flow.

This state machine lives in `dialogue_manager.py` and is persisted per
browser session in the `dialogue_sessions` table (`database.py`), so it
survives across the stateless HTTP requests Flask handles.

## Image-based diagnosis (Gemini Vision)

Users can attach a photo (e.g. of a cable, a BSOD, visible component damage)
via the 📷 button. Image **understanding** is handled by Google's Gemini
Vision API — **called directly from the browser**, not from the Flask server.

Why client-side: this project's Flask backend was developed inside a sandboxed
environment whose outbound network is restricted to a fixed allowlist that
does not include Google's API host. Running the Gemini call from the user's
own browser sidesteps that, and as a side benefit means the user's API key
never has to be sent to or stored on the server — it lives only in the
browser's `localStorage` (set via the "⚙ VISION KEY" button).

Setup for the end user:
1. Get a free key at https://aistudio.google.com/apikey
2. Click "⚙ VISION KEY" in the app and paste it in.
3. Attach a photo and send — `static/js/vision.js` sends it to
   `gemini-2.0-flash`, gets back a short description + suspected fault
   category, and merges that into the same hybrid rule/ML diagnosis pipeline
   used for text (`POST /api/chat` accepts an optional `image_finding` field).

If no key is set, the app still works fully for text-only diagnosis.

## Incorporating the external Kaggle datasets

The following datasets were requested for training data expansion:
- General Computer Hardware Dataset (Dilshaan Sandhu)
- Electronic components and devices (aryaminus)
- Electronic Parts Dataset (olavomendes)
- Software Reliability Dataset (vasanthkumarch)
- Software Defect Prediction (semustafacevik)

**These could not be downloaded automatically** — kaggle.com is not on this
project's sandboxed network allowlist. To bring them in:

1. Download and extract each dataset from Kaggle yourself.
2. Copy each dataset's **extracted folder contents** into the matching
   pre-created folder under `data/external/` (whatever Kaggle ships inside —
   CSVs, multiple CSVs, image subfolders, nested directories — is fine; the
   script searches recursively, so internal layout doesn't need to be flattened):
   ```
   data/external/computer_hardware/
   data/external/electronic_components/
   data/external/electronic_parts/
   data/external/software_reliability/
   data/external/software_defects/
   ```
3. Run:
   ```bash
   py ingest_external_datasets.py
   ```

Some of these datasets are folders containing **multiple CSVs** (e.g. a main
table plus a lookup table); others are folders of **labeled images**
(class-named subfolders of photos). `ingest_external_datasets.py` handles
both within the same folder:

- **CSVs** — every `.csv` found anywhere inside a dataset folder is loaded,
  its columns and a sample row are printed, and any row with a usable
  free-text column (description/notes/issue/etc.) is keyword-matched against
  the project's existing fault categories. CSVs that turn out to be specs or
  metrics tables (no usable text column) are skipped with an explanation —
  matched rows are written to `data/hardware_faults_dataset_audit.csv` for
  manual review before merging into the live training set.
- **Image folders** — any subdirectory containing image files directly is
  treated as a candidate labeled class (folder name = label), and every
  image path inside it is recorded to `data/external_image_manifest.csv`
  with that label. Images are only cataloged, not moved/copied/uploaded.
  Folder-name labels are a structural guess, not validated against the
  chatbot's six fault categories — review the manifest before using it.

**Important caveat:** based on their public Kaggle descriptions, the
hardware/electronics datasets are likely component **spec/catalog** data
(model numbers, prices, specifications, possibly photos of individual parts)
and the software datasets are **code-metric** tables (cyclomatic complexity,
defect counts) for predicting bugs from source code — neither is naturally
"user complaint text → fault category" data, which is what `train_model.py`
needs for the chatbot's text classifier. The script won't silently invent
fake labels from spec tables; expect it to report 0 mapped text rows for
several of these and treat that as informative, not a failure.

If, after inspecting the actual contents, the layout or column names differ
from what's assumed here, share the printed folder/column listing and a
couple of sample rows and the mapping logic in `ingest_external_datasets.py`
can be adjusted precisely to fit.



To add a new fault category, add an entry to `data/knowledge_base.json` with
`keywords`, `causes`, `solution_steps`, `severity`, and `confidence_base`, then
add a handful of example phrasings labeled with that category to
`data/hardware_faults_dataset.csv` and re-run `python3 train_model.py`.

## Notes on the ML component

The bundled dataset is intentionally small (a demonstration set, not a
production-scale corpus), so cross-validated ML accuracy alone is modest. This
is why the system is a **hybrid**: the deterministic rule-based knowledge base
compensates for the ML classifier's limited training data, while the ML layer
still contributes generalization to phrasings not explicitly covered by the
keyword lists. Expanding `hardware_faults_dataset.csv` with more labeled
examples will directly improve standalone ML accuracy.
