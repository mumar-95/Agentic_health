# Cancer Healthcare Assistant

Beginner-level RAG chatbot that answers cancer awareness questions from hospital documents first, then falls back to Groq general knowledge when relevant hospital content is unavailable.

**This is an educational awareness tool, not a diagnostic or treatment system.**

## Features

- Streamlit chat interface with conversation history
- Hospital document ingestion for PDF, DOCX, and TXT
- Local persistent ChromaDB vector store
- HuggingFace `sentence-transformers/all-MiniLM-L6-v2` embeddings
- Groq API answers with healthcare safety instructions
- Relevance-aware RAG retrieval and cautious LLM fallback
- Supabase persistence for chat sessions and messages

## Project layout

```
Agentic_health/
├── app.py                 # Streamlit UI
├── config/prompt.py       # Safety and response rules
├── documents/             # Add your hospital files here
├── supabase/schema.sql    # Database tables (run once in Supabase)
├── utils/
│   ├── document_loader.py
│   ├── text_chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── rag_pipeline.py
│   └── supabase_client.py
├── vector_db/chroma_db/   # Created at runtime
├── .env.example
├── requirements.txt
└── README.md
```

## Setup

### 1. Create a Python environment

With Conda:

```bash
conda create -n cancer_rag python=3.11
conda activate cancer_rag
```

Or with venv:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Groq

1. Copy `.env.example` to `.env`
2. Set your API key:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
```

Choose a chat model available on your Groq account. Never commit `.env`.

### 4. Configure Supabase

Add these to `.env` (see `.env.example`):

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_PUBLISHABLE_KEY=sb_publishable_your_key
SUPABASE_SECRET_KEY=sb_secret_your_key
SUPABASE_JWKS_URL=https://your-project.supabase.co/auth/v1/.well-known/jwks.json
```

Then create the tables once:

1. Open the [Supabase SQL Editor](https://supabase.com/dashboard/project/_/sql)
2. Paste and run the contents of `supabase/schema.sql`

The app uses `SUPABASE_SECRET_KEY` on the server to save:

- `chat_sessions` — one row per browser session
- `chat_messages` — user and assistant messages (`role`, `content`, `source`)
- `hospital_documents` — file name, type, size, extracted text, chunk counts, hashes
- `hospital_document_chunks` — individual text chunks linked to each document

If chat tables already exist, you can run only `supabase/add_documents_tables.sql`.

Document rows are upserted when the local `documents/` folder changes (or when you click **Sync documents now** / **Rebuild index**).

### 5. Add hospital documents

Place approved, non-identifying PDF, DOCX, or TXT files in `documents/`.

Do not upload patient-identifiable records.

If the folder is empty, the app still answers general educational questions and will say hospital-specific facts cannot be verified.

### 6. Run the app

From the project root:

```bash
streamlit run app.py
```

Open the local URL printed in the terminal.

## How answering works

1. Search hospital chunks in ChromaDB (top 3)
2. Keep chunks whose distance is at or below the relevance threshold (default `0.55`)
3. If relevant: answer with Groq using hospital context
4. If not relevant and the question is general education: cautious Groq fallback
5. If hospital-specific and unsupported: say it cannot be verified
6. If out of scope: brief refusal

### Tuning relevance

Chroma distance depends on metric and embeddings. After adding real documents, test known relevant and irrelevant questions and adjust `DEFAULT_RELEVANCE_DISTANCE_THRESHOLD` in `utils/rag_pipeline.py` if needed. Lower threshold = stricter hospital matches.

The index rebuilds when document names, sizes, or modification times change. Use **Rebuild index** in the sidebar if needed.

## Suggested test questions

| Question | Expected behavior |
| --- | --- |
| What are common early warning signs of cancer? | Hospital content if present; otherwise cautious general awareness |
| What is the difference between benign and malignant tumors? | Hospital content if present; otherwise general education |
| What cancer screening facilities does this hospital provide? | Only if supported by hospital documents; otherwise cannot verify |
| Who is the head of the oncology department? | Only if explicitly in hospital documents |
| What is today's weather? | Brief out-of-scope response |

## Safety and privacy

- Present the app as cancer awareness education, not diagnosis
- Do not tell users they have or do not have cancer
- Encourage professional care for concerning symptoms
- Do not recommend medication or treatment changes
- Keep API keys and hospital documents out of source control
- `.env`, `vector_db/`, and `documents/*` (except `documents/README.md`) are gitignored
