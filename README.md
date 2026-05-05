# Minimal RAG Lab

RAG lab for course exercises. It uses LangChain, OpenAI models, Streamlit, PostgreSQL, and pgvector.

## Setup

Copy the example environment file and add your OpenAI API key:

```bash
cp .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY=your-key-here
OPENAI_CHAT_MODEL=gpt-5.4-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Add `.txt`, `.md`, or `.pdf` files to `data/`.

Optional: download Kubernetes docs as example data:

```bash
./scripts/download_kubernetes_docs.sh en
```

Use another language code if the Kubernetes website has that docs tree:

```bash
./scripts/download_kubernetes_docs.sh de
```

## Platform Notes

On Linux and macOS, run the Kubernetes example data script from a terminal:

```bash
./scripts/download_kubernetes_docs.sh en
```

On Windows, run the script from Git Bash:

```bash
sh scripts/download_kubernetes_docs.sh en
```

If PowerShell is preferred, the same Git command can be run manually:

```powershell
git clone --depth 1 --filter=blob:none --sparse https://github.com/kubernetes/website.git kubernetes-docs
cd kubernetes-docs
git sparse-checkout set content/en/docs
cd ..
$source = "kubernetes-docs\content\en\docs"
$target = "data\kubernetes\en"
Get-ChildItem $source -Recurse -Include *.md,*.txt | ForEach-Object {
    $relative = $_.FullName.Substring((Resolve-Path $source).Path.Length + 1)
    $destination = Join-Path $target $relative
    New-Item -ItemType Directory -Force -Path (Split-Path $destination) | Out-Null
    Copy-Item $_.FullName $destination
}
rmdir /S /Q kubernetes-docs
```

## Run

Start the lab:

```bash
docker compose up --build
```

Open Streamlit at:

```text
http://localhost:8501
```

Click **Ingest data** in the sidebar before asking questions.

The sidebar also lets you choose hybrid, vector-only, or keyword-only search and filter retrieval to a specific source folder.

## CLI

Ingest files:

```bash
docker compose run --rm app python src/ingest.py
```

Retrieve relevant chunks:

```bash
docker compose run --rm app python src/retrieve.py "your question"
```

Ask with RAG:

```bash
docker compose run --rm app python src/rag.py "your question"
```

## Files

- `src/ingest.py` loads `.txt`, `.md`, and `.pdf` files from `data/`, splits them, and stores embeddings in pgvector.
- `src/retrieve.py` searches pgvector with hybrid keyword/vector retrieval and optional source-folder filtering.
- `src/rag.py` retrieves context and asks the chat model, including streaming support for the UI.
- `src/app.py` provides a Streamlit chat UI and shows source filename, chunk number, score, and content.
- `scripts/download_kubernetes_docs.sh` downloads Kubernetes docs into `data/kubernetes/<language>/`.
