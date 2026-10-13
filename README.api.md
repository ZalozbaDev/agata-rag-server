# API Reference

All HTTP endpoints exposed by the RAG server. Base URL assumes `http://localhost:8000` (empty `API_PREFIX`).

Interactive docs: `http://localhost:8000/docs`

---

## `GET /health`

Liveness check.

```bash
curl http://localhost:8000/health
```

Response:

```json
{ "status": "ok" }
```

---

## Collections

Every request may target a specific Qdrant collection. This is how one server
serves several personas (Agata, Załožba, …) from separate knowledge bases.

- Omitting `collection` uses `QDRANT_COLLECTION` (default `documents`).
- Names must match `[A-Za-z0-9][A-Za-z0-9_-]{0,62}`; anything else returns `400`.
- Set `QDRANT_ALLOWED_COLLECTIONS` (comma separated) to restrict which names are
  accepted. When unset, any valid name is allowed.
- A collection that does not exist yet is created on demand, so a typo yields an
  empty result set rather than an error. Use the allowlist to catch typos.

### `GET /collections`

List the collections that exist in Qdrant with their point counts.

```bash
curl http://localhost:8000/collections
```

```json
{
  "default": "documents",
  "allowed": [],
  "collections": [
    { "name": "documents", "points": 1423 },
    { "name": "zalozba_spechowane_projekty", "points": 0 }
  ]
}
```

---

## Parse endpoints

Turn HTML, URLs, or PDFs into text sections. Shared response shape:

```json
[
  { "title": "Section title", "text": "Extracted text...", "language": "de" }
]
```

Shared optional fields:

| Field | Default | Description |
| --- | --- | --- |
| `min_chars` | `40` | Drop sections shorter than this many characters |
| `store_in_db` | `false` | When `true`, embed and store sections in Qdrant |
| `collection` | `QDRANT_COLLECTION` | Target collection when storing |
| `language` | `auto` | `de`, `hsb`, or `auto`. `auto` (and omitting the field) detects from the text, then falls back to `DOCUMENT_LANGUAGE`. The resolved value is written onto each section and into the Qdrant payload. |

### `POST /parseHtml`

Parse raw HTML into sections. Uses a site-specific parser when `url` / `source_url` / filename matches a known source; otherwise the generic HTML parser.

Accepts JSON or multipart form data.

**JSON**

```bash
curl -X POST http://localhost:8000/parseHtml \
  -H "Content-Type: application/json" \
  -d '{
    "html": "<html><body><h1>Title</h1><p>Long enough article text...</p></body></html>",
    "source_url": "https://example.com/artikel",
    "min_chars": 40,
    "store_in_db": false
  }'
```

**Multipart – HTML field**

```bash
curl -X POST http://localhost:8000/parseHtml \
  -F "html=<html><body><p>Article body...</p></body></html>" \
  -F "url=https://example.com/artikel" \
  -F "min_chars=40" \
  -F "store_in_db=false"
```

**Multipart – file upload**

```bash
curl -X POST http://localhost:8000/parseHtml \
  -F "file=@./example.html" \
  -F "source_url=https://example.com/artikel" \
  -F "store_in_db=true"
```

**Fields:** `html` (required unless `file` is set), `url` or `source_url` (optional), `min_chars`, `store_in_db`, `collection`, `language`.  
`min_chars` / `store_in_db` / `collection` / `language` may also be query params.

---

### `POST /parseUrl`

Fetch a live URL, then parse the downloaded HTML into sections.

```bash
curl -X POST http://localhost:8000/parseUrl \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/artikel",
    "min_chars": 40,
    "store_in_db": true
  }'
```

**Fields:** `url` (required), `min_chars`, `store_in_db`, `collection`, `language`.

Errors: `400` invalid URL or unknown collection, `502` fetch failed, `504` fetch timeout.

---

### `POST /parsePdf`

Extract text from one or more PDF uploads (one section per page that meets `min_chars`).

Requires `multipart/form-data`. Upload fields: `files`, `file`, or `uploads`.

```bash
curl -X POST http://localhost:8000/parsePdf \
  -F "files=@./doc1.pdf" \
  -F "files=@./doc2.pdf" \
  -F "min_chars=40" \
  -F "store_in_db=false"
```

Store a document in a persona-specific collection:

```bash
curl -X POST http://localhost:8000/parsePdf \
  -F "file=@./foerderrichtlinie.pdf" \
  -F "store_in_db=true" \
  -F "collection=zalozba_spechowane_projekty" \
  -F "language=de"
```

**Fields:** one or more PDF files (required), `min_chars`, `store_in_db`, `collection`, `language` (form or query).

Errors: `400` missing/non-PDF/invalid/encrypted PDF, unknown collection, or unknown `language`.

---

## `POST /ask`

Retrieve relevant context chunks and sources for a question from the vector store.

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Worum geht es in den Dokumenten?",
    "collection": "zalozba_spechowane_projekty",
    "top_k": 8,
    "history": [
      { "role": "user", "content": "Hallo Agata" },
      { "role": "assistant", "content": "Hallo! Wie kann ich helfen?" }
    ],
    "isPhoneCall": false
  }'
```

**Fields:**

| Field | Required | Description |
| --- | --- | --- |
| `question` | yes | User question (non-empty) |
| `collection` | no | Collection to search; defaults to `QDRANT_COLLECTION` |
| `top_k` / `topK` | no | Chunks to retrieve (1–50); defaults to `RETRIEVAL_TOP_K` |
| `language` | no | Document language of the collection (`de` / `hsb`); picks BM25 stemming. Defaults to `BM25_LANGUAGE` |
| `history` | no | Optional conversation history |
| `isPhoneCall` / `is_phone_call` | no | Optional phone-call flag |

Send `language` in the language the *collection* is written in, not the language
of the user. A Sorbian question against a German corpus should be translated
before it reaches this endpoint, otherwise BM25 has nothing to match on.

Response:

```json
{
  "answer": "",
  "contexts": [
    "Relevant chunk text..."
  ],
  "sources": [
    {
      "source_type": "url",
      "source_url": "https://example.com/artikel",
      "title": "Section title",
      "score": 0.82
    }
  ],
  "scores": [0.82]
}
```

If not enough strong retrieval hits are found, `contexts`, `sources`, and `scores` are empty.

`contexts[i]`, `sources[i]`, and `scores[i]` are 1:1 aligned so the caller can map
citations and retrieval quality back to their metadata.

Errors: `400` unknown collection, `502` / `504` provider failures, `500` unexpected server error.

---

## Managed persona documents

These endpoints only accept collections named `persona_` plus 24 hex characters.
That is the collection the Agata server creates for one MongoDB persona. The
default collection and the static Załožba collections are rejected, so a persona
upload cannot land in another knowledge base.

`persona_<id>` stays reachable even when `QDRANT_ALLOWED_COLLECTIONS` is set.

### `POST /documents`

Index one PDF or HTML file. Multipart field: `file`. Form fields: `collection`
(required), optional `min_chars` and `language`.

```bash
curl -X POST http://localhost:8000/documents \
  -F "file=@./foerderrichtlinie.pdf" \
  -F "collection=persona_0123456789abcdef01234567"
```

```json
{
  "source_id": "pdf:…",
  "collection": "persona_0123456789abcdef01234567",
  "section_count": 4,
  "chunk_count": 6,
  "filename": "foerderrichtlinie.pdf"
}
```

Errors: `400` wrong type, empty file, file over 20 MB, or a non-persona collection.
`422` when the file has no indexable text. `502` / `504` provider failures.

### `DELETE /documents`

Remove every chunk whose `source_id` belongs to that persona collection.

```bash
curl -X DELETE http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{
    "collection": "persona_0123456789abcdef01234567",
    "source_id": "pdf:…"
  }'
```

### `DELETE /collections/{name}`

Drop a managed persona collection after the persona itself is deleted.
`documents` and other static names return `400`. A missing persona collection
returns `{ "deleted": false }` so a retry can finish the MongoDB cleanup.
