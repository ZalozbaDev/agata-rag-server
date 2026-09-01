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
  { "title": "Section title", "text": "Extracted text..." }
]
```

Shared optional fields:

| Field | Default | Description |
| --- | --- | --- |
| `min_chars` | `40` | Drop sections shorter than this many characters |
| `store_in_db` | `false` | When `true`, embed and store sections in Qdrant |
| `collection` | `QDRANT_COLLECTION` | Target collection when storing |
| `language` | `DOCUMENT_LANGUAGE` (`hsb`) | Written to the `language` payload field |

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

Errors: `400` missing/non-PDF/invalid/encrypted PDF, or unknown collection.

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
| `history` | no | Optional conversation history |
| `isPhoneCall` / `is_phone_call` | no | Optional phone-call flag |

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
      "title": "Section title"
    }
  ]
}
```

If not enough strong retrieval hits are found, `contexts` and `sources` are empty.

`contexts[i]` and `sources[i]` are 1:1 aligned so the caller can map citations
back to their metadata.

Errors: `400` unknown collection, `502` / `504` provider failures, `500` unexpected server error.
