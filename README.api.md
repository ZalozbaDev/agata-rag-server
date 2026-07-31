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

**Fields:** `html` (required unless `file` is set), `url` or `source_url` (optional), `min_chars`, `store_in_db`.  
`min_chars` / `store_in_db` may also be query params.

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

**Fields:** `url` (required), `min_chars`, `store_in_db`.

Errors: `400` invalid URL, `502` fetch failed, `504` fetch timeout.

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

Single file:

```bash
curl -X POST http://localhost:8000/parsePdf \
  -F "file=@./document.pdf" \
  -F "store_in_db=true"
```

**Fields:** one or more PDF files (required), `min_chars`, `store_in_db` (form or query).

Errors: `400` missing/non-PDF/invalid/encrypted PDF.

---

## `POST /ask`

Retrieve relevant context chunks and sources for a question from the vector store.

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Worum geht es in den Dokumenten?",
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

Errors: `502` / `504` provider failures, `500` unexpected server error.
