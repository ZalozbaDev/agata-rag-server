# Async RAG Server – Grundgerüst

Dieses Projekt ist ein **Grundgerüst** für einen Python-Server mit:

- `POST /parseHtml`
- `POST /parseUrl`
- `POST /ask`
- optionales Flag `store_in_db` für `parseHtml` und `parseUrl`
- zusätzlichen Alias-Endpunkten `GET /parse` und `POST /parse_html`
- Vektor-Datenbank mit **Qdrant**
- täglichem Re-Parsing echter Live-URLs
- Docker / Docker Compose
- asynchroner API mit FastAPI

## Architektur

```text
Client
  -> FastAPI
      -> ParserService
      -> IndexingService
          -> Embedding Provider
          -> Qdrant
      -> RetrievalService
      -> RagService
      -> APScheduler
```

## Parser-Integration

Dein hochgeladenes Parser-Set ist bereits eingebunden unter:

- `app/parsers/site_parsers/`

Der Adapter `app/parsers/adapters.py` wählt anhand von `url`, `source_url`, Dateiname oder Pfad automatisch den passenden Parser aus. Wenn kein spezifischer Parser passt, wird auf den generischen `parse_function(...)` zurückgefallen.

## Endpunkte

### `GET /parse`

Kompatibel zu deinem bisherigen Flask-Endpoint.

```bash
curl "http://localhost:8000/parse?url=https://example.com&min_chars=40&store_in_db=true"
```

### `POST /parseUrl`

JSON-Body mit URL.

```bash
curl -X POST http://localhost:8000/parseUrl   -H "Content-Type: application/json"   -d '{"url":"https://example.com","min_chars":40,"store_in_db":true}'
```

Antwort:

```json
[
  { "title": "Titel 1", "text": "..." },
  { "title": "Titel 2", "text": "..." }
]
```

### `POST /parseHtml`

Unterstützt alle drei Varianten:

1. **multipart/form-data** mit Datei
2. **multipart/form-data** mit Feld `html`
3. **application/json** mit `html`

Optional jeweils mit `url` oder `source_url`, `min_chars` und `store_in_db`.

`store_in_db` ist standardmäßig `false`. Nur wenn das Flag auf `true` gesetzt ist, werden die geparsten Daten in Qdrant gespeichert.

#### Datei-Upload

```bash
curl -X POST http://localhost:8000/parseHtml   -F "file=@./example.html"   -F "source_url=https://example.com/artikel"   -F "store_in_db=true"
```

#### JSON

```bash
curl -X POST http://localhost:8000/parseHtml   -H "Content-Type: application/json"   -d '{
        "source_url":"https://example.com/artikel",
        "html":"<html><body>...</body></html>",
        "min_chars":40,
        "store_in_db":true
      }'
```

Zusätzlich existiert der Alias `POST /parse_html`, damit dein bisheriges Request-Format weiter nutzbar bleibt.

### `POST /ask`

JSON-Body mit Frage.

```bash
curl -X POST http://localhost:8000/ask   -H "Content-Type: application/json"   -d '{"question":"Worum geht es in den Dokumenten?"}'
```

Antwort:

```json
"Hier steht die generierte Antwort als String"
```

## Starten

```bash
cp .env.example .env
docker compose up --build
```

API danach unter:

- `http://localhost:8000/docs`
- `http://localhost:8000/health`

## Wichtige Konfiguration

In `.env`:

- `OPENAI_API_KEY`
- `OPENAI_CHAT_MODEL`
- `OPENAI_EMBEDDING_MODEL`
- `EMBEDDING_DIMENSION`
- `CORS_ORIGINS`
- `CORS_ALLOW_CREDENTIALS`

## Re-Parsing alle 24h

Dann ruft der Scheduler diese Seiten alle 24 Stunden live ab, parsed sie neu und aktualisiert die Einträge in Qdrant.

Wichtig: Seiten mit starkem JavaScript-Rendering, Bot-Schutz oder Login können beim Server-seitigen Fetching unvollständige oder gar keine Inhalte liefern.

## Nächste sinnvolle Ausbaustufen

- echte Dokument-IDs und Versionierung
- Metadatenfilter pro Quelle / Mandant
- Antwort mit Quellenstellen statt nur String
- dedizierter Worker für Hintergrundjobs
- Redis / Queue für größere Parsing-Jobs
- Observability (Tracing, Metrics, strukturierte Logs)
- Authentifizierung vor den Endpunkten
- Rate Limiting

## Hinweis zur Skalierung

Der eingebaute Scheduler ist für ein **einfaches Setup** sinnvoll. Wenn du mehrere API-Instanzen parallel betreibst, solltest du den Scheduler in einen separaten Worker auslagern, damit Jobs nicht doppelt laufen.
