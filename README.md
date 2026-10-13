# build

docker build -f docker/Dockerfile -t agata-rag-server --progress=plain .  

# qdrant management

access dashboard at

http://IP:6333/dashboard

upload PDF document:

see README.api.md

# retrieval pipeline

Search runs in stages. Each one exists because the stage before it cannot do
the job alone.

1. **Dense candidates** — cosine search with a `RETRIEVAL_SCORE_THRESHOLD`
   floor. This is the relevance gate. Without it an off-topic question still
   returns the top-k documents of the collection, which matters a lot for the
   RAG-only personas that are not allowed to fall back on model knowledge.
2. **Sparse candidates** — BM25 for exact terms (names, paragraph numbers,
   dates) that embeddings tend to blur. No threshold, BM25 scores are unbounded.
3. **RRF fusion** — merges both lists by rank.
4. **Rerank (optional)** — a cross-encoder reads question and passage together.
   The only stage that can tell "mentions the same words" from "answers the
   question". Off by default because the model is ~1.1 GB.
5. **Collapse overlaps** — `CHUNK_OVERLAP` makes neighbouring chunks share
   text, so the top-k can contain the same passage several times. Adjacent
   chunks get stitched, near-duplicates dropped.

`RETRIEVAL_CANDIDATE_MULTIPLIER` controls how much stages 4 and 5 have to work
with; at 1 they can only reorder what fusion already chose.

## Embedding prompts

EmbeddingGemma is trained with task prefixes (`task: search result | query: `
for questions, `title: … | text: …` for passages) and loses accuracy without
them. `EMBEDDING_PROMPT_STYLE` is auto-detected from the model name; set it to
`none` for models that do not want prefixes, such as OpenAI's.

**Queries and documents must use the same scheme.** Changing
`EMBEDDING_PROMPT_STYLE`, the embedding model, or `EMBEDDING_DIMENSION` invalidates
existing vectors and requires re-indexing every collection.

## BM25 language

`BM25_LANGUAGE` selects Snowball stemming and stopwords for German (and other
supported languages), overridable per request via `language`.

Upper Sorbian (`hsb` / `dsb`) is indexed **without** stemming or stopwords.
Qdrant/bm25 has no Czech or Polish model, and the German list would delete real
Sorbian function words (`so`, `a`). Dense embeddings remain the main retrieval
path for Sorbian; BM25 only keeps exact surface forms.

# writing to qdrant

`store_sections` is incremental. Point ids are derived from
`source_id + position + text + language`, so a re-parse only embeds chunks that
actually changed and deletes the ones that disappeared. Re-indexing an
unchanged document costs no embedding calls at all. Changing the stored
language (for example after a mis-tag) produces new ids, so a re-parse with
`store_in_db=true` rewrites the payload.

Two ordering rules protect the stored data:

- new points are written **before** stale ones are deleted, so a provider
  outage mid-run cannot leave a source missing
- a parse that yields zero chunks leaves the collection untouched instead of
  wiping the previous version

Embeddings and upserts are batched (`EMBEDDING_BATCH_SIZE`, `UPSERT_BATCH_SIZE`)
so a large PDF does not become one oversized request.

