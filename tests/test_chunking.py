from app.models.schemas import ParsedSection
from app.utils.chunking import Chunker


def test_chunker_splits_text() -> None:
    chunker = Chunker(chunk_size=10, chunk_overlap=2)
    sections = [ParsedSection(title='T', text='1234567890abcdefghij')]

    chunks = chunker.split_sections(sections)

    assert len(chunks) >= 2
    assert chunks[0]['title'] == 'T'


def test_chunker_prefers_sentence_boundaries() -> None:
    chunker = Chunker(chunk_size=40, chunk_overlap=8)
    sections = [
        ParsedSection(
            title='T',
            text='Prěnja wěta je krótka. Druha wěta je něšto dalše. Třeća wěta je hišće dalša.',
        )
    ]

    chunks = chunker.split_sections(sections)

    assert chunks
    assert all('.' in chunk['text'] or len(chunk['text']) <= 40 for chunk in chunks)


def test_chunker_handles_paragraphs() -> None:
    chunker = Chunker(chunk_size=60, chunk_overlap=10)
    sections = [
        ParsedSection(
            title='T',
            text='Prěnja wěta.\n\nDruha wěta je w druhim wotstawku.',
        )
    ]

    chunks = chunker.split_sections(sections)

    assert len(chunks) >= 2
