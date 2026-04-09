from app.models.schemas import ParsedSection
from app.utils.chunking import Chunker


def test_chunker_splits_text() -> None:
    chunker = Chunker(chunk_size=10, chunk_overlap=2)
    sections = [ParsedSection(title='T', text='1234567890abcdefghij')]

    chunks = chunker.split_sections(sections)

    assert len(chunks) >= 2
    assert chunks[0]['title'] == 'T'
