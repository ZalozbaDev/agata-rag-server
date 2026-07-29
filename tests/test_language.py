from app.utils.language import detect_query_language


def test_detect_query_language_hsb_chars() -> None:
    assert detect_query_language('Kotry je wobsah teje knihi?') == 'hsb'


def test_detect_query_language_hsb_markers() -> None:
    assert detect_query_language('Chcy wědźeć nětko wjace wo serbski.') == 'hsb'


def test_detect_query_language_german() -> None:
    assert detect_query_language('Was ist der Inhalt dieses Buches?') == 'de'
