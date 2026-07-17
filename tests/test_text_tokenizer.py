from preprocessing.text.tokenizer import GraphemeTokenizer


def test_roundtrip():
    tokenizer = GraphemeTokenizer.build_from_texts(["hello world", "another sentence"])
    text = "hello world"
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text


def test_save_load(tmp_path):
    tokenizer = GraphemeTokenizer.build_from_texts(["hello world"])
    path = tmp_path / "vocab.json"
    tokenizer.save(path)
    loaded = GraphemeTokenizer.load(path)
    assert loaded.vocab == tokenizer.vocab
    assert loaded.encode("hello") == tokenizer.encode("hello")
