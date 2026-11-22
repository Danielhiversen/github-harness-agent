from github_harness_agent.review_solver_context import focused_source_excerpt


def test_matching_method_includes_owner_imports_and_referenced_definitions():
    noise = "\n".join(f"NOISE_{n} = {n}" for n in range(200))
    source = f"""from pathlib import Path
{noise}

class Reader:
    def __init__(self, path):
        self.path = Path(path)

    def load(self):
        return decode(self.path.read_text())

def decode(text):
    return text.strip()
"""
    result = focused_source_excerpt(
        source, "test_load_preserves_whitespace", max_chars=400
    )
    assert "def load(self):" in result
    assert "class Reader:" in result
    assert "self.path = Path(path)" in result
    assert "from pathlib import Path" in result
    assert "def decode(text):" in result
    assert "NOISE" not in result
    assert len(result) <= 400


def test_review_anchor_selects_correct_owner_for_ambiguous_method():
    source = """class Incoming:
    def update(self):
        return 1

class Outgoing:
    def update(self):
        return 2
"""
    result = focused_source_excerpt(
        source,
        "test_update",
        review_note="Preserve outgoing state",
        anchor=7,
        max_chars=90,
    )
    assert "return 2" in result
    assert "return 1" not in result
    assert len(result) <= 90


def test_large_active_method_leaves_room_for_constructor_and_relevant_lines():
    filler = "\n".join(f"        unused_{n} = {n}" for n in range(70))
    source = (
        "class Decoder:\n"
        "    def __init__(self, encoding):\n"
        "        self.encoding = encoding\n"
        "    def decode(self, raw):\n"
        f"{filler}\n"
        "        return raw.decode(self.encoding)\n"
    )
    result = focused_source_excerpt(source, "test_decode", anchor=75, max_chars=520)
    assert "def decode(self, raw):" in result
    assert "return raw.decode(self.encoding)" in result
    assert "self.encoding = encoding" in result
    assert len(result) <= 520
    unanchored = focused_source_excerpt(source, "test_decode", max_chars=520)
    assert "self.encoding = encoding" in unanchored
