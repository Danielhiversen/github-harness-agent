from github_harness_agent.review_solver_context import focused_source_excerpt
from github_harness_agent.review_solver import _proposal_messages
from github_harness_agent.review_recovery import ReviewRecovery


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


def test_anchor_beats_matching_wrapper_and_unrelated_class():
    source = """class ExportPreview:
    def preview(self):
        return "distractor"

class Reader:
    def __init__(self):
        self.offset = 1

    def decode(self, data):
        return data + self.offset

def export_preview(reader, data):
    return reader.decode(data)
"""
    result = focused_source_excerpt(
        source, "test_export_preview", review_note="Fix export preview", anchor=10
    )
    assert result.startswith("# Line 9\n    def decode")
    assert "def export_preview" in result
    assert "self.offset = 1" in result
    assert "distractor" not in result


def test_top_level_assignment_anchor_includes_consuming_function():
    source = """DEFAULT_LIMIT = 4

def keep_first(values):
    return values[:DEFAULT_LIMIT]

def read_large_export(values):
    return values
"""
    result = focused_source_excerpt(source, "test_read_large_export", anchor=1)
    assert result.startswith("# Line 1\nDEFAULT_LIMIT = 4")
    assert "def keep_first(values):" in result
    assert "def read_large_export" not in result


def test_innermost_function_wins_when_two_definitions_contain_anchor():
    source = """def export(values):
    def normalize(value):
        return value.strip()
    return [normalize(value) for value in values]
"""
    result = focused_source_excerpt(source, "test_export", anchor=3)
    assert result.startswith("# Line 2\n    def normalize")


def test_prompt_matches_review_note_using_selected_fixture_body():
    source = """def trim(values):
    return values[:4]

def normalize(values):
    return values
"""
    review = ReviewRecovery(
        (
            ("module.py", 2, "Preserve trim behavior"),
            ("module.py", 5, "Normalize the numeric category"),
        ),
        "",
    )
    target = (0, "tests/test_shared_name.py::test_case")
    fixtures = {
        "elsewhere/test_shared_name.py": "def test_case():\n    assert trim(values)\n",
        "test_shared_name.py": (
            "def test_other():\n    assert trim(values)\n"
            "def test_case():\n    assert normalize(numeric_category)\n"
        ),
    }
    messages = _proposal_messages(
        target, {"module.py": source}, {target: ""}, False, "", review, fixtures, (), ""
    )
    prompt = messages[-1]["content"]
    assert prompt.startswith("Selected failure:")
    assert "Editable checkout paths" in prompt
    assert "Reviewer feedback: Normalize the numeric category" in prompt
    assert "Focused source from module.py:\n# Line 4\ndef normalize" in prompt
