import json

import pytest

from github_harness_agent.review_line_ranges import (
    apply_line_range_edit,
    line_range_prompt,
    prepare_line_range_edit,
)

PATH = "pkg/queue.py"
SYMBOL = (PATH, "Queue.receive")
SOURCE = """before = 1

class Queue:
    @staticmethod
    def receive(
        values,
    ):
        # preserve this comment
        result = sum(values)
        return result + 1

    def unrelated(self):
        return 9

after = 2
"""


def _prepare(sources=None):
    return prepare_line_range_edit(
        {PATH: SOURCE} if sources is None else sources, SYMBOL
    )


def _reply(snapshot, *edits):
    return json.dumps({"snapshot": snapshot.digest, "edits": edits})


def _edit(start, end, *lines):
    return {"start": start, "end": end, "lines": list(lines)}


def test_method_relative_ranges_update_only_localized_body():
    sources = {PATH: SOURCE, "unrelated.py": "x = 3\n"}
    snapshot = _prepare(sources)
    prompt = line_range_prompt(snapshot)
    assert "pkg/queue.py::Queue.receive" in prompt
    assert snapshot.digest in prompt
    assert "1 |     @staticmethod" in prompt
    assert "5 |         # preserve this comment" in prompt
    assert "6 |         result = sum(values)" in prompt
    assert "unrelated" not in prompt
    example = json.loads(
        next(row for row in prompt.splitlines() if row.startswith('{"snapshot"'))
    )
    assert example["edits"] == [
        _edit(snapshot.first_body_line, snapshot.first_body_line, "        pass")
    ]

    response = _reply(
        snapshot,
        _edit(7, 7, "        return result + bonus"),
        _edit(6, 6, "        result = sum(values)", "        bonus = 2"),
    )
    revised = apply_line_range_edit(response, sources, snapshot)
    assert revised[PATH] == SOURCE.replace(
        "        return result + 1", "        bonus = 2\n        return result + bonus"
    )
    assert revised["unrelated.py"] == sources["unrelated.py"]
    assert sources[PATH] == SOURCE


@pytest.mark.parametrize(
    "edits",
    [
        (),
        (_edit(4, 4, "    ):"),),
        (_edit(8, 8, "        return 0"),),
        (_edit(6, 7), _edit(7, 7, "        return 2")),
        (_edit(True, 6),),
        (_edit(7, 7, "    def add_new_method(self): return 1"),),
        (_edit(7, 7, "        return 0\nclass Escape: pass"),),
        (_edit(7, 7, "        return result + 1"),),
        tuple(_edit(7, 7, "        return 2") for _ in range(7)),
    ],
)
def test_rejects_malformed_or_out_of_scope_ranges(edits):
    snapshot = _prepare()
    with pytest.raises(ValueError):
        apply_line_range_edit(_reply(snapshot, *edits), {PATH: SOURCE}, snapshot)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (_edit(400, 400, "        return 0"), "numbered definition rows 6 through 7"),
        (_edit(7, 7, "return 0"), "body indentation"),
        (_edit(7, 7, "        return 0\n"), "one string per source line"),
    ],
)
def test_invalid_range_feedback_identifies_the_specific_format_error(edit, message):
    snapshot = _prepare()
    with pytest.raises(ValueError, match=message):
        apply_line_range_edit(_reply(snapshot, edit), {PATH: SOURCE}, snapshot)


def test_rejects_changed_snapshot_and_ambiguous_payloads():
    sources = {PATH: SOURCE, "unrelated.py": "x = 1\n"}
    snapshot = _prepare(sources)
    valid = _reply(snapshot, _edit(7, 7, "        return 42"))
    for stale in (
        {**sources, PATH: SOURCE.replace("before = 1", "before = 4")},
        {**sources, "unrelated.py": "x = 2\n"},
    ):
        with pytest.raises(ValueError, match="snapshot"):
            apply_line_range_edit(valid, stale, snapshot)
    for malformed in (
        valid.replace(snapshot.digest, "0" * 64),
        valid.replace('"start": 7', '"start": 6, "start": 7'),
        valid + valid,
        valid[:-1] + ', "path": "unrelated.py"}',
    ):
        with pytest.raises(ValueError):
            apply_line_range_edit(malformed, sources, snapshot)


def test_refuses_ambiguous_named_scope_and_inline_function_body():
    assert _prepare({PATH: SOURCE + "\nclass Queue: pass\n"}) is None
    assert _prepare({PATH: "class Queue:\n    def receive(self): return 1\n"}) is None


def test_ranges_keep_crlf_and_replace_absent_final_newline():
    source = "class Queue:\r\n    def receive(self):\r\n        return 1"
    sources = {PATH: source}
    snapshot = _prepare(sources)
    assert (
        apply_line_range_edit(
            _reply(snapshot, _edit(2, 2, "        value = 2", "        return value")),
            sources,
            snapshot,
        )[PATH]
        == "class Queue:\r\n    def receive(self):\r\n        value = 2\r\n        return value"
    )



def test_accepts_one_complete_json_fence_without_other_response_text():
    sources = {PATH: SOURCE}
    snapshot = _prepare(sources)
    valid = _reply(snapshot, _edit(7, 7, "        return 42"))
    expected = apply_line_range_edit(valid, sources, snapshot)
    for tag in ("", "json"):
        fenced = f"```{tag}\n{valid}\n```"
        assert apply_line_range_edit(fenced, sources, snapshot) == expected
        for ambiguous in ("proposal:\n" + fenced, fenced + "\n" + fenced, fenced[:-1]):
            with pytest.raises(ValueError):
                apply_line_range_edit(ambiguous, sources, snapshot)
