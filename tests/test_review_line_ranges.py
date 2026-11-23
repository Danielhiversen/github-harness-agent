import json

import pytest

from github_harness_agent.review_line_ranges import (
    apply_line_range_edit,
    apply_offered_line_reply,
    apply_native_line_range_edit,
    line_range_prompt,
    select_line_snapshot,
    full_method_native_tool,
    selectable_line_tool,
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


def test_native_line_ranges_use_only_the_pending_snapshot_and_displayed_rows():
    sources = {PATH: SOURCE}
    snapshot = _prepare(sources)
    arguments = json.dumps({"edits": [_edit(7, 7, "        return result + 2")]})
    function = {"name": "replace_python_line_ranges", "arguments": arguments}
    call = {"type": "function", "function": function}
    rows = frozenset({7})

    revised = apply_native_line_range_edit((call,), sources, snapshot, allowed_rows=rows)
    assert revised[PATH] == SOURCE.replace("return result + 1", "return result + 2")
    invalid = (
        (),
        (call, call),
        ({**call, "function": {**function, "name": "replace_python_definition"}},),
        ({**call, "function": {**function, "arguments": arguments[:-1] + ', "path": "other.py"}'}},),
        ({**call, "function": {**function, "arguments": arguments[:-1] + ', "edits": []}'}},),
        ({**call, "function": {**function, "arguments": arguments[:-1] + ', "snapshot": "fake"}'}},),
    )
    for calls in invalid:
        with pytest.raises(ValueError):
            apply_native_line_range_edit(calls, sources, snapshot, allowed_rows=rows)
    with pytest.raises(ValueError, match="shown scoped edit windows"):
        apply_native_line_range_edit((call,), sources, snapshot, allowed_rows=frozenset({6}))
    with pytest.raises(ValueError, match="does not match"):
        apply_native_line_range_edit((call,), revised, snapshot, allowed_rows=rows)


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

def test_selected_snapshot_rejects_unoffered_hash_and_duplicate_json_keys():
    snapshot = _prepare({PATH: SOURCE})
    for response in (
        '{"snapshot":"unoffered","edits":[]}',
        f'{{"snapshot":"unoffered","snapshot":"{snapshot.digest}","edits":[]}}',
    ):
        with pytest.raises(ValueError):
            select_line_snapshot(response, (snapshot,))

def test_native_selection_binds_alternate_body_to_offered_hash():
    sources = {PATH: SOURCE}
    initial = _prepare(sources)
    alternate = prepare_line_range_edit(sources, (PATH, "Queue.unrelated"))
    field = selectable_line_tool((initial, alternate))["function"]["parameters"]
    assert field["required"] == ["snapshot", "edits"]
    assert field["properties"]["snapshot"]["enum"] == [initial.digest, alternate.digest]
    arguments = _reply(alternate, _edit(2, 2, "        return 10"))
    calls = ({"type": "function", "function": {"name": "replace_python_line_ranges", "arguments": arguments}},)
    revised = apply_native_line_range_edit(calls, sources, initial, allowed_rows=frozenset({6}), alternatives=(alternate,))
    assert revised[PATH] == SOURCE.replace("return 9", "return 10")

@pytest.mark.parametrize("form", ("fence", "symbol", "prefix", "qualified", "symbol_as_id", "block", "implicit_block", "unified"))
def test_free_text_edit_binds_only_the_selected_current_caller(form):
    import difflib
    from github_harness_agent.review_solver import _BLOCK

    sources = {PATH: SOURCE}
    anchor = _prepare(sources)
    caller = prepare_line_range_edit(sources, (PATH, "Queue.unrelated"))
    expected = SOURCE.replace("return 9", "return 10")
    edit = _edit(2, 2, "        return 10")
    if form == "fence":
        reply = "```json\n" + _reply(caller, edit) + "\n```"
    elif form in {"block", "implicit_block"}:
        reply = f"{PATH}\n<<<<<<< SEARCH\n        return 9\n=======\n        return 10\n>>>>>>> REPLACE"
        if form == "implicit_block":
            reply = reply.partition("\n")[2]
    elif form == "symbol_as_id":
        reply = json.dumps({"path": PATH, "snapshot": "Queue.unrelated", "edits": [edit]})
    elif form == "unified":
        reply = "".join(difflib.unified_diff(SOURCE.splitlines(True), expected.splitlines(True), f"a/{PATH}", f"b/{PATH}"))
    elif form == "symbol":
        reply = json.dumps({"symbol": "Queue.unrelated", "edits": [edit]})
    elif form == "qualified":
        reply = json.dumps({"symbol": PATH + "::Queue.unrelated", "snapshot": caller.digest[:12], "edits": [edit]})
    else:
        reply = json.dumps({"snapshot": caller.digest[:12], "edits": [edit]})
    assert apply_offered_line_reply(reply, sources, (anchor, caller), block_pattern=_BLOCK) == {PATH: expected}
    stale = {PATH: "# changed before reply\n" + SOURCE}
    with pytest.raises(ValueError):
        apply_offered_line_reply(reply, stale, (anchor, caller), block_pattern=_BLOCK)


@pytest.mark.parametrize("form", ("ambiguous", "outside", "signature", "mixed", "no_symbol", "wrong_hash", "too_big", "wrong_file"))
def test_free_text_edit_preserves_existing_scope_and_change_limits(form):
    from github_harness_agent.review_solver import _BLOCK

    text = SOURCE + "\ndef another():\n        return 9\n" if form == "ambiguous" else SOURCE
    sources = {PATH: text}
    anchor = _prepare(sources)
    caller = prepare_line_range_edit(sources, (PATH, "Queue.unrelated"))
    old, new = {
        "ambiguous": ("        return 9", "        return 10"),
        "outside": ("after = 2", "after = 3"),
        "signature": ("    def unrelated(self):", "    def unrelated(self, x):"),
        "mixed": ("        return 9\n\nafter = 2", "        return 10\n\nafter = 3"),
        "too_big": ("        return 9", "        pass\n" * 24 + "        return 10"),
    }.get(form, ("        return 9", "        return 10"))
    reply = f"{PATH}\n<<<<<<< SEARCH\n{old}\n=======\n{new}\n>>>>>>> REPLACE"
    if form == "no_symbol":
        reply = json.dumps({"edits": [_edit(2, 2, new)]})
    elif form == "wrong_hash":
        reply = json.dumps({"snapshot": anchor.digest, "symbol": "Queue.unrelated", "edits": [_edit(2, 2, new)]})
    elif form == "wrong_file":
        reply = reply.replace(PATH, "other.py")
    with pytest.raises(ValueError):
        apply_offered_line_reply(reply, sources, (anchor, caller), block_pattern=_BLOCK)

def _long_body(prefix=90):
    source = (
        "# file prefix\n" * prefix + "class Processor:\n    def process(self):\n"
        + "".join(f"        item_{i} = {i}\n" for i in range(32))
        + "        return item_0\n\n    def stable(self):\n        return 88\n"
    )
    sources = {PATH: source}
    snapshot = prepare_line_range_edit(sources, (PATH, "Processor.process"))
    return sources, snapshot, frozenset(range(snapshot.first_body_line, len(snapshot.lines) + 1))


def test_echoed_context_and_full_single_method_rewrite_use_actual_changed_rows():
    from github_harness_agent.review_solver import _BLOCK

    sources, snapshot, allowed = _long_body()
    echoed = list(snapshot.lines[1:])
    echoed[0] = "        item_0 = 100"
    echoed[-2] = "        item_31 = 131"
    reply = _reply(snapshot, _edit(2, len(snapshot.lines), *echoed))
    partial = apply_line_range_edit(reply, sources, snapshot, allowed_rows=allowed)
    assert partial[PATH] == sources[PATH].replace("item_0 = 0", "item_0 = 100").replace("item_31 = 31", "item_31 = 131")
    widened = [f"        item_{i} = {100+i}" for i in range(20)] + list(snapshot.lines[21:])
    reply = _reply(snapshot, _edit(2, len(snapshot.lines), *widened))
    with pytest.raises(ValueError, match=r"old=20, new=20, max=24"):
        apply_line_range_edit(reply, sources, snapshot, allowed_rows=allowed)
    expected = sources[PATH]
    for i in range(20):
        expected = expected.replace(f"item_{i} = {i}\n", f"item_{i} = {100+i}\n")
    result = apply_offered_line_reply(reply, sources, (snapshot,), block_pattern=_BLOCK, full_method=True)
    assert result[PATH] == expected and "def stable(self):\n        return 88" in result[PATH]
    params = full_method_native_tool(snapshot)["function"]["parameters"]["properties"]["edits"]
    assert params["maxItems"] == 6 and params["items"]["properties"]["lines"]["maxItems"] >= len(snapshot.lines)
    calls = ({"type":"function","function":{"name":"replace_python_line_ranges","arguments":json.dumps({"edits": json.loads(reply)["edits"]})}},)
    assert apply_native_line_range_edit(calls, sources, snapshot, allowed_rows=allowed, full_method=True) == result


def test_full_method_absolute_rows_must_be_unambiguous_and_contained():
    sources, snapshot, allowed = _long_body()
    absolute_start = sources[PATH].splitlines().index(snapshot.lines[1]) + 1
    absolute = _reply(snapshot, _edit(absolute_start, absolute_start, "        item_0 = 100"))
    expected = sources[PATH].replace("item_0 = 0", "item_0 = 100")
    assert apply_line_range_edit(absolute, sources, snapshot, allowed_rows=allowed, full_method=True)[PATH] == expected
    for rows in (allowed, allowed - {3}):
        with pytest.raises(ValueError, match="numbered definition rows"):
            apply_line_range_edit(absolute, sources, snapshot, allowed_rows=rows, full_method=rows != allowed)
    for reply in (
        _reply(snapshot, _edit(2, 2, "        pass"), _edit(absolute_start, absolute_start, "        pass")),
        _reply(snapshot, _edit(absolute_start + len(snapshot.lines) + 1, absolute_start + len(snapshot.lines) + 1, "        pass")),
        _reply(snapshot, _edit(1, 1, "    def different(self):")),
    ):
        with pytest.raises(ValueError, match="numbered definition rows"):
            apply_line_range_edit(reply, sources, snapshot, allowed_rows=allowed, full_method=True)
    # Absolute row 6 also names an offered relative row; keep the documented relative interpretation.
    nearby, near, rows = _long_body(prefix=2)
    actual = apply_line_range_edit(_reply(near, _edit(6, 6, "        pass")), nearby, near, allowed_rows=rows, full_method=True)
    assert actual[PATH] == nearby[PATH].replace(near.lines[5], "        pass")


def test_full_and_partial_method_limits_report_counts_without_content():
    sources, snapshot, allowed = _long_body()
    records = tuple(_edit(2+i, 2+i, f"        confidential_{i} = 0") for i in range(4))
    with pytest.raises(ValueError, match=r"changed ranges=4, max=3") as err:
        apply_line_range_edit(_reply(snapshot, *records), sources, snapshot, allowed_rows=allowed)
    assert "confidential" not in str(err.value)
    assert apply_line_range_edit(_reply(snapshot, *records), sources, snapshot, allowed_rows=allowed, full_method=True)
    outside = _reply(snapshot, _edit(2, 4, *snapshot.lines[1:4]))
    with pytest.raises(ValueError, match=r"outside rows=1, ranges=1"):
        apply_line_range_edit(outside, sources, snapshot, allowed_rows=allowed-{3}, full_method=True)
    huge = _reply(snapshot, _edit(2, 26, *["        pass"] * 25))
    with pytest.raises(ValueError, match=r"old=25, new=25, max=48"):
        apply_line_range_edit(huge, sources, snapshot, allowed_rows=allowed, full_method=True)
    invalid = _reply(snapshot, _edit(2, 2, "        if"))
    with pytest.raises(ValueError, match="valid Python"):
        apply_line_range_edit(invalid, sources, snapshot, allowed_rows=allowed, full_method=True)
    assert "confidential" not in sources[PATH]
