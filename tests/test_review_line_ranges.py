import json

import pytest

from github_harness_agent.review_line_ranges import (
    apply_line_range_edit,
    apply_offered_line_reply,
    apply_native_line_range_edit,
    line_range_prompt,
    literal_reference_rows,
    StatementFragment,
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


@pytest.mark.parametrize("form", ("implicit", "python", "indented"))
def test_sole_method_text_rechecks_snapshot_and_preserves_other_definitions(form):
    from github_harness_agent.review_solver import _BLOCK

    sources = {PATH: SOURCE}
    snapshot = prepare_line_range_edit(sources, (PATH, "Queue.unrelated"))
    raw = "def unrelated(self):\n    return 10"
    reply = {
        "implicit": json.dumps({"edits": [_edit(2, 2, "        return 10")]}),
        "python": "```python\n" + raw + "\n```",
        "indented": "    def unrelated(self):\n        return 10",
    }[form]
    def apply(current, choices):
        return apply_offered_line_reply(reply, current, choices, block_pattern=_BLOCK, full_method=True)
    assert apply(sources, (snapshot,)) == {PATH: SOURCE.replace("return 9", "return 10")}
    with pytest.raises(ValueError):
        apply({PATH: "# stale\n" + SOURCE}, (snapshot,))
    with pytest.raises(ValueError):
        apply(sources, (snapshot, _prepare(sources)))


@pytest.mark.parametrize("reply", (
    "def different(self):\n    return 10", "def unrelated(self, other):\n    return 10",
    "async def unrelated(self):\n    return 10", "@property\ndef unrelated(self):\n    return 10",
    "def unrelated(self):\n    return 10\ndef foreign():\n    return 12",
    "def unrelated(self):\n    return (", "def unrelated(self):\n" + "    pass\n" * 50,
    json.dumps({"path": "other.py", "edits": [_edit(2, 2, "        return 10")]}),
    json.dumps({"snapshot": "unknown", "edits": [_edit(2, 2, "        return 10")]}),
))
def test_sole_method_text_rejects_foreign_invalid_or_oversize_replies(reply):
    from github_harness_agent.review_solver import _BLOCK

    sources = {PATH: SOURCE}
    snapshot = prepare_line_range_edit(sources, (PATH, "Queue.unrelated"))
    with pytest.raises(ValueError):
        apply_offered_line_reply(reply, sources, (snapshot,), block_pattern=_BLOCK, full_method=True)


def test_scoped_blocks_apply_atomically_against_original_method():
    from github_harness_agent.review_solver import _BLOCK

    sources, snap, _ = _long_body()
    def block(old, new, path=""):
        return f"{path + chr(10) if path else ''}<<<<<<< SEARCH\n{old}\n=======\n{new}\n>>>>>>> REPLACE"
    a = "        item_0 = 0"
    b = "        item_1 = 1"
    c = "        item_2 = 2"
    first = block(a, a + "\n        inserted = 1", PATH)
    last = block(c, c + "0")
    def apply(reply):
        return apply_offered_line_reply(reply, sources, (snap,), block_pattern=_BLOCK, full_method=True)
    reply = last + "\n\n" + first
    assert apply(reply)[PATH] == sources[PATH].replace(a, a + "\n        inserted = 1").replace(c, c + "0")
    for suffix in (
        block(a + "\n" + b, a + "\n" + b + "0"), block("        inserted = 1", b),
        block(b, c, "other.py"), block("        return 88", b), block(b, "         invalid +"),
        "prose", "<<<<<<< SEARCH", block(b, b + "\n" + "\n".join(["        pass"] * 49)),
    ):
        with pytest.raises(ValueError):
            apply(first + "\n" + suffix)
    assert "inserted" not in sources[PATH]


def test_statement_fragment_uses_exact_suite_and_declines_scattered_or_stale_changes():
    from github_harness_agent.review_line_ranges import apply_statement_fragment, isolate_statement_fragment

    sources, snapshot, _ = _long_body()
    updates = sources[PATH].replace("item_3 = 3", "item_3 = -3")
    tentative = {PATH: updates}
    edited = prepare_line_range_edit(tentative, snapshot.symbol)
    selected = isolate_statement_fragment(snapshot, edited)
    assert selected.accepted == ("        item_3 = 3",)
    fenced = "```python\nitem_3 = 8\n```"
    assert apply_statement_fragment("Here is the fix:\n" + fenced + "\nDone.", tentative, edited, selected)[PATH] == updates.replace("item_3 = -3", "item_3 = 8")
    for extra in ("return 5", "other.py", "X" * 241, fenced):
        with pytest.raises(ValueError):
            apply_statement_fragment(extra + "\n" + fenced, tentative, edited, selected)
    result = apply_statement_fragment(
        "```python\nif item_1:\n    item_3 = 8\nelse:\n    item_3 = -3\n```", tentative, edited, selected,
    )
    assert result[PATH] == updates.replace("        item_3 = -3", "        if item_1:\n            item_3 = 8\n        else:\n            item_3 = -3")
    scattered = prepare_line_range_edit({PATH: updates.replace("item_31 = 31", "item_31 = -31")}, snapshot.symbol)
    assert isolate_statement_fragment(snapshot, scattered) is None
    with pytest.raises(ValueError, match="snapshot"):
        apply_statement_fragment("item_3 = 8", sources, edited, selected)
    for invalid in ("if item_1:", "def other_method(): return 8", 'item_3 = """unterminated', "pass\n"*19):
        with pytest.raises(ValueError):
            apply_statement_fragment(invalid, tentative, edited, selected)


def test_complete_method_accepts_seven_small_disjoint_edits_with_original_limits():
    from github_harness_agent.review_solver import _BLOCK

    sources, snapshot, _ = _long_body()
    lines = list(snapshot.lines)
    records = []
    for i in range(0, 14, 2):
        row = lines.index(f"        item_{i} = {i}")
        lines[row] += " + 1"
        records.append(_edit(row + 1, row + 1, lines[row]))

    def apply(reply):
        return apply_offered_line_reply(reply, sources, (snapshot,), block_pattern=_BLOCK, full_method=True)

    method = "\n".join(lines)
    result = apply(method)
    assert result[PATH] == sources[PATH].replace("\n".join(snapshot.lines), method)
    with pytest.raises(ValueError, match="range count outside parser limit"):
        apply(json.dumps({"edits": records}))
    with pytest.raises(ValueError):
        apply(method + "\n    def unrelated(self):\n        return 123")
    oversized = [line + " + 1" if "item_" in line and "=" in line else line for line in lines]
    with pytest.raises(ValueError, match="changed-row limit"):
        apply("\n".join(oversized))


@pytest.mark.parametrize("gap", ("        # explain input\n\n        # explain output\n\n", "        tick()\n" * 4))
def test_tiny_hunks_bridge_only_bounded_context(gap):
    from github_harness_agent.review_line_ranges import select_statement_fragment, apply_statement_fragment

    padding = "".join(f"        a_{i} = {i}\n" for i in range(25))
    before = "class Reader:\n    def collect(self):\n" + padding + "        x = 0\n" + gap + "        y = 0\n        return x + y\n"
    after = before.replace("x = 0", "x = 1").replace("y = 0", "y = 1")
    old, new = [prepare_line_range_edit({PATH: source}, (PATH, "Reader.collect")) for source in (before, after)]
    decision = select_statement_fragment(old, new)
    assert decision.reason == "common-suite" and (decision.hunks, decision.old_rows, decision.new_rows) == (2, 2, 2)
    assert decision.fragment.accepted == tuple(("        x = 0\n"+gap+"        y = 0").splitlines())
    actual = apply_statement_fragment("x = 3\ny = 4", {PATH: after}, new, decision.fragment)
    assert actual[PATH] == before.replace("        x = 0\n"+gap+"        y = 0", "        x = 3\n        y = 4")


@pytest.mark.parametrize("mutation", ("insert", "delete"))
def test_pure_edits_offer_a_surviving_statement_on_both_checkouts(mutation):
    from github_harness_agent.review_line_ranges import select_statement_fragment, apply_statement_fragment

    sources, old, _ = _long_body()
    original = sources[PATH]
    line = "        item_3 = 3\n"
    with_step = original.replace(line, line+"        item_3 += 1\n")
    before, after = (original, with_step) if mutation == "insert" else (with_step, original)
    old, new = [prepare_line_range_edit({PATH: source}, old.symbol) for source in (before, after)]
    decision = select_statement_fragment(old, new)
    assert decision.reason == "common-suite" and decision.hunks == 1
    offered = new.lines[decision.fragment.first-1:decision.fragment.last]
    stable = set(offered) & set(decision.fragment.accepted)
    assert len(stable) == 1
    replacement = list(offered)
    changed = "        item_3 += 1"
    if changed in replacement:
        replacement[replacement.index(changed)] = "        item_3 += 2"
    else:
        replacement.insert(0 if offered[0] != line.rstrip() else 1, "        item_3 += 2")
    result = apply_statement_fragment("\n".join(replacement), {PATH: after}, new, decision.fragment)
    assert result[PATH] == original.replace(line, line+"        item_3 += 2\n")


def test_literal_hunk_preserves_external_syntax_and_rejects_new_execution_scope():
    from github_harness_agent.review_line_ranges import select_statement_fragment, apply_statement_fragment, preceding_local_names

    wide = "".join(f"            total += {i}\n" for i in range(27))
    prefix = "class Reader:\n    def collect(self, legacy):\n        total = 0\n"
    suffix = "        return total\n\n    def unchanged(self):\n        return 47\n"
    accepted = prefix + "        if False:\n" + wide + suffix
    tentative = prefix + "        use = True\n        gate = use\n        if gate:\n" + wide + suffix
    old, new = [prepare_line_range_edit({PATH: source}, (PATH, "Reader.collect")) for source in (accepted, tentative)]
    selection = select_statement_fragment(old, new)
    span = selection.fragment
    assert selection.reason == "single-literal-hunk" and span.literal and span.last-span.first == 2
    assert preceding_local_names(new, span.first) == ("total",)
    result = apply_statement_fragment("if not legacy:", {PATH: tentative}, new, span)
    assert result[PATH] == prefix + "        if not legacy:\n" + wide + suffix
    for wrong in ("def nested():", "class Wrapper:", "if broken(", 's = """', "return 4", "pass\n"*13):
        with pytest.raises(ValueError):
            apply_statement_fragment(wrong, {PATH: tentative}, new, span)
    with pytest.raises(ValueError, match="snapshot"):
        apply_statement_fragment("if not legacy:", {PATH: accepted}, new, span)



def test_literal_blank_edges_retain_indentation_and_restore_only_verified_blank_rows():
    from github_harness_agent.review_line_ranges import apply_statement_fragment, select_statement_fragment

    prefix = "class Reader:\n    def score(self, legacy):\n        result = (\n            0\n"
    prefix += "".join(f"            + 0  # prefix {i}\n" for i in range(27))
    suffix = "            + 0  # unchanged suffix\n        )\n        return result\n\n    def guard(self): return 9\n"
    blank = " " * 12 + "\n"
    added = "\n            + (4\n               )\n"
    original, tentative = prefix + blank + suffix, prefix + added + suffix
    def snapshot(source):
        return prepare_line_range_edit({PATH: source}, (PATH, "Reader.score"))
    old, new = snapshot(original), snapshot(tentative)
    choice = select_statement_fragment(old, new)
    assert (choice.reason, choice.old_rows, choice.new_rows) == ("single-literal-hunk", 1, 3)
    fragment = choice.fragment
    assert fragment.accepted == (blank[:-1],)
    assert apply_statement_fragment("RESTORE_ACCEPTED_BLANK", {PATH: tentative}, new, fragment) == {PATH: original}
    replacement = "+ (0 if legacy else 4)"
    fixed = apply_statement_fragment(replacement, {PATH: tentative}, new, fragment)
    assert fixed == {PATH: prefix + "            " + replacement + "\n" + suffix}
    for source, expected in ((original, (0, 0)), (tentative, (4, 4)), (fixed[PATH], (4, 0))):
        namespace = {}
        exec(source, namespace)
        reader = namespace["Reader"]()
        assert (reader.score(False), reader.score(True)) == expected
        assert reader.guard() == 9
    reverse = select_statement_fragment(new, old).fragment
    assert reverse.literal
    assert apply_statement_fragment(replacement, {PATH: original}, old, reverse) == fixed
    with pytest.raises(ValueError):
        apply_statement_fragment("RESTORE_ACCEPTED_BLANK", {PATH: original}, old, reverse)
    for broken in ('"""', ")", "def escaped(): pass", ""):
        with pytest.raises(ValueError):
            apply_statement_fragment(broken, {PATH: tentative}, new, fragment)



def test_literal_hunk_can_restore_verified_else_attachment_without_reverting_behavior():
    from github_harness_agent.review_line_ranges import apply_statement_fragment, select_statement_fragment

    prefix = "class Reader:\n    def score(self, active):\n        value = 0\n        if active:\n"
    prefix += "".join(f"            value += 0  # context {i}\n" for i in range(26))
    suffix = "        else:\n            value = 10\n        return value\n"
    blank = " " * 7 + "\n"
    inserted = "        if not active:\n            value = 6\n            value += 1\n"
    accepted, tentative = prefix + blank + suffix, prefix + inserted + suffix
    def snap(source):
        return prepare_line_range_edit({PATH: source}, (PATH, "Reader.score"))
    old, new = snap(accepted), snap(tentative)
    fragment = select_statement_fragment(old, new).fragment
    assert fragment.literal and len(fragment.accepted) == 1
    fixed = apply_statement_fragment("elif not active: value = 7", {PATH: tentative}, new, fragment)
    assert fixed[PATH] == prefix + "        elif not active: value = 7\n" + suffix
    for version, expected in ((accepted, (0, 10)), (tentative, (10, 7)), (fixed[PATH], (0, 7))):
        namespace = {}
        exec(version, namespace)
        reader = namespace["Reader"]()
        assert (reader.score(True), reader.score(False)) == expected
    from dataclasses import replace
    wrong_reference = replace(fragment, accepted_digest="0" * 64)
    with pytest.raises(ValueError, match=r"exterior Python syntax \(AST\)"):
        apply_statement_fragment("elif not active: value = 7", {PATH: tentative}, new, wrong_reference)
    # Even an authentic prior digest is unusable if the rest of the repository
    # differs; the old fingerprint cannot be spliced into another checkout.
    extra = {PATH: tentative, "other.py": "different_checkout = 1\n"}
    extra_snapshot = prepare_line_range_edit(extra, new.symbol)
    with pytest.raises(ValueError, match="exterior Python syntax"):
        apply_statement_fragment("elif not active: value = 7", extra, extra_snapshot, fragment)
    assert apply_statement_fragment("RESTORE_ACCEPTED_BLANK", {PATH: tentative}, new, fragment)[PATH] == accepted



def test_literal_window_counts_changes_and_caps_comment_and_blank_allowance():
    from github_harness_agent.review_line_ranges import apply_statement_fragment, select_statement_fragment

    prefix = "class Selector:\n    def choose(self, flagged):\n        value = 0\n"
    suffix = "".join(f"            value += {i}\n" for i in range(25)) + "        return value\n"
    old = prefix + "        if False:\n" + suffix
    current = prefix + "        apply = True\n        select = apply\n        if select:\n" + suffix
    key = (PATH, "Selector.choose")
    original = prepare_line_range_edit({PATH: old}, key)
    snapshot = prepare_line_range_edit({PATH: current}, key)
    fragment = select_statement_fragment(original, snapshot).fragment
    assert fragment.literal and fragment.last-fragment.first == 2
    # Three removed rows and 21 new substantive rows. Four decoration rows cost no additional change budget.
    setup = ["criteria = [flagged]"] + [f"criteria.append({i} >= 0)" for i in range(19)]
    rows = [*setup, "# The final decision uses every criterion.", "", "# Choose from this input.", "", "if all(criteria):"]
    result = apply_statement_fragment("\n".join(rows), {PATH: current}, snapshot, fragment)
    assert result[PATH] == prefix + "".join("        "+line+"\n" if line else "\n" for line in rows) + suffix
    namespace = {}
    exec(result[PATH], namespace)
    assert namespace["Selector"]().choose(True) == sum(range(25))
    assert namespace["Selector"]().choose(False) == 0
    for addition in ("criteria.append(True)", "# fifth decoration"):
        with pytest.raises(ValueError, match="changed-row limit"):
            apply_statement_fragment("\n".join([addition, *rows]), {PATH: current}, snapshot, fragment)
    with pytest.raises(ValueError, match=r"submitted=33, max=32"):
        apply_statement_fragment("\n".join(["# filler"]*8 + rows), {PATH: current}, snapshot, fragment)
    data = ['text = """', '# string content', '# string content', '# string content', '"""', *setup, "if all(criteria):"]
    with pytest.raises(ValueError, match="changed-row limit"):
        apply_statement_fragment("\n".join(data), {PATH: current}, snapshot, fragment)


def test_literal_reference_rows_show_remote_bindings_and_uses_in_same_method():
    source = """def collect(values):
    pending = []
    note = "pending appears in a string"
    before(pending)
    def nested():
        pending = [42]
    for value in values:
        pending.append(value)
        untouched(value)
    checkpoint()
    checkpoint()
    checkpoint()
    checkpoint()
    drain(pending)
    return note
"""
    snapshot = prepare_line_range_edit({PATH: source}, (PATH, "collect"))
    accepted_source = source.replace("pending.append(value)", "pending.append(note)")
    accepted = prepare_line_range_edit({PATH: accepted_source}, snapshot.symbol)
    fragment = StatementFragment(8, 8, (accepted.lines[7],), literal=True)

    assert literal_reference_rows(snapshot, accepted, fragment, visible_first=5, visible_last=12) == (
        "2 |     pending = []",
        '3 |     note = "pending appears in a string"',
        "4 |     before(pending)",
        "14 |     drain(pending)",
        "15 |     return note",
    )
