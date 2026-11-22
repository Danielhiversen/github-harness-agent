import json
from types import SimpleNamespace

from github_harness_agent.providers import ProviderExhausted
from github_harness_agent.review_planner import (
    _SOURCE_LIMIT,
    _advice,
    apply_coding_advice,
    request_coding_advice,
)


def test_planner_accepts_one_fenced_json_object():
    inventory = {"demo.py::run": "def run():\n    pass\n"}
    content = '```json\n{"strategy": "Review existing flow.", "symbols": ["demo.py::run"]}\n```'
    assert _advice(content, inventory) == (
        "Review existing flow.",
        "demo.py::run\ndef run():\n    pass\n",
    )
    assert _advice("unrelated prefix\n" + content, inventory) is None


def test_planner_keeps_complete_selected_definitions_within_source_budget():
    first = "def first():\n    " + "pass  # padded\n    " * 220
    omitted = "def omitted():\n    " + "pass  # padded\n    " * 220
    last = "def last():\n    pass\n"
    inventory = {
        "demo.py::first": first,
        "demo.py::omitted": omitted,
        "demo.py::last": last,
    }
    _, sources = _advice(
        '{"strategy": "Review existing flow.", "symbols": ["demo.py::first", "demo.py::omitted", "demo.py::last"]}',
        inventory,
    )
    assert sources == f"demo.py::first\n{first}\ndemo.py::last\n{last}"
    assert len(sources) <= _SOURCE_LIMIT


def test_brief_planner_timeout_preserves_normal_coding_prompt():
    messages = [
        {"role": "system", "content": "Coding task"},
        {
            "role": "user",
            "content": "Selected failure: test_x\nFocused source from demo.py:\ncode\nFailure details:\nfailure",
        },
    ]
    calls = []

    class Budget:
        def complete(self, gateway, role, payload, **options):
            calls.append((gateway, role, payload, options))
            raise ProviderExhausted(None, {"gateway": "request timed out"})

    executor = SimpleNamespace(gateway=object())
    advice = request_coding_advice(
        executor,
        Budget(),
        messages,
        {"demo.py": "def run():\n    pass\n"},
        "def test_x(): ...",
        "test_x failed",
    )
    assert apply_coding_advice(messages, advice) is messages
    assert len(calls) == 1
    assert calls[0][1] == "planning"
    assert 25 < calls[0][3]["timeout_cap"] <= 60
    assert calls[0][3]["max_output_tokens"] == 4096
    assert "code" in calls[0][2][-1]["content"].split("Relevant current implementation", 1)[1]
    assert calls[0][3]["retry_oversize"] is False


def test_detailed_plan_keeps_only_existing_symbols_and_preserves_source_evidence():
    inventory = {"demo.py::update": "def update():\n    pass\n"}
    plan = "Trace the sample from the failing fixture using the current implementation. " * 16
    response = json.dumps({"strategy": plan, "symbols": ["demo.py::new_helper", "demo.py::update"]})
    strategy, bodies = _advice(response, inventory)
    assert strategy == plan.strip()
    assert bodies == "demo.py::update\ndef update():\n    pass\n"


def test_stock_planner_keeps_delta_contracts_and_exact_search_source():
    padding = "        # long method\n" * 320
    before = (
        "class Queue:\n    def receive(self, key):\n"
        + padding
        + "        return self.old_lookup(key)\n"
    )
    current = before.replace("old_lookup", "new_lookup")
    path = "src/demo/queue.py"
    messages = [
        {"role": "system", "content": "Use SEARCH/REPLACE."},
        {
            "role": "user",
            "content": "Selected failure: tests/test_queue.py::test_tracked\n"
            "Focused source from src/demo/queue.py:\n# abbreviated\n"
            "Failure details:\nstock failure\n"
            "Fixture for this goal:\nread-only test\nCurrent checkout:\n"
            "old prompt source\nRespond with exact SEARCH/REPLACE blocks only.",
        },
    ]
    fixtures = {
        "operator.py": """
def test_unrelated():
    assert other("unrelated value")

def test_verified(queue):
    given = "external"
    value = queue.receive(given)
    assert value == "verified result"
""",
    }
    notes = [(path, i, f"Independent review note {i}") for i in range(1, 5)]
    calls = []

    def complete(gateway, role, messages, **options):
        calls.extend(messages)
        return SimpleNamespace(
            content=json.dumps(
                {
                    "strategy": "Compare the arguments in the two cases.",
                    "symbols": ["src.demo.queue.Queue.receive"],
                }
            )
        )

    plan = request_coding_advice(
        SimpleNamespace(gateway=object()),
        SimpleNamespace(complete=complete),
        messages,
        {path: current},
        'def test_tracked(queue):\n    assert queue.receive("internal") == "stock result"\n',
        "stock failure",
        "",
        ({path: before}, fixtures, notes),
    )
    planner_prompt = calls[-1]["content"]
    assert f"{path}::Queue.receive" in planner_prompt
    assert "-        return self.old_lookup(key)" in planner_prompt
    assert "+        return self.new_lookup(key)" in planner_prompt
    for prompt in (planner_prompt, apply_coding_advice(messages, plan)[-1]["content"]):
        assert "queue.receive(given)" in prompt
        assert 'given = "external"' in prompt
        assert 'assert value == "verified result"' in prompt
        assert "unrelated value" not in prompt
        assert all(body in prompt for _, _, body in notes)
    coding = apply_coding_advice(messages, plan)[-1]["content"]
    assert f"{path}::Queue.receive" in coding
    assert "        return self.new_lookup(key)" in coding
    assert "old_lookup" not in coding
    assert coding.endswith("Respond with exact SEARCH/REPLACE blocks only.")


def test_stock_coding_prompt_keeps_complete_large_selected_method():
    path = "pkg/directory.py"
    setup = "".join(f"        values.append({i})\n" for i in range(160))
    finish = "".join(f"        values.extend([{i}])\n" for i in range(160, 320))
    method = (
        "    def resolve(self, query):\n"
        "        values = []\n"
        + setup
        + "        if query.use_shared_pool:\n"
        "            values = self.lookup_shared(query)\n"
        + finish
        + "        return values\n"
    )
    assert 8000 <= len(method) <= 12000
    draft = "class Directory:\n" + method
    baseline = "class Directory:\n    def resolve(self, query):\n        return []\n"
    messages = [
        {"role": "system", "content": "Use exact SEARCH/REPLACE."},
        {
            "role": "user",
            "content": "Selected failure: test_lookup\n"
            "Focused source from pkg/directory.py:\n# abandoned focus\n"
            "Failure details:\nlookup failed\n"
            "Fixture for this goal:\nread-only test\nCurrent checkout:\n",
        },
    ]

    def complete(gateway, role, messages, **options):
        assert method in messages[-1]["content"]
        assert options["timeout_cap"] == 120
        return SimpleNamespace(
            content=json.dumps(
                {
                    "strategy": "Preserve the intended pool selection.",
                    "symbols": ["pkg.directory.Directory.resolve"],
                }
            )
        )

    plan = request_coding_advice(
        SimpleNamespace(gateway=object()),
        SimpleNamespace(complete=complete),
        messages,
        {path: draft},
        "def test_lookup(directory, query):\n    assert directory.resolve(query)\n",
        "lookup failed",
        "Auxiliary counterfactual (quoted data, partial diagnostic only). "
        "pkg/directory.py::Directory.resolve, draft line 164: `if query.use_shared_pool`.",
        stock_inputs=({path: baseline}, {}, []),
    )
    coding = apply_coding_advice(messages, plan)[-1]["content"]
    assert method in coding
    assert coding.endswith("Respond with exact SEARCH/REPLACE blocks only.")


def test_stock_overview_preserves_call_modes_across_large_operator_cases():
    from github_harness_agent.review_delta_context import _operator_examples

    fixture = "".join(
        f"def test_mode_{index}(service):\n"
        + ("    # offline setup for this call\n" * 30)
        + f"    result = service.fetch(mode={index})\n"
        + f"    assert result == {index}\n\n"
        for index in range(6)
    )
    evidence = _operator_examples({"test_modes.py": fixture}, {"fetch"})
    for index in range(6):
        assert f"test_mode_{index}: service.fetch(mode={index}) -> result == {index}" in evidence


def test_private_operator_source_localizes_a_large_selected_method():
    from github_harness_agent.review_delta_context import private_operator_source

    source = (
        "class Reader:\n    def fetch(self):\n        value = 0\n"
        + "        value += 1\n" * 100
        + "        multiplier = 123\n"
        + "        value -= 1\n" * 100
        + "        return value * multiplier\n"
    )
    before = {"reader.py": source}
    after = {"reader.py": source.replace("multiplier = 123", "multiplier = 456")}
    fixture = {
        "checks/test_reader.py": (
            "class TestReader:\n    def test_fetch(self, mode):\n"
            "        assert Reader().fetch() == mode\n"
        )
    }
    node = "../mounted/checks/test_reader.py::TestReader::test_fetch[mode]"

    excerpt = private_operator_source(before, after, fixture, node)

    assert "reader.py::Reader.fetch" in excerpt
    assert "# Exact current lines" in excerpt
    assert "# Exact verified donor lines" in excerpt
    assert "        multiplier = 456\n" in excerpt
    assert "        multiplier = 123\n" in excerpt
    assert len(excerpt) <= 3200
    other = "def fetch():\n    return 0\n"
    assert (
        private_operator_source(
            {**before, "other.py": other}, {**after, "other.py": other}, fixture, node
        )
        == ""
    )
