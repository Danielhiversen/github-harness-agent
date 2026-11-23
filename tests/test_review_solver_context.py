import ast
import tarfile
from io import BytesIO

import pytest

from github_harness_agent.review_solver_context import (
    focused_check_excerpt,
    focused_source_excerpt,
    stock_pytest_excerpt,
)
from github_harness_agent import review_solver_context
from github_harness_agent.task_context import compact_focused_review
from github_harness_agent.review_solver import _proposal_messages
from github_harness_agent.review_recovery import ReviewRecovery
from github_harness_agent.task_context import MESSAGE_BYTES, wire_size
from github_harness_agent.task_context import TEST_MODULE_REPLY_ENDING
from github_harness_agent import review_solver


@pytest.fixture
def tracked_stock_tests():
    archive = BytesIO()
    test = (
        "class TestWeather:\n"
        "    def test_forecast(self, calibrated):\n"
        + "        # large test setup\n" * 150
        + "        assert calibrated == 24\n"
        "\n    def test_unrelated(self):\n        assert False\n"
    )
    with tarfile.open(fileobj=archive, mode="w") as snapshot:
        for path, contents in {
            "tests/test_weather.py": test,
            "tests/conftest.py": (
                "import pytest\n"
                "@pytest.fixture\n"
                "def calibrated(raw):\n    return raw + 1\n"
                "@pytest.fixture\n"
                "def unrelated():\n    return 'unrelated fixture'\n"
            ),
            "conftest.py": "import pytest\n@pytest.fixture\ndef raw():\n    return 23\n",
        }.items():
            data = contents.encode()
            entry = tarfile.TarInfo(path)
            entry.size = len(data)
            snapshot.addfile(entry, BytesIO(data))
        link = tarfile.TarInfo("tests/test_link.py")
        link.type = tarfile.SYMTYPE
        link.linkname = "test_weather.py"
        snapshot.addfile(link)
    return archive.getvalue()


def test_stock_pytest_excerpt_prioritizes_the_assertion_and_relevant_fixtures(tracked_stock_tests):
    excerpt = stock_pytest_excerpt(
        tracked_stock_tests,
        ("uv", "run", "pytest"),
        "tests/test_weather.py::TestWeather::test_forecast[param]",
        "/work/tests/test_weather.py:153: AssertionError",
    )
    assert excerpt.index("assert calibrated == 24") < excerpt.index("def test_forecast")
    assert "def calibrated(raw):\n    return raw + 1" in excerpt
    assert "def raw():\n    return 23" in excerpt
    assert "unrelated" not in excerpt
    assert len(excerpt) <= 6000


@pytest.mark.parametrize("path", ("/tmp/tests/test_weather.py", "../tests/test_weather.py", "tests/test_link.py", "tests/missing.py"))
def test_stock_pytest_excerpt_rejects_paths_outside_exact_regular_tracked_files(tracked_stock_tests, path):
    assert stock_pytest_excerpt(
        tracked_stock_tests,
        ("pytest",),
        f"{path}::TestWeather::test_forecast",
        "tests/test_weather.py:153: AssertionError",
    ) == ""


def test_fresh_test_prompt_uses_complete_tracked_example_from_reviewed_module():
    archive = BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as snapshot:
        for path, source in {
            "tests/test_other.py": "from other.formatting import format_amount\n"
            "\ndef test_other():\n    assert format_amount(0) == 'unrelated'\n",
            "tests/test_formatting.py": "from example.formatting import format_amount as format_value\n"
            "\ndef test_format_amount():\n"
            + "    # lengthy setup\n" * 180
            + "    assert format_value(0) == 'large'\n"
            "\ndef test_decimal_amount():\n    assert format_value(12) == '12.00'\n",
        }.items():
            data = source.encode()
            entry = tarfile.TarInfo(path)
            entry.size = len(data)
            snapshot.addfile(entry, BytesIO(data))

    production = {"src/example/formatting.py": "def format_amount(value: int) -> str:\n    return str(value)\n"}
    example = review_solver_context.repository_pytest_example(archive.getvalue(), production)
    assert "from example.formatting import format_amount as format_value" in example
    assert "def test_decimal_amount():\n    assert format_value(12) == '12.00'" in example
    assert "test_format_amount" not in example
    assert "unrelated" not in example

    path = "tests/test_review_regressions.py"
    review = ReviewRecovery(
        feedback=(("src/example/formatting.py", 1, "cover decimal amounts"),),
        previous_failure="",
    )
    messages = _proposal_messages(
        (-1, "new regression tests"),
        {**production, path: ""},
        {},
        True,
        "",
        review,
        {},
        [],
        path,
        allow_test_module=True,
        repository_example=example,
    )
    compacted = compact_focused_review(messages, max_bytes=1600)
    for prompt in (messages[-1]["content"], compacted[-1]["content"]):
        assert "format_value(12) == '12.00'" in prompt
        assert prompt.endswith(TEST_MODULE_REPLY_ENDING)
    assert "def format_amount(value: int) -> str" in messages[-1]["content"]


def test_repository_example_supports_module_imports_and_complete_pytest_fixtures():
    archive = BytesIO()
    source = (
        "import pytest\n"
        "from example import formatting as formatter\n"
        "@pytest.fixture\n"
        "def amount_text():\n"
        "    return formatter.format_amount(88)\n"
    )
    with tarfile.open(fileobj=archive, mode="w") as snapshot:
        data = source.encode()
        entry = tarfile.TarInfo("tests/conftest.py")
        entry.size = len(data)
        snapshot.addfile(entry, BytesIO(data))
    example = review_solver_context.repository_pytest_example(
        archive.getvalue(),
        {"src/example/formatting.py": "def format_amount(amount):\n    return str(amount)\n"},
    )
    assert example == "# tests/conftest.py\n" + source.rstrip()


def test_repository_example_includes_module_local_fixture_dependencies():
    archive = BytesIO()
    source = (
        "import pytest as pt\n"
        "from pytest import fixture as setup\n"
        "from pathlib import Path\n"
        "from example.formatting import format_amount\n"
        "from other import fixture\n"
        "@pt.fixture\n"
        "def base_path():\n"
        "    return Path('input.txt')\n"
        "@setup(name='amount')\n"
        "def build_amount(base_path):\n"
        "    return len(base_path.name)\n"
        "@fixture\n"
        "def amount():\n"
        "    return 'unrelated setup'\n"
        "def test_amount(amount):\n"
        "    assert format_amount(amount) == '9'\n"
    )
    with tarfile.open(fileobj=archive, mode="w") as snapshot:
        data = source.encode()
        entry = tarfile.TarInfo("tests/test_formatting.py")
        entry.size = len(data)
        snapshot.addfile(entry, BytesIO(data))

    production = {"src/example/formatting.py": "def format_amount(amount):\n    return str(amount)\n"}
    example = review_solver_context.repository_pytest_example(archive.getvalue(), production)
    assert "Module-local pytest fixture setup" in example
    assert "from pathlib import Path" in example
    assert "import pytest as pt" in example
    assert "from pytest import fixture as setup" in example
    assert "@pt.fixture\ndef base_path():" in example
    assert "@setup(name='amount')\ndef build_amount(base_path):" in example
    assert "def test_amount(amount):" in example
    assert "unrelated setup" not in example
    assert "from other import fixture" not in example


@pytest.mark.parametrize("duplicate", (False, True))
def test_repository_example_skips_oversized_or_ambiguous_local_fixture(duplicate):
    archive = BytesIO()
    source = (
        "import pytest\n"
        "from example.formatting import format_amount\n"
        "@pytest.fixture(name='amount')\n"
        "def provide_amount():\n"
        + ("    return 9\n" if duplicate else "    # long setup\n" * 100 + "    return 9\n")
        + ("@pytest.fixture\ndef amount():\n    return 10\n" if duplicate else "")
        + "def test_amount(amount):\n    assert format_amount(amount) == '9'\n"
        "def test_default():\n    assert format_amount(0) == '0'\n"
    )
    with tarfile.open(fileobj=archive, mode="w") as snapshot:
        data = source.encode()
        entry = tarfile.TarInfo("tests/test_formatting.py")
        entry.size = len(data)
        snapshot.addfile(entry, BytesIO(data))

    example = review_solver_context.repository_pytest_example(
        archive.getvalue(),
        {"src/example/formatting.py": "def format_amount(amount):\n    return str(amount)\n"},
        max_bytes=300,
    )
    assert "def test_default():" in example
    assert "test_amount" not in example
    assert len(example.encode()) <= 300


def test_fresh_test_context_keeps_grounded_usage_from_oversized_tracked_test():
    archive = BytesIO()
    source = (
        "from example.formatting import Formatter, build_formatter\n"
        "from other import calculate\n"
        "def test_small():\n    assert Formatter().ready()\n"
        "class TestCalculation:\n"
        "    def test_calculation(self):\n"
        + "        # long setup\n" * 150
        + "        client = build_formatter()\n"
        "        with mock_context(Formatter(marker='context only')):\n"
        "            total, report = client.calculate(values, precision=3)\n"
        "            assert report.summary == 'hidden expected value'\n"
        "        unrelated = calculate(values, bogus=True)\n"
        "        ignored = helper('private setup')\n"
    )
    with tarfile.open(fileobj=archive, mode="w") as snapshot:
        data = source.encode()
        entry = tarfile.TarInfo("tests/test_formatting.py")
        entry.size = len(data)
        snapshot.addfile(entry, BytesIO(data))
    path = "src/example/formatting.py"
    production = {path: (
        "class Formatter:\n    def ready(self): ...\n"
        "    def calculate(self, values, *, precision): ...\n"
        "def build_formatter() -> Formatter:\n    return Formatter()\n"
    )}
    review = ReviewRecovery(feedback=((path, 3, "calculate report summary"),), previous_failure="")
    example, usage = review_solver_context.repository_pytest_context(
        archive.getvalue(), production, "calculate report summary", usage_bytes=330,
    )
    assert "test_small" in example
    assert "test_calculation" not in example
    assert "client = build_formatter()" in usage
    assert "total, report = client.calculate(values, precision=...)" in usage
    assert "report.summary" in usage
    assert all(item not in usage for item in ("assert", "hidden expected", "bogus", "private setup"))
    assert "mock_context" not in usage and "marker=" not in usage
    assert len(usage.encode()) <= 330

    new_path = "tests/test_review_regressions.py"
    messages = _proposal_messages(
        (-1, "new regression tests"), {**production, new_path: example}, {}, True,
        "NameError: report", review, {}, [], new_path, allow_test_module=True,
        repository_example=example, repository_usage=usage, repair_failed_tests=True,
    )
    compacted = compact_focused_review(messages, max_bytes=2100)
    for prompt in (messages[-1]["content"], compacted[-1]["content"]):
        assert "test_small" in prompt and "precision=..." in prompt
        assert "preserve passing cases" in prompt
        assert "NameError: report" in prompt
        assert prompt.endswith(TEST_MODULE_REPLY_ENDING)


def test_test_repair_prioritizes_the_missing_fixture_and_setup_frame():
    path = "src/example/formatting.py"
    new_path = "tests/test_review_regressions.py"
    node = f"{new_path}::test_formatter"
    output = (
        "===== ERRORS =====\n"
        "_____ ERROR at setup of test_formatter _____\n"
        f"file /work/{new_path}, line 2\n"
        "  def test_formatter(local_formatter):\n"
        "E       fixture 'local_formatter' not found\n"
        "===== short test summary info =====\n"
        f"ERROR {node}\n"
    )
    report = review_solver._test_detail(output, node)
    note = review_solver._failed_test_feedback({(0, node): report}, new_path)
    messages = _proposal_messages(
        (-1, "new regression tests"),
        {
            path: "def build_formatter():\n    return str\n",
            new_path: "def test_formatter(local_formatter):\n    assert local_formatter('x') == 'x'\n",
        },
        {}, True, note,
        ReviewRecovery(feedback=((path, 1, "cover the formatter"),), previous_failure=""),
        {}, [], new_path, allow_test_module=True, repair_failed_tests=True,
    )

    lead = messages[-1]["content"].partition("Edit only")[0]
    assert "fixture 'local_formatter' not found" in lead
    assert f"/work/{new_path}:2" in lead
    assert "a fixture available in this module's scope" in lead


def test_test_only_requirements_survive_compaction_without_operator_inventory():
    path = "src/example/formatting.py"
    new_path = "tests/test_review_regressions.py"
    required = (
        "Construct a nested item, round-trip its identifier, "
        "preserve the original name, and check the empty result."
    )
    body = "Context. " * 41 + required + " Applicability." * 22
    assert 800 <= len(body.encode()) <= 900
    review = ReviewRecovery(feedback=((path, 1, body),), previous_failure="", test_only=True)
    source = "def format_item(item):\n    return item\n"
    fixture = "def test_operator_internal():\n    assert format_item('operator-internal')\n"
    messages = _proposal_messages(
        (-1, "new regression tests"), {path: source, new_path: ""}, {}, True, "",
        review, {"tests/test_operator.py": fixture}, ["bulky context\n" * 150],
        new_path, allow_test_module=True,
    )
    compact = compact_focused_review(messages, max_bytes=2600)

    assert compact is not None and wire_size(compact) <= 2600
    for variant in (messages, compact):
        prompt = variant[-1]["content"]
        assert required in prompt.partition("Edit only")[0]
        assert "Operator test names:" not in prompt
        assert "Operator fixture setup excerpt" not in prompt
        assert "operator-internal" not in prompt
        assert prompt.endswith(TEST_MODULE_REPLY_ENDING)


def test_fresh_test_prompt_distinguishes_requested_fixtures_from_acceptance_inventory():
    path = "src/example/formatting.py"
    new_path = "tests/test_review_regressions.py"
    review = ReviewRecovery(
        feedback=(
            (path, 1, "Include a regression fixture for decimal amounts."),
            (path, 2, "Include a fixture for negative amounts."),
        ),
        previous_failure="",
    )
    messages = _proposal_messages(
        (-1, "new regression tests"),
        {path: "def format_amount(amount):\n    return str(amount)\n", new_path: ""},
        {}, True, "", review,
        {"operator/checks.py": "def test_operator_matrix():\n    assert True\n"},
        [], new_path, allow_test_module=True,
    )

    prompt = messages[-1]["content"]
    assert "two or three focused cases" in prompt
    assert "cover every distinct requested fixture" in prompt
    assert "Combine overlapping examples" in prompt
    assert "test_operator_matrix" in prompt and "already-passing acceptance suite" in prompt
    assert all(body in prompt for _, _, body in review.feedback)


def test_fresh_test_prompt_includes_only_the_reviewed_private_signature_at_mapped_anchor():
    path = "src/example/formatting.py"
    new_path = "tests/test_review_regressions.py"
    base = (
        "def _classify(value: int, *, strict: bool = True) -> str:\n"
        "    result = 'implementation details'\n"
        "    return result\n\n"
        "def format_amount(value):\n"
        "    return _unrelated(value)\n\n"
        "def __dunder__():\n"
        "    return 0\n"
    )
    verified = "def _unrelated(value, unused=None):\n    return str(value)\n\n" + base
    review = ReviewRecovery(
        feedback=(
            (path, 2, "Cover classification with an offline fixture."),
            (path, 6, "Keep public formatting behavior."),
            (path, 9, "Keep the special method behavior."),
        ),
        previous_failure="",
    )
    mapped = review_solver._draft_review(review, {path: base}, {path: verified})
    messages = _proposal_messages(
        (-1, "new regression tests"), {path: verified, new_path: ""}, {}, True,
        "", mapped, {}, [], new_path, allow_test_module=True,
    )
    expanded = [messages[0], {
        **messages[1],
        "content": messages[1]["content"].replace(
            "\nCurrent checkout:\n", "\nCurrent checkout:\n" + "unused checkout data\n" * 500,
        ),
    }]
    compacted = compact_focused_review(expanded, max_bytes=2400)

    for system, user in (messages, compacted):
        assert "Prefer public APIs" in system["content"]
        assert f"{path}\ndef _classify(value: int, *, strict: bool=True) -> str" in user["content"]
        assert all(secret not in user["content"] for secret in (
            "_unrelated", "__dunder__", "implementation details",
        ))
        assert "Cover classification with an offline fixture." in user["content"]


@pytest.mark.parametrize("shared", (True, False))
def test_failed_test_feedback_selects_one_traceback_and_names_other_cases(shared):
    path = "tests/test_review_regressions.py"
    first, second, third = (f"{path}::test_{item}" for item in ("a", "b", "c"))
    failures = {
        (0, first): f"{path}:13: AssertionError\nE   AssertionError: different value",
        (0, second): f"{path}:20: TypeError\nE   TypeError: missing 'inputs'",
        (0, third): "E   TypeError: " + ("missing 'inputs'" if shared else "missing 'output'"),
    }

    note = review_solver._failed_test_feedback(failures, path)

    if shared:
        assert note.startswith(f"Selected failing case: {second}\n")
        assert f"{path}:20: TypeError" in note and "E   TypeError: missing 'inputs'" in note
        assert "may share a fixture or setup error" in note
        assert "different value" not in note
    else:
        assert note.startswith(f"Selected failing case: {first}\n")
        assert f"{path}:13: AssertionError" in note and "different value" in note
        assert "may share" not in note and "missing 'inputs'" not in note
    assert all(node in note for node in (first, second, third))


@pytest.mark.parametrize("scope,await_call", [("with", ""), ("async with", "await ")])
def test_usage_shapes_tracks_with_bodies_without_leaking_rebound_origins(scope, await_call):
    node, = ast.parse(
        f"""async def test_calculation():
    client = build_formatter()
    {scope} mock_context():
        result = {await_call}client.calculate(inside=True)
        {scope} mock_context() as client:
            client.calculate(shadowed=True)
        client.calculate(overwritten=True)
    assert result.not_proven_after_with
    client.calculate(not_restored=True)
    other = build_formatter()
    if flag:
        other = unreviewed()
    other.calculate(possibly_rebound=True)
"""
    ).body
    declarations = ast.parse("""
class Formatter:
    def calculate(self, **options): ...
def build_formatter() -> Formatter: ...
""").body
    reviewed_calls = review_solver_context._reviewed_calls(declarations)
    usage = "\n".join(
        row
        for _, rows in review_solver_context._usage_shapes(node, reviewed_calls)
        for row in rows
    )
    assert f"result = {await_call}client.calculate(inside=...)" in usage
    assert all(
        marker not in usage
        for marker in (
            "mock_context", "shadowed", "overwritten", "not_proven_after_with",
            "not_restored", "possibly_rebound",
        )
    )


def test_usage_shapes_extracts_only_public_keys_from_proven_mapping_results():
    declarations = ast.parse("""
class Formatter:
    def calculate(self) -> tuple[int, dict[str, int]]: ...
    def totals(self) -> dict[str, int]: ...
def build_formatter() -> Formatter: ...
""").body
    node, = ast.parse("""
def test_calculation(uncertain):
    client = build_formatter()
    with fixture_context():
        total, summary = client.calculate()
    assert summary['observed_count'] == 42, 'do not show this assertion'
    assert summary['_private'] == 5
    assert summary['12.3,45.6'] == 5
    assert summary[key_name] == 5
    assert summary['invalid-key'] == 5
    assert uncertain['fixture_key'] == 5
    assert total['wrong_tuple_slot'] == 5
    own = client.totals()
    size = own['public_total']
    _private_mapping = client.totals()
    assert _private_mapping['private_receiver'] == 5
    conflicting = uncertain
    with fixture_context():
        _, conflicting = client.calculate()
    assert conflicting['ambiguous_key'] == 0
""").body
    shapes = dict(review_solver_context._usage_shapes(node, review_solver_context._reviewed_calls(declarations)))
    lookups = {name for name in shapes if "[" in name}
    assert lookups == {"summary['observed_count']", "own['public_total']"}
    assert shapes["summary['observed_count']"][-1] == "summary['observed_count']"
    assert "do not show" not in str(shapes) and "42" not in str(shapes)


def test_operator_test_context_quotes_only_verified_call_shapes_and_mapping_keys():
    path = "src/example/formatter.py"
    sources = {path: """
class Formatter:
    def calculate(self, value, *, precision) -> dict[str, int]: ...
    def unknown(self) -> object: ...
"""}
    fixtures = {"operator/test_contract.py": """
from example.formatter import Formatter as Service
from other import Unrelated
def make_service(secret):
    return Service(secret)
def make_unrelated(secret):
    return Unrelated(secret)
def test_operator_only():
    direct = make_service('operator fixture value').calculate('secret', precision=7)
    assert direct['public_total'] == 987654 and direct['_private'] == 6
    assert direct['12.3,45.6'] in 'hidden expected contents'
    with mock_context():
        client = make_service(123)
        observed = client.calculate(812, precision=8)
    assert observed['observation_count'] == 654321
    unknown = make_service(1).unknown()
    assert unknown['do_not_show'] == 8
    unrelated = make_unrelated(2).calculate(8, unproved=5)
    assert unrelated['also_hidden'] == 53
"""}
    contract = review_solver_context.operator_api_contract(fixtures, sources)
    assert set(contract.splitlines()) == {
        "Formatter.calculate(..., precision=...)",
        "result['public_total']",
        "result['observation_count']",
    }
    review = ReviewRecovery(feedback=((path, 3, "cover calculated totals"),), previous_failure="")
    new_path = "tests/test_review_regressions.py"
    for repairing in (False, True):
        messages = _proposal_messages(
            (-1, "new regression tests"), {**sources, new_path: ""}, {}, True,
            "Verifier expects a mapping" if repairing else "", review, fixtures, [], new_path,
            allow_test_module=True, repair_failed_tests=repairing,
        )
        compacted = compact_focused_review(messages, max_bytes=2300)
        for prompt in (messages[-1]["content"], compacted[-1]["content"]):
            assert "Verified operator API contract (quoted data)" in prompt
            assert contract in prompt
            assert "987654" not in prompt and "654321" not in prompt
            assert "hidden expected contents" not in prompt
            assert prompt.endswith(TEST_MODULE_REPLY_ENDING)
        if repairing:
            prompt = compacted[-1]["content"]
            assert prompt.index("Verifier expects a mapping") < prompt.index(contract)


def test_failed_test_focus_uses_exact_reviewed_traceback_row_and_compacts_it():
    path = "src/example/formatter.py"
    source = (
        "unrelated_top = None\n"
        "class Formatter:\n"
        "    def calculate(self, fixture):\n"
        "        fields = fixture.fields\n"
        "        count = fields['count']\n"
        "        return count + 1\n"
        "\n"
        "unrelated_bottom = None\n"
    )
    note = (
        "  File \"/work/src/example/formatter.py\", line 5, in calculate\n"
        "E KeyError: 'untrusted-runner-value'\n"
        "/elsewhere/src/example/formatter.py:8: KeyError\n"
        "src/example/formatter.py:900: KeyError\n"
    )
    focus = review_solver_context.test_failure_focus({path: source}, note)
    assert focus == f"{path}:5\n" + "".join(source.splitlines(True)[2:7]).rstrip("\n")
    assert "untrusted-runner-value" not in focus and "elsewhere" not in focus
    assert review_solver_context.test_failure_focus(
        {path: source}, f"/elsewhere/{path}:5: KeyError\n{path}:900: KeyError"
    ) == ""

    new_path = "tests/test_review_regressions.py"
    review = ReviewRecovery(feedback=((path, 5, "cover fixture fields"),), previous_failure="")
    messages = _proposal_messages(
        (-1, "new regression tests"), {path: source, new_path: "def test_example(): ...\n"},
        {}, True, note, review, {}, [], new_path, allow_test_module=True, repair_failed_tests=True,
    )
    compacted = compact_focused_review(messages, max_bytes=2650)[-1]["content"]
    for prompt in (messages[-1]["content"], compacted):
        assert "Verified failure source (quoted data)" in prompt
        assert focus in prompt
        assert "preserve passing cases" in prompt
        assert prompt.endswith(TEST_MODULE_REPLY_ENDING)


@pytest.mark.parametrize("prefix,message", (
    ("E   ", "got an unexpected keyword argument 'untrusted-runner-value'"),
    ("tests/test_fixture.py:8: ", "missing 1 required keyword-only argument: 'untrusted-runner-value'"),
))
def test_failed_test_focus_prefers_a_unique_reviewed_typeerror_signature(prefix, message):
    path = "src/example/formatter.py"
    source = "class Formatter:\n    def calculate(self, value, *, precision: int = 2) -> str:\n        ...\n"
    note = f"{path}:3: TypeError\n{prefix}TypeError: Formatter.calculate() {message}"
    focus = review_solver_context.test_failure_focus({path: source}, note)
    assert focus == f"{path}:2\nFormatter.calculate(self, value, *, precision: int=2) -> str"
    matches = {path: source, "src/other/formatter.py": source}
    assert review_solver_context.test_failure_focus(matches, note.splitlines()[-1]) == ""


def test_review_imported_models_use_pinned_package_source_including_relative_imports():
    archive = BytesIO()
    contents = (
        "from dataclasses import dataclass\n"
        "@dataclass\nclass Request:\n    value: int\n    precision: int = 2\n"
        "class Result:\n    def __init__(self, text: str):\n        self.text = text\n"
        "class Unrelated:\n    other: str\n"
    )
    with tarfile.open(fileobj=archive, mode="w") as snapshot:
        data = contents.encode()
        entry = tarfile.TarInfo("src/example/models.py")
        entry.size = len(data)
        snapshot.addfile(entry, BytesIO(data))
    declarations = review_solver_context.imported_model_declarations(
        archive.getvalue(),
        {"src/example/formatting.py": "from .models import Request\n"
         "from example.models import Result\n"
         "def format_amount(value: Request) -> Result:\n    return Result(str(value.value))\n"},
    )
    assert "class Request:\n    value: int\n    precision: int = 2" in declarations
    assert "class Result:\n    def __init__(self, text: str)" in declarations
    assert "Unrelated" not in declarations


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


def test_stock_pytest_context_keeps_full_source_when_it_fits_the_cloud_request():
    source = "def main():\n    return items()\n" + "# padding\n" * 2800
    source += "def neighboring_helper():\n    return 12345\n"
    target = (1, "tests/test_items.py::test_selection")
    review = ReviewRecovery((("module.py", 1, "Preserve selection order"),), "")
    messages = _proposal_messages(
        target, {"module.py": source}, {target: "AssertionError"}, False, "",
        review, {}, (), "", stock_fixture="def test_selection():\n    assert main()",
    )
    assert source in messages[-1]["content"]
    assert wire_size(messages) <= MESSAGE_BYTES


@pytest.mark.parametrize(
    ("command", "diagnostic"),
    [
        (("ruff", "check", "."), "I001 [*] Imports unsorted\n --> core.py:1:1\n"),
        (("ruff", "check", "."), "core.py:2:1: F401 Unused import\n"),
        (("mypy", "."), "core.py:7: error: Unexpected type [assignment]\n"),
        (("pytest",), "/work/core.py:7: AssertionError: Unexpected value\n"),
    ],
)
def test_check_location_precedes_historical_review_in_compact_prompt(
    command, diagnostic
):
    core = (
        "import zlib\nimport array\n\nfrom pathlib import (\n    Path,\n)\n"
        "VALUE = 32\n\n" + "x = 1\n" * 1600
    )
    target = (0, "check 1 (exit 1)")
    review = ReviewRecovery((("reviewed.py", 2, "Update the original feature"),), "")
    messages = _proposal_messages(
        target,
        {"core.py": core, "reviewed.py": "def previous():\n    pass\n"},
        {target: diagnostic},
        False,
        "",
        review,
        {},
        (),
        "",
        check_command=command,
    )
    compact = compact_focused_review(messages, max_bytes=4900)
    prompt = compact[-1]["content"]
    assert prompt.index("Focused source from core.py:") < prompt.index(
        "Reviewer feedback:"
    )
    assert "from pathlib import (\n    Path,\n)" in prompt
    if command[0] == "ruff":
        assert "import zlib\nimport array" in prompt
    assert "Current checkout:" not in prompt


def test_check_focus_rejects_unrecognized_tools_and_external_paths():
    sources = {"core.py": "import array\n"}
    assert (
        focused_check_excerpt(
            ("custom-check",), "core.py:1:1: F401 Unused import", sources
        )
        is None
    )
    assert (
        focused_check_excerpt(
            ("pytest",), "/operator/core.py:1: AssertionError", sources
        )
        is None
    )


def test_compact_review_keeps_middle_of_caller_and_selected_fixture():
    filler = "".join(f"    ignored_{n} = {n}\n" for n in range(350))
    scheduling = (
        "    available = load_index(items)\n"
        "    pending = []\n"
        "    for candidate in available:\n"
        "        if candidate.ready:\n"
        "            pending.append(dispatch(candidate))\n"
    )
    source = (
        "def load_index(items):\n    return Index().select(items)\n\n"
        "def publish(items):\n"
        f"{filler}{scheduling}{filler}"
        "    return pending\n"
        "\nclass Index:\n"
        "    def __init__(self):\n"
        + "".join("    " + row for row in filler.splitlines(keepends=True))
        + "        self.segments: list[tuple[int, int]] = []\n"
        + "".join("    " + row for row in filler.splitlines(keepends=True))
        + "\n    def register(self, segment):\n"
        "        self.segments.append(segment)\n"
        "\n    def select(self, items):\n"
        "        return sorted(items)\n"
    )
    target = (0, "test_publish.py::test_selected")
    fixtures = {
        "test_publish.py": (
            "def setup_noise():\n"
            + filler
            + "def test_selected():\n    assert publish(inputs) == submitted\n"
            + "def test_unrelated():\n    assert not publish([])\n"
        )
    }
    review = ReviewRecovery((("module.py", 2, "Keep scheduling order"),), "")
    messages = _proposal_messages(
        target,
        {"module.py": source},
        {target: "Bad order"},
        False,
        "Earlier proposal omitted the dispatch body",
        review,
        fixtures,
        ("unrelated imported interface\n" * 400,),
        "",
    )
    compact = compact_focused_review(messages, max_bytes=12 * 1024)
    assert compact is not None
    assert wire_size(compact) <= 12 * 1024
    content = compact[1]["content"]
    assert scheduling in content
    assert "def publish(items):" in content
    assert "self.segments: list[tuple[int, int]] = []" in content
    assert "self.segments.append(segment)" in content
    assert "def select(self, items):" in content
    assert "assert publish(inputs) == submitted" in content
    assert "Keep scheduling order" in content
    assert "Earlier proposal omitted the dispatch body" in content
    assert "def test_unrelated" not in content
def test_stock_failure_site_logs_only_the_tracked_location_and_exception():
    from github_harness_agent.review_solver_context import stock_failure_site

    node = "tests/test_counts.py::test_report"
    assert stock_failure_site(node, "tests/test_counts.py:8: KeyError: 'private value'") == "line 8 (KeyError)"
    assert stock_failure_site(node, "/work/tests/test_counts.py:9: assert 2 == 1") == "line 9 (AssertionError)"
    assert stock_failure_site(node, "other.py:8: AssertionError") == ""



def test_short_pytest_frame_localizes_only_the_selected_test_and_evaluated_assert():
    from github_harness_agent.review_solver_context import stock_assertion_evaluated, stock_failure_site
    from github_harness_agent.review_solver import _test_detail

    path = "tests/test_counts.py"
    node = f"{path}::TestCounts::test_count[one]"
    frame = f"{path}:3: in test_count"
    source = b"class TestCounts:\n    def test_count(self, report):\n        assert report['count'] == 1\n"
    archive = BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as snapshot:
        entry = tarfile.TarInfo(path)
        entry.size = len(source)
        snapshot.addfile(entry, BytesIO(source))
    def detail(error):
        output = (
            "=================================== FAILURES ===================================\n"
            "_________________________ TestCounts.test_count[one] _________________________\n"
            f"{frame}\n    assert report['count'] == 1\nE   {error}\n"
            "=========================== short test summary info ============================\n"
            f"FAILED {node} - {error}\n"
        )
        return _test_detail(output, node)

    lookup = detail("KeyError: 'count'")
    evaluated = detail("assert 2 == 1")
    assert stock_failure_site(node, lookup) == "line 3 (KeyError)"
    assert stock_failure_site(node, evaluated) == "line 3 (AssertionError)"
    assert stock_assertion_evaluated(archive.getvalue(), node, lookup, evaluated)
    assert not stock_failure_site(node, lookup.replace("in test_count", "in helper"))
    assert not stock_failure_site(node, lookup.replace("E   ", "src/helper.py:4: in helper\nE   "))
    assert not stock_assertion_evaluated(archive.getvalue(), node, lookup, evaluated.replace(":3:", ":4:"))


def _tracked_assertion_archive(source):
    archive = BytesIO()
    content = source.encode()
    with tarfile.open(fileobj=archive, mode="w") as snapshot:
        member = tarfile.TarInfo("tests/test_counts.py")
        member.size = len(content)
        snapshot.addfile(member, BytesIO(content))
    return archive.getvalue()


@pytest.mark.parametrize(
    "old, new, expected",
    (
        ("5 == 1", "3 == 1", True),
        ("-4 == 1", "-2 == 1", True),
        ("5 == 1", "5 == 1", False),
        ("3 == 1", "5 == 1", False),
        ("5 == 1", "0 == 1", False),
        ("5 == 1", "3 == 2", False),
        ("5 == 1", "True == 1", False),
        ("5 == 1", "3.0 == 1", False),
        ("5 == 1", "'3' == 1", False),
        ("5 == 1", "03 == 1", False),
        ("5 == 1", "3 == 1 or False", False),
        ("5 == 1", "3 == 1\nE    assert 2 == 1", False),
    ),
)
def test_stock_integer_progress_requires_a_strict_same_side_literal(old, new, expected):
    from github_harness_agent.review_solver_context import stock_assertion_progress

    archive = _tracked_assertion_archive(
        "def test_count():\n    assert read_count() == 1\n"
    )
    node = "tests/test_counts.py::test_count"
    before = f"tests/test_counts.py:2: assert {old}"
    after = f"/work/tests/test_counts.py:2: in test_count\nE   assert {new}"
    assert bool(stock_assertion_progress(archive, node, before, after)) is expected
    assert not stock_assertion_progress(
        archive, node, before, after, changed_paths={"tests/test_counts.py"}
    )


@pytest.mark.parametrize(
    "expression",
    (
        "read_count() == True",
        "read_count() == expected",
        "read_count() == 1 == 1",
        "5 == 1",
    ),
)
def test_stock_integer_progress_requires_an_immutable_literal_comparison(expression):
    from github_harness_agent.review_solver_context import stock_assertion_progress

    archive = _tracked_assertion_archive(
        f"def test_count():\n    assert {expression}\n"
    )
    assert not stock_assertion_progress(
        archive,
        "tests/test_counts.py::test_count",
        "tests/test_counts.py:2: assert 5 == 1",
        "tests/test_counts.py:2: assert 3 == 1",
    )


@pytest.mark.parametrize(
    "body, old, new, expected",
    (
        ("    assert first()\n    assert second()\n", 2, 3, True),
        ("    assert first()\n    assert second()\n", 3, 2, False),
        ("    assert first()\n    if enabled:\n        assert second()\n", 2, 4, False),
        (
            "    assert first()\n    if enabled:\n        pass\n    assert second()\n",
            2,
            5,
            False,
        ),
        (
            "    if enabled:\n        assert first()\n        assert second()\n",
            3,
            4,
            True,
        ),
        (
            "    for item in (0,):\n        assert first()\n        assert second()\n",
            3,
            4,
            True,
        ),
        (
            "    for item in (0, 1):\n        assert first()\n        assert second()\n",
            3,
            4,
            False,
        ),
        (
            "    for item in items:\n        assert first()\n        assert second()\n",
            3,
            4,
            False,
        ),
        ("    assert first(); assert hidden()\n    assert second()\n", 2, 3, False),
    ),
)
def test_stock_later_assertion_progress_requires_unambiguous_straight_line_suite(
    body, old, new, expected
):
    from github_harness_agent.review_solver_context import stock_assertion_progress

    archive = _tracked_assertion_archive("def test_count():\n" + body)
    before = f"tests/test_counts.py:{old}: assert False"
    after = f"tests/test_counts.py:{new}: assert False"
    assert (
        bool(
            stock_assertion_progress(
                archive, "tests/test_counts.py::test_count", before, after
            )
        )
        is expected
    )
