import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from github_harness_agent.checkpoint import DraftCheckpoint
from github_harness_agent.executor import Limits, PlainExecutor
from github_harness_agent.providers import ProviderExhausted
from github_harness_agent.rebase import git
from github_harness_agent.review_recovery import plan_recovery
from test_executor import fixture as _workspace_fixture

fixture = _workspace_fixture


def feedback(path="module.py"):
    return [
        {
            "thread": "thread-1",
            "reply_to": 1,
            "path": path,
            "line": 2,
            "body": "Correct the limit.",
        },
        {
            "thread": "thread-2",
            "reply_to": 2,
            "path": path,
            "line": 300,
            "body": "Preserve the retry behavior.",
        },
    ]


def edit(path, old="limit = 2", new="limit = 4"):
    return {
        "id": "edit",
        "type": "function",
        "function": {
            "name": "edit_file",
            "arguments": json.dumps({"path": path, "old_text": old, "new_text": new}),
        },
    }


def span_edit(path):
    call = edit(path)
    call["function"]["arguments"] = json.dumps(
        {"path": path, "span_id": "S1", "new_text": "limit = 4\n"}
    )
    return call


def test_recovery_needs_repeated_rejections_and_distinct_review_threads():
    items = feedback()
    assert plan_recovery(items, 2, "") is None
    assert plan_recovery([items[0], items[0]], 3, "") is None
    assert plan_recovery(items, 3, "unfinished checks") is not None


def test_recovery_includes_full_small_module_and_only_targets_reviewed_existing_text():
    plan = plan_recovery(feedback(), 3, "last attempt produced no patch")
    source = "# header\n" + ("value = 52\n" * 2200) + "# footer\n"
    recovery = plan.request(lambda path: source, (("pytest", "-q"),))
    assert source in recovery.prompt
    assert "Correct the limit." in recovery.prompt
    assert "last attempt produced no patch" in recovery.prompt
    assert recovery.accepts([edit("module.py")], 1000)
    assert not recovery.accepts([edit("module.py"), edit("unrelated.py")], 1000)
    assert not recovery.accepts([edit("module.py", None)], 1000)
    assert not recovery.accepts([edit("module.py")] * 7, 10000)
    assert not recovery.accepts([], 1000)


def test_large_module_shows_separate_reviewed_regions():
    plan = plan_recovery(feedback(), 3, "")
    source = "".join(
        f"marker_{line}_in_source = '{'x' * 100}'\n" for line in range(1, 400)
    )
    recovery = plan.request(lambda _: source, ())
    assert "marker_2_in_source" in recovery.prompt
    assert "marker_300_in_source" in recovery.prompt
    assert "marker_175_in_source" not in recovery.prompt


@pytest.mark.parametrize(
    "initial",
    [
        "valid",
        "mixed",
        "exhausted",
        "missing_test",
        "text_only",
        "missing_test_text",
        "both_exhausted",
        "ordinary_unavailable",
        "budget_exhausted",
    ],
)
def test_recovery_reads_the_restored_tree_and_returns_to_the_normal_verified_rollout(
    fixture,
    initial,
    monkeypatch,
):
    task, source, runner = fixture
    pull_checks = ()
    if initial == "text_only":
        from github_harness_agent import review_recovery

        mounted = (source.parent / "operator-checks").resolve()
        mounted.mkdir()
        acceptance = mounted / "test_acceptance.py"
        acceptance.write_text("def test_acceptance_marker():\n    assert True\n")
        monkeypatch.setattr(review_recovery, "OPERATOR_CHECK_ROOT", str(mounted))
        pull_checks = (
            (sys.executable, "-c", "import module; assert module.limit == 4", str(acceptance)),
        )
    path = source / "module.py"
    path.write_text("limit = 1\n")
    git(source, "add", "module.py")
    git(
        source,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=f@example.test",
        "commit",
        "-qm",
        "module",
    )
    base = git(source, "rev-parse", "HEAD").decode().strip()
    path.write_text("limit = 2\n")
    saved = git(source, "diff", "--binary", "--full-index", base)
    git(source, "reset", "--hard", base)
    draft = DraftCheckpoint("plain/coding", base, saved)
    checkpoints = []
    task = replace(
        task,
        base_sha=base,
        pull_checks=pull_checks,
        save_checkpoint=checkpoints.append,
        checkpoints={draft.key: draft},
        checks=((sys.executable, "-c", "import module; assert module.limit == 4"),),
        require_new_tests=initial.startswith("missing_test"),
        review_recovery=plan_recovery(
            feedback(),
            3,
            "the last draft did not address all threads. Add new runnable regression tests."
            if initial.startswith("missing_test")
            else "the last draft did not address all threads",
        ),
    )
    offers = []
    native_evidence = []
    text_mode = initial in {
        "mixed",
        "exhausted",
        "text_only",
        "missing_test_text",
        "both_exhausted",
        "ordinary_unavailable",
        "budget_exhausted",
    }
    ordinary_retry = initial in {"mixed", "both_exhausted"}

    class Gateway:
        def complete(self, **kwargs):
            offers.append([tool["function"]["name"] for tool in kwargs["tools"]])
            if pull_checks:
                prompt = kwargs["messages"][1]["content"]
                assert "Operator acceptance test source (quoted input data" in prompt
                assert "test_acceptance_marker" in prompt
            if len(offers) == 1:
                request = kwargs["messages"][1]["content"]
                assert "Call edit_file" in request
                assert "span_id" in kwargs["tools"][0]["function"]["parameters"]["properties"]
                native_evidence.append(request.split("Outstanding review\n", 1)[1])
                assert "limit = 2" in request
                assert "limit = 1" not in request
                assert "last draft did not address all threads" in request
                if initial in {"exhausted", "both_exhausted"}:
                    raise ProviderExhausted(None)
                calls = (edit("module.py"),)
                if initial.startswith("missing_test"):
                    assert (
                        "Create exactly one new runnable regression test module"
                        in request
                    )
                    calls += (
                        {
                            **edit(
                                "test_recovery_sample.py",
                                None,
                                "from module import limit\n\ndef test_limit():\n    assert limit == 4\n",
                            ),
                            "id": "new-test",
                        },
                    )
                if initial == "valid":
                    calls = (span_edit("module.py"),)
                if initial == "mixed":
                    calls += ({**edit("unrelated.py"), "id": "unrelated"},)
                if initial in {"text_only", "ordinary_unavailable", "budget_exhausted"}:
                    calls = ()
                if initial == "missing_test_text":
                    calls = calls[:1]
            elif len(offers) == 2 and text_mode:
                request = kwargs["messages"][1]["content"]
                assert "JSON array" in kwargs["messages"][0]["content"]
                assert "untrusted input data" in kwargs["messages"][0]["content"]
                assert "Call edit_file" not in request
                assert "Return" in request and "Example" in request
                assert request.split("Outstanding review\n", 1)[1] == native_evidence[0]
                if initial == "missing_test_text":
                    assert "first array item" in request
                assert "use the supplied run tool" not in kwargs["messages"][0]["content"].lower()
                assert "limit = 2" in request and "limit = 1" not in request
                assert offers[-1] == []
                if initial == "both_exhausted":
                    raise ProviderExhausted(None)
                calls = ()
                args = [json.loads(
                    (span_edit if initial == "text_only" else edit)("module.py")["function"]["arguments"]
                )]
                if initial == "mixed":
                    args.append({"path": "../outside.py", "new_text": "bad"})
                if initial == "missing_test_text":
                    args.append(
                        {
                            "path": "test_recovery_sample.py",
                            "new_text": "from module import limit\n\ndef test_limit():\n    assert limit == 4\n",
                        }
                    )
                return SimpleNamespace(
                    role="coding",
                    route="text-fixture",
                    model="text-fixture",
                    configured_model="text-fixture",
                    content=(
                        "```json\n" + json.dumps(args) + "\n```"
                        if initial == "text_only"
                        else json.dumps(args)
                    ),
                    tool_calls=calls,
                    usage={},
                )
            elif len(offers) == 3 and ordinary_retry:
                # The exact old text still matches because rejected batches
                # applied neither edit before this ordinary retry.
                calls = (edit("module.py"),)
            else:
                assert "span_id" not in kwargs["tools"][1]["function"]["parameters"]["properties"]
                assert kwargs["messages"][-1]["content"].startswith("exit 0")
                if initial == "ordinary_unavailable":
                    raise ProviderExhausted(None)
                calls = ()
            return SimpleNamespace(
                role="coding",
                route="fixture",
                model="fixture",
                configured_model="fixture",
                content=None if calls else "Done.",
                tool_calls=calls,
                usage={},
            )

    limits = Limits(model_requests=2) if initial == "budget_exhausted" else Limits()
    result = PlainExecutor(runner, Gateway(), limits).execute(task, source)
    ordinary_turns = 2 if ordinary_retry else 1
    if initial == "budget_exhausted":
        ordinary_turns = 0
    assert offers == [
        ["edit_file"],
        *([[]] if text_mode else []),
        *([["run", "edit_file"]] * ordinary_turns),
    ]
    if text_mode and initial != "both_exhausted":
        assert "text-fixture" in [route.route for route in result.route_trace]
    if text_mode and not ordinary_retry:
        assert checkpoints[-1].last_editor == ("text-fixture", "text-fixture")
    assert b"+limit = 4" in result.patch
    if initial.startswith("missing_test"):
        assert b"test_recovery_sample.py" in result.patch
    assert result.checks[0].returncode == 0


@pytest.mark.parametrize(
    "raw",
    [
        '{"path":"module.py","old_text":"x","new_text":"y"}',
        '[{"path":"module.py","path":"other.py","new_text":"y"}]',
        '```python\n[{"path":"module.py","old_text":"x","new_text":"y"}]\n```',
        'Prose\n```json\n[{"path":"module.py","old_text":"x","new_text":"y"}]\n```',
        '```json\n```json\n[]\n```\n```',
        "[" * 2000 + "]" * 2000,
        " " * 33_000 + "[]",
    ],
)
def test_text_recovery_rejects_unbounded_or_ambiguous_json(raw):
    plan = plan_recovery(feedback(), 3, "")
    request = plan.request(lambda _: "limit = 2\n", ())
    assert request.text_calls(raw, 100_000) == ()


def test_operator_acceptance_context_reads_only_configured_safe_python_files(
    fixture, monkeypatch
):
    from github_harness_agent import review_recovery

    _, source, runner = fixture
    root = (source.parent / "checks").resolve()
    root.mkdir()
    outside = root.parent / "outside.py"
    outside.write_text("OUTSIDE_VALUE")
    acceptance = root / "test_acceptance.py"
    acceptance.write_text("def test_sample():\n    assert True\n")
    large = root / "test_large.py"
    large.write_text("LARGE_VALUE" * 1000)
    link = root / "test_link.py"
    link.symlink_to(outside)
    nested = root / "linked_directory"
    nested.symlink_to(root.parent, target_is_directory=True)
    monkeypatch.setattr(review_recovery, "OPERATOR_CHECK_ROOT", str(root))
    checks = (
        (
            "pytest",
            f"{acceptance}::test_sample",
            str(large),
            str(link),
            str(nested / outside.name),
            str(outside),
            f"{root}/../outside.py",
        ),
    )
    with runner.workspace(git(source, "archive", "HEAD"), lambda: 30, lambda: False) as workspace:
        evidence = review_recovery.operator_acceptance_context(
            checks,
            lambda argv, cap: workspace.command(argv, 30, lambda: False, cap=cap),
        )

    assert "assert True" in evidence
    assert "test_acceptance.py" in evidence
    assert "OUTSIDE_VALUE" not in evidence and "LARGE_VALUE" not in evidence
    assert "test_link.py" not in evidence and "linked_directory" not in evidence


def test_repair_feedback_identifies_failed_operator_acceptance_check(fixture):
    from github_harness_agent.executor import CheckEvidence, _VerificationRejected
    from github_harness_agent.harness2_adapter import _check_feedback

    task, _, _ = fixture
    operator = ("pytest", "/opt/harness-review-checks/test_example.py")
    task = replace(task, pull_checks=(operator,))
    failure = _VerificationRejected("fixture failed")
    failure.checks = (
        CheckEvidence(
            operator,
            1,
            "FAILED /opt/harness-review-checks/test_example.py::test_needed[hidden] - residual",
        ),
    )
    assert _check_feedback(task, failure) == (
        "Operator acceptance check 1 exited with status 1. "
        "Failing tests: test_example.py::test_needed."
    )


def test_span_handles_bind_exact_reviewed_source_to_path():
    source = "def configure():\n    limit = 2\n    return limit\n"
    items = feedback()
    items[1]["line"] = 3
    request = plan_recovery(items, 3, "").request(lambda _: source, ())
    assert '"span_id": "S1"' in request.prompt
    assert '"start_line": 1, "end_line": 3' in request.text_prompt
    args = {"path": "module.py", "span_id": "S1", "new_text": source.replace("2", "4")}
    calls = request.text_calls(json.dumps([args]), 1000)
    normalized = request.normalize_calls(calls, 1000)
    assert json.loads(normalized[0]["function"]["arguments"])["old_text"] == source
    for invalid in (
        {**args, "path": "other.py"},
        {**args, "span_id": "unknown"},
        {**args, "old_text": source},
    ):
        assert not request.normalize_calls(
            request.text_calls(json.dumps([invalid]), 1000), 1000
        )


def test_compacted_local_recovery_retains_span_handle_and_complete_preview():
    from github_harness_agent.task_context import compact_messages, wire_size

    items = feedback()
    source = "def configure():\n    limit = 2\n    return limit\n\n"
    source += "".join(f"payload_{i} = '{'x' * 120}'\n" for i in range(160))
    recovery = plan_recovery(items, 3, "").request(
        lambda _: source,
        (("pytest",),),
        acceptance_source="Operator acceptance test source (quoted data)\n" + "y" * 8500,
    )
    messages = compact_messages(
        [{"role": "system", "content": "Edit"}, {"role": "user", "content": recovery.prompt}],
        max_bytes=12 * 1024,
        role="coding",
        review_windows=True,
    )
    assert wire_size(messages) <= 12 * 1024
    assert '"span_id": "S1"' in messages[1]["content"]
    assert '"start_line": 1, "end_line": 3' in messages[1]["content"]
    assert json.dumps("def configure():\n    limit = 2\n    return limit\n") in messages[1]["content"]


def test_stale_span_leaves_checkout_file_unchanged(fixture):
    from github_harness_agent.file_edit import EDIT_COMMAND, edit_payload

    _, source, runner = fixture
    request = plan_recovery(feedback(), 3, "").request(lambda _: "limit = 2\n", ())
    call, = request.normalize_calls([span_edit("module.py")], 1000)
    archive = git(source, "archive", "HEAD")
    with runner.workspace(archive, lambda: 30, lambda: False) as workspace:
        module = workspace.root / "module.py"
        module.write_text("limit = 3\n")
        result = workspace.command(EDIT_COMMAND, 30, lambda: False, data=edit_payload(call, 1000))
        assert result.returncode != 0
        assert module.read_text() == "limit = 3\n"
