import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from github_harness_agent.checkpoint import DraftCheckpoint
from github_harness_agent.executor import PlainExecutor
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


@pytest.mark.parametrize("initial", ["valid", "mixed", "exhausted", "missing_test"])
def test_recovery_reads_the_restored_tree_and_returns_to_the_normal_verified_rollout(
    fixture,
    initial,
):
    task, source, runner = fixture
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
    task = replace(
        task,
        base_sha=base,
        checkpoints={draft.key: draft},
        checks=((sys.executable, "-c", "import module; assert module.limit == 4"),),
        require_new_tests=initial == "missing_test",
        review_recovery=plan_recovery(
            feedback(),
            3,
            "the last draft did not address all threads. Add new runnable regression tests."
            if initial == "missing_test"
            else "the last draft did not address all threads",
        ),
    )
    offers = []

    class Gateway:
        def complete(self, **kwargs):
            offers.append([tool["function"]["name"] for tool in kwargs["tools"]])
            if len(offers) == 1:
                request = kwargs["messages"][1]["content"]
                assert "limit = 2" in request
                assert "limit = 1" not in request
                assert "last draft did not address all threads" in request
                if initial == "exhausted":
                    raise ProviderExhausted(None)
                calls = (edit("module.py"),)
                if initial == "missing_test":
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
                if initial == "mixed":
                    calls += ({**edit("unrelated.py"), "id": "unrelated"},)
            elif len(offers) == 2 and initial in {"mixed", "exhausted"}:
                if initial == "mixed":
                    assert kwargs["messages"][-1]["content"].startswith(
                        "The recovery batch requires"
                    )
                # The exact old text still matches because the rejected batch
                # applied neither edit before this ordinary retry.
                calls = (edit("module.py"),)
            else:
                assert kwargs["messages"][-1]["content"].startswith("exit 0")
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

    result = PlainExecutor(runner, Gateway()).execute(task, source)
    assert offers == [
        ["edit_file"],
        *([["run", "edit_file"]] * (2 if initial in {"mixed", "exhausted"} else 1)),
    ]
    assert b"+limit = 4" in result.patch
    if initial == "missing_test":
        assert b"test_recovery_sample.py" in result.patch
    assert result.checks[0].returncode == 0
