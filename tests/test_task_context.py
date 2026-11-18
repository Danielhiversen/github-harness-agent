import json

from github_harness_agent.task_context import (
    build_task_context,
    compact_messages,
    github_history,
    wire_size,
)


def test_large_issue_and_feedback_preserve_previews_and_allow_full_retrieval():
    issue = "Start of issue\n" + "æ😊" * 18000 + "\nLast requirement"
    review = {
        "key": "change-1",
        "author": "maintainer",
        "path": "src/archive.py",
        "line": 42,
        "body": "Fix the path selection.\n" + "x" * 18000,
        "diff_hunk": "@@ -1 +1 @@\n-old\n+new\n" + "x\n" * 8000,
    }
    context = build_task_context(
        issue,
        repository_guide="Style rules\n" + "g" * 16000,
        feedback=[review, {"body": "Add a regression test.", "line": 53}]
        + [{"body": "Follow up item " + str(i)} for i in range(40)],
        previous_failure="Latest attempt failed the trusted check.",
    )
    assert len(context.instructions.encode()) < 18000
    assert "Last requirement" in context.instructions
    assert "Latest attempt failed the trusted check." in context.instructions
    assert "Fix the path selection" in context.instructions
    assert "+new" in context.instructions
    assert "Add a regression test." in context.instructions
    assert "review-42" in context.instructions
    assert (
        json.loads(
            context.instructions.split("Review requests against this commit:\n")[1]
        )[0]["line"]
        == 42
    )

    fragments = []
    page = 1
    while page is not None:
        result = context.read("task", page)
        fragments.append(result["content"])
        page = result["next_page"]
    assert "".join(fragments) == issue
    assert "Follow up item 39" in context.read("review-42", 1)["content"]


def test_scoped_github_history_is_lazy_cached_and_can_page_an_active_thread():
    class API:
        requests = []

        def rest(self, method, path, *, params):
            self.requests.append((method, path, params))
            return [{"body": "Older issue comment", "user": {"login": "member"}}]

        def graphql(self, query, variables):
            self.requests.append(("graphql", variables))
            second = variables["cursor"] is not None
            return {
                "node": {
                    "comments": {
                        "nodes": [
                            {
                                "body": "reply" if second else "review root",
                                "state": "SUBMITTED",
                            }
                        ],
                        "pageInfo": {"hasNextPage": not second, "endCursor": "cursor"},
                    }
                }
            }

    api = API()
    feedback = [{"key": "requested-change", "thread": "allowed-node", "reply_to": 8}]
    context = build_task_context(
        "Fix issue",
        feedback=feedback,
        history=github_history(api, "org/repo", 3, 4, feedback),
    )
    assert not api.requests
    assert "pull-reviews" in context.read("sections", 1)["content"]
    assert "Unknown" in context.read("https://example.invalid/secret", 1)["content"]
    assert not api.requests
    assert "Older issue comment" in context.read("issue-comments", 1)["content"]
    assert context.read("issue-comments", 1) == context.read("issue-comments", 1)
    assert context.read("review-thread-1", 2)["content"].endswith("reply")
    assert context.read("review-thread-1", 1)["content"].endswith("review root")
    assert api.requests == [
        ("GET", "/repos/org/repo/issues/3/comments", {"per_page": 4, "page": 1}),
        ("graphql", {"thread": "allowed-node", "cursor": None}),
        ("graphql", {"thread": "allowed-node", "cursor": "cursor"}),
    ]


def test_compaction_preserves_recent_tool_turns_and_gemini_metadata():
    messages = [
        {"role": "system", "content": "Follow repository guidance."},
        {"role": "user", "content": "Fix the archive."},
    ]
    for i in range(20):
        messages += [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": str(i),
                        "type": "function",
                        "function": {"name": "run", "arguments": "{}"},
                    }
                ],
                "gateway_route": ["google", "gemma"],
                "provider_context": {
                    "route": "google",
                    "model": "gemma",
                    "parts": [{"thoughtSignature": str(i)}],
                },
            },
            {
                "role": "tool",
                "tool_call_id": str(i),
                "content": "exit 1\nLatest error detail"
                if i == 3
                else "exit 0\n" + "x" * 7200,
            },
        ]
    compact = compact_messages(messages, max_bytes=16000)
    assert wire_size(compact) <= 16000
    assert "Latest error detail" in compact[1]["content"]
    assert compact[-2:] == messages[-2:]
    assert all(
        len(tail) == 2 and tail[0]["tool_calls"][0]["id"] == tail[1]["tool_call_id"]
        for i in range(2, len(compact), 2)
        for tail in [compact[i : i + 2]]
    )
    assert len(messages) == 42


def test_proposer_evidence_is_bounded_without_breaking_structured_context():
    evidence = {f"history/{i}.md": "details\n" * 900 for i in range(50)}
    content = json.dumps(
        {
            "task": "Fix archive",
            "mission": "Improve plan",
            "current_guidance": "Follow checks",
            "evidence": evidence,
        }
    )
    compact = compact_messages(
        [
            {"role": "system", "content": "Propose"},
            {"role": "user", "content": content},
        ],
        max_bytes=16000,
        role="proposal",
    )
    payload = json.loads(compact[1]["content"])
    assert payload["task"] == "Fix archive"
    assert payload["current_guidance"] == "Follow checks"
    assert 0 < len(payload["evidence"]) < len(evidence)
    assert payload["evidence_omitted"]
    assert wire_size(compact) <= 16000


def test_budget_retries_reported_413_once_with_smaller_context():
    from github_harness_agent.executor import Limits, TaskSnapshot, _Budget
    from github_harness_agent.providers import Completion, ProviderExhausted

    messages = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Task " + "x" * 45000},
    ]
    task = TaskSnapshot("org/repo", "a" * 40, "Task", ())
    budget = _Budget(Limits(model_requests=3), lambda: False, task)
    lengths = []

    class Gateway:
        def complete(self, *, messages, **kwargs):
            lengths.append(wire_size(messages))
            if len(lengths) == 1:
                raise ProviderExhausted(None, {"test": "upstream HTTP 413"})
            return Completion("done", (), "test", "model", "coding", {})

    gateway = Gateway()
    assert budget.complete(gateway, "coding", messages).content == "done"
    assert budget.complete(gateway, "coding", messages).content == "done"
    assert lengths[0] <= 57344
    assert lengths[1] < lengths[0] and lengths[1] <= 24576
    assert lengths[2] == lengths[1]
    assert not messages[1]["content"].endswith("[excerpt shortened]")
