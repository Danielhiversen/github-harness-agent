from github_harness_agent.review_parameter_use import lost_iteration_argument


def test_source_windows_include_a_seed_and_direct_child_record_declaration():
    prior = "def visit(*, limit):\n    for depth in range(limit):\n        pass\n"
    current = (
        "class Frame:\n"
        '    """One outstanding state."""\n'
        "    depth: int\n"
        "\ndef visit(*, limit):\n"
        "    heap = []\n"
        "    heappush(heap, Frame(0))\n"
        "    while heap:\n"
        "        frame = heappop(heap)\n"
        "        if frame.depth < 3:\n"
        "            heappush(heap, Frame(frame.depth + 1))\n"
    )
    result = lost_iteration_argument(
        "visit(limit=1)",
        {"sample.py": prior},
        {"sample.py": current},
        source_windows=True,
    )
    assert "7 |     heappush(heap, Frame(0))" in result
    assert "9 |         frame = heappop(heap)" in result
    assert "11 |             heappush(heap, Frame(frame.depth + 1))" in result
    assert "1 | class Frame:" in result and "3 |     depth: int" in result
    assert "2 |     for depth in range(limit):" in result
    ambiguous = lost_iteration_argument(
        "visit(limit=1)",
        {"a.py": prior, "b.py": prior},
        {"a.py": current, "b.py": current},
        source_windows=True,
    )
    assert "keyword limit" in ambiguous and "Current traversal" not in ambiguous


PINNED = """def scan(items, rounds):
    if rounds < 1:
        raise ValueError()
    for step in range(rounds):
        consume(items, step)
"""
DRAFT = PINNED.replace("for step in range(rounds):", "for step in items:")


def test_explains_only_a_tested_limit_that_lost_its_loop_use():
    fixture = "def test_limit():\n    client.scan(data, rounds=1)\n"
    result = lost_iteration_argument(fixture, {"m.py": PINNED}, {"m.py": DRAFT})
    assert "scan" in result and "rounds" in result and "input validation" in result
    assert not lost_iteration_argument(fixture, {"m.py": PINNED}, {"m.py": PINNED})
    assert not lost_iteration_argument(
        fixture.replace("rounds=", "other="), {"m.py": PINNED}, {"m.py": DRAFT}
    )
    assert not lost_iteration_argument(
        fixture, {"m.py": PINNED}, {"m.py": DRAFT + "    run(items, rounds)\n"}
    )
