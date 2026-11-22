from github_harness_agent.review_parameter_use import lost_iteration_argument


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
