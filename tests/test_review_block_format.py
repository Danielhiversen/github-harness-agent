import pytest

from github_harness_agent.review_solver import apply_blocks


def test_extracts_complete_model_edit_surrounded_by_explanation():
    reply = """Here is the focused edit.

File: `example.py`
```python
<<<<<<< SEARCH
value = 1
=======
value = 2
>>>>>>> REPLACE
```

This changes the return value."""
    assert apply_blocks(reply, {"example.py": "value = 1\n"}, "test_new.py") == {
        "example.py": "value = 2\n"
    }
    with pytest.raises(ValueError):
        apply_blocks(
            reply + "\nexample.py\n<<<<<<< SEARCH\noops",
            {"example.py": "value = 1\n"},
            "test_new.py",
        )
