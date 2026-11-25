from github_harness_agent.review_recovery import ReviewRecovery
from github_harness_agent.review_solver import _draft_review


def test_anchor_tracks_edits_from_earlier_review_goals():
    source = {"m.py": "first\nkeep\nnext\nlast\n"}
    draft = {"m.py": "extra\nlines\nfirst\nkeep\nnext\nlast\n"}
    review = ReviewRecovery((("m.py", 3, "third line"),), "")
    assert _draft_review(review, source, draft).feedback == (("m.py", 5, "third line"),)
    assert review.feedback[0][1] == 3
