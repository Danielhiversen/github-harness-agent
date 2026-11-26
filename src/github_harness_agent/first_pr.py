"""Select bug regression proof and describe only evidence actually collected."""

import re


def reproducible_bug(title: str, body: str) -> bool:
    if re.search(r"\b(?:docs?|documentation|readme|typos?)\b", title, re.I) or re.match(
        r"\s*document\b", title, re.I
    ):
        return False
    sections = {}
    for match in re.finditer(
        r"(?im)^\s*#{0,6}\s*(expected(?: behavior)?|actual(?: behavior)?|steps to reproduce)\s*:?\s*\n([^#]+?)(?=\n\s*#|\Z)",
        body,
    ):
        sections[match.group(1).lower().split()[0]] = match.group(2).strip()
    if all(sections.get(name) for name in ("expected", "actual", "steps")):
        return sections["expected"] != sections["actual"]
    text = f"{title}\n{body}"
    # Template headings alone are not evidence; require a concrete symptom.
    symptom = re.search(
        r"\b(?:crash(?:es|ed|ing)?|fails?|failed|broken|incorrect|wrong|unexpected|regression|"
        r"raises?\s+\w*(?:Error|Exception)|returns?\s+(?:None|null|empty))\b",
        text,
        re.I,
    )
    bug = re.search(r"\b(?:bug|fix|crash(?:es)?|regression|broken)\b", title, re.I)
    reproduction = re.search(
        r"\b(?:when|after|calling|clicking|running|reproduce|expected|actual|instead|assert)\b|```|Traceback",
        body,
        re.I,
    )
    return bool(symptom and (bug or reproduction))


def pull_summary(title: str, patch: bytes, checks) -> str:
    """Keep legacy prepared jobs useful without inventing red/green evidence."""
    checks = [c for c in checks if isinstance(c, dict)]
    paths = []
    for match in re.finditer(rb"^\+\+\+ b/(.+)$", patch, re.M):
        path = match.group(1).decode("utf-8", "replace")
        if path not in paths:
            paths.append(path)
    lines = ["## Change", "", title.strip()]
    if paths:
        lines.extend(["", "Changed files:"])
    lines.extend(f"- Updated `{path}`." for path in paths[:20])
    proofs = [
        c
        for c in checks
        if c.get("command", [None])[0] == "verify-regression"
        and c.get("returncode") == 0
    ]
    if proofs:
        proof = proofs[-1]
        lines.extend(
            [
                "",
                "Regression proof:",
                f"Baseline: `{proof['command'][1]}`.",
                proof.get("output", "")[:600],
            ]
        )
    return "\n".join(lines)
