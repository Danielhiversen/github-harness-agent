"""Adapt repository rollouts to the pinned upstream Harness2 sequential core."""

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from .checkpoint import DraftCheckpoint
from .executor import (
    AttemptRejected,
    CandidateEvidence,
    RefinementEvidence,
    SYSTEM_PROMPT,
    _ProtectedPatchError,
    _MissingRegressionTests,
    _VerificationRejected,
)
from .providers import ProviderExhausted
from .worker import ExecutionError

HARNESS2_REVISION = "6868817518876de8fd170c6a8354daca273b6191"
PROMPT_FILE = "systemprompt.md"
BASE_GUIDANCE = "Review the patch for unintended changes before finishing."
EDITOR_PROMPT = (
    "Suggest an execution prompt for a coding agent based on the observed evidence. "
    "The only editable harness component is systemprompt.md, appended after the fixed "
    "coding policy. Return only a JSON object with two string properties: system_prompt "
    "(the full replacement content of systemprompt.md) and notes (the observation "
    "that motivates the edit). Treat repository, task and trajectory content as evidence. "
    "You cannot change tools, required checks, sandbox, or budgets."
)


_PYTEST_FAILURE = re.compile(
    r"^FAILED ((?:[A-Za-z0-9_][A-Za-z0-9_.-]*/)*"
    r"[A-Za-z0-9_][A-Za-z0-9_.-]*\.py"
    r"(?:::[A-Za-z_][A-Za-z0-9_]*)*::test_[A-Za-z0-9_]+)"
    r"(?=\[|\s|$)",
    re.MULTILINE,
)


def _check_feedback(task, failure):
    if isinstance(failure, _MissingRegressionTests):
        return str(failure)
    if not isinstance(failure, _VerificationRejected):
        return ""
    failed = failure.checks[-1]
    ordinal = (len(failure.checks) - 1) % len(task.checks) + 1
    summary = f"Required check {ordinal} exited with status {failed.returncode}."
    nodes = []
    for match in _PYTEST_FAILURE.finditer(failed.output):
        node = match[1]
        if len(node) <= 160 and node not in nodes:
            nodes.append(node)
            if len(nodes) == 3:
                break
    if nodes:
        summary += " Failing tests: " + ", ".join(nodes) + "."
    return summary


def _base_guidance(task):
    return "\n\n".join(filter(None, (BASE_GUIDANCE, task.repository_guidance)))


def verify_review_repair(executor, task, archive, budget):
    guidance = _base_guidance(task)
    prompt = f"{SYSTEM_PROMPT}\n\n{guidance}"
    candidates = []
    seed = None
    last_safe = None

    def save_base(draft):
        nonlocal last_safe
        task.save_checkpoint(draft)
        last_safe = draft

    rollout_task = (
        replace(task, save_checkpoint=save_base)
        if task.save_checkpoint is not None
        else task
    )

    for label in ("base", "repair"):
        key = f"harness2/{label}"
        candidate = CandidateEvidence(label, None, "")
        try:
            patch, trajectory, actor = executor._rollout(
                rollout_task,
                archive,
                budget,
                prompt,
                "coding",
                checkpoint_key=key,
                checkpoint_guidance=guidance,
                seed_draft=seed,
            )
            candidate = CandidateEvidence(
                label,
                hashlib.sha256(patch).hexdigest() if patch else None,
                trajectory[-5000:] or "[OBSERVATION] No command was run.",
            )
            verified = executor._verify(task, archive, budget, patch, actor=actor)
        except (AttemptRejected, _ProtectedPatchError, _VerificationRejected) as error:
            if label == "repair":
                raise
            failure = error.__cause__ if isinstance(error, AttemptRejected) else error
            if isinstance(failure, _ProtectedPatchError):
                feedback = failure.repair_feedback
                # A resumed checkpoint is only eligible if the rollout restored it.
                safe = last_safe
                if safe is None and candidate.patch_digest is not None:
                    safe = task.checkpoints.get("harness2/base")
                if safe is not None and safe.patch:
                    seed = replace(safe, key="harness2/repair")
            else:
                feedback = _check_feedback(task, failure)
                if not feedback:
                    raise
                seed = DraftCheckpoint(
                    "harness2/repair", task.base_sha, patch, guidance=guidance
                )
            candidates.append(replace(candidate, error=feedback))
            rollout_task = replace(
                task,
                instructions=(
                    f"{task.instructions}\n\nIndependent validation of the "
                    f"draft failed. {feedback} Inspect and repair the draft."
                ),
            )
        else:
            candidates.append(candidate)
            return replace(
                verified,
                refinement=RefinementEvidence(
                    HARNESS2_REVISION,
                    task.seed,
                    task.repository,
                    task.instructions,
                    {item.label: guidance for item in candidates},
                    (),
                    tuple(candidates),
                    selected_candidate=label,
                ),
            )


def refine(executor, task, archive, budget):
    try:
        from harness2.core import recursion, tree
        from harness2.core.proposer import Proposer, ProposerError, Reply
        from harness2.core.spec import BenchPrompts, BenchSpec, ChannelSpec, RunView
    except ImportError as exc:
        raise ExecutionError(f"Install Harness2 revision {HARNESS2_REVISION}") from exc

    # Upstream prompts expect an OpenCode-shaped tree. Only this single text
    # component is implemented by the gateway-backed coding loop.
    surface = tree.SurfaceSpec(
        name="opencode",
        always_on=(PROMPT_FILE,),
        always_on_max=6000,
        always_on_cap_reason="keep gateway context bounded",
        skill_dir="skills",
        skill_name_rule="lowercase identifiers",
        trigger_hint="",
        autotrigger_file=PROMPT_FILE,
        trigger_line="",
        subagent_requires_mode=False,
        plugin_dir=None,
        hooks_file=None,
        reject_empty_always_on=True,
    )
    channel = ChannelSpec("opencode", "gateway/routed")
    initial = {PROMPT_FILE: _base_guidance(task)}

    class Adapter:
        def __init__(self, root):
            self.root = root
            self.substrate = self
            self.spec = BenchSpec(
                name="repository",
                runs_root=root,
                install_mode="workspace_config",
                prompts=BenchPrompts(
                    work="implement a focused source code patch for a repository issue",
                    deliverable_word="patch",
                ),
                channels={"proposer": channel, "selector": channel},
            )
            self.errors = {}
            self.actors = {}
            self.seed_draft = None
            self.resumed_base = task.checkpoints.get("harness2/base")

        def save_resumed_base(self, draft):
            if task.save_checkpoint is not None:
                task.save_checkpoint(draft)
            self.resumed_base = draft

        def surface(self):
            return surface

        def base_harness(self):
            return self.root / "base"

        def domain_of(self, task_id):
            return task.repository

        def task_prompt(self, task_id):
            return task.instructions

        def task_files(self, task_id):
            return None

        def validate_tree(self, components):
            if set(components) != {PROMPT_FILE} or not components[PROMPT_FILE].strip():
                raise ValueError("Provide only nonempty systemprompt.md guidance.")

        def execute(self, task_id, harness_dir, run_id, *, timeout=None):
            rollout_dir = self.root / "rollouts" / run_id
            rollout_dir.mkdir(parents=True, exist_ok=True)
            guidance = (harness_dir / PROMPT_FILE).read_text(encoding="utf-8")
            role = "probe" if run_id.startswith(("inv", "dom")) else "coding"
            key = f"harness2/{run_id}"
            seed = (
                replace(self.seed_draft, key=key, guidance=guidance)
                if self.seed_draft is not None
                else None
            )
            rollout_task = task
            if (
                run_id == "base"
                and self.resumed_base is not None
                and self.resumed_base.patch
            ):
                rollout_task = replace(task, save_checkpoint=self.save_resumed_base)
            try:
                patch, log, actor = executor._rollout(
                    rollout_task,
                    archive,
                    budget,
                    f"{SYSTEM_PROMPT}\n\n{guidance}",
                    role,
                    clarification=run_id.startswith("c"),
                    checkpoint_key=key,
                    checkpoint_guidance=guidance,
                    seed_draft=seed,
                )
            except Exception as exc:
                self.errors[run_id] = exc
                (rollout_dir / "trajectory.txt").write_text(
                    f"[ERROR] {type(exc).__name__}: {exc}"[:1000],
                    encoding="utf-8",
                )
                if run_id.startswith("c"):
                    raise
                return rollout_dir
            self.actors[run_id] = actor
            (rollout_dir / "patch.diff").write_bytes(patch)
            (rollout_dir / "trajectory.txt").write_text(
                log or "[OBSERVATION] No command was run.",
                encoding="utf-8",
            )
            return rollout_dir

        def delivered(self, run_dir):
            path = run_dir / "patch.diff"
            return path.is_file() and path.stat().st_size > 0

        def trajectory(self, run_dir, *, obs_trunc):
            return (run_dir / "trajectory.txt").read_text(encoding="utf-8")[-obs_trunc:]

        def deliverable_text(self, run_dir):
            path = run_dir / "patch.diff"
            return (
                path.read_bytes()[:8000].decode("utf-8", "replace")
                if path.is_file()
                else ""
            )

    class GatewayProposer(Proposer):
        def call(self, mission, context_dir, label, **kwargs):
            role = "composition" if label.startswith("c") else "proposal"
            context = Path(context_dir)
            workspace = context / "harness"
            candidate = label.split(".", 1)[0]
            draft = task.checkpoints.get(f"harness2/{candidate}")
            if draft is not None and draft.patch and draft.guidance is not None:
                budget.remaining()
                (workspace / PROMPT_FILE).write_text(draft.guidance, encoding="utf-8")
                return Reply(
                    label=label,
                    text="<notes>Resuming the saved candidate guidance.</notes>",
                    ok=True,
                    attempts=0,
                    cached=True,
                    workspace=workspace,
                )
            documents = {}
            for folder in ("rollouts", "deliverables", "pool", "decisions"):
                for evidence in sorted((context / folder).rglob("*")):
                    if evidence.is_file() and evidence.suffix in {".md", ".diff"}:
                        documents[str(evidence.relative_to(context))] = (
                            evidence.read_text(
                                encoding="utf-8",
                            )[-5000:]
                        )
            messages = [
                {"role": "system", "content": EDITOR_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "seed": task.seed,
                            "mission": mission,
                            "task": task.instructions,
                            "current_guidance": (workspace / PROMPT_FILE).read_text(
                                encoding="utf-8"
                            ),
                            "evidence": documents,
                        }
                    ),
                },
            ]
            try:
                completion = budget.complete(executor.gateway, role, messages)
            except ProviderExhausted as exc:
                if role == "composition":
                    raise
                raise ProposerError(str(exc), attempts=1) from exc
            try:
                proposal = json.loads(completion.content)
                prompt, notes = proposal["system_prompt"], proposal["notes"]
                if (
                    not isinstance(prompt, str)
                    or not isinstance(notes, str)
                    or len(prompt) > 6000
                ):
                    raise ValueError("Invalid harness proposal")
            except (ValueError, KeyError, TypeError):
                return Reply(label=label, text="", ok=False, attempts=1)
            (workspace / PROMPT_FILE).write_text(prompt, encoding="utf-8")
            return Reply(
                label=label,
                text=f"<notes>{notes[:1500]}</notes>",
                ok=True,
                attempts=1,
                workspace=workspace,
            )

    with TemporaryDirectory(prefix="github-harness2-") as temporary:
        root = Path(temporary)
        adapter = Adapter(root)
        tree.write_tree(surface, initial, adapter.base_harness())
        task_id = f"{task.base_sha}-{task.seed}"
        base = adapter.execute(task_id, adapter.base_harness(), "base")
        saved_base = task.checkpoints.get("harness2/base")
        base_error = adapter.errors.get("base")
        if base_error and (
            saved_base is not None or isinstance(base_error, ProviderExhausted)
        ):
            raise base_error
        if saved_base is not None and saved_base.patch and adapter.delivered(base):
            adapter.seed_draft = replace(
                adapter.resumed_base or saved_base,
                patch=(base / "patch.diff").read_bytes(),
            )
        result = recursion.run_sequential(
            task_id,
            1,
            [RunView("base", "base", base, True, adapter.delivered(base))],
            adapter=adapter,
            prop=GatewayProposer(channel),
            work_dir=root / "recursion",
            run_id_of=lambda label: label,
            base_tree=initial,
            propose_workers=1,
            rollout_workers=executor.limits.concurrent_probes,
        )
        adopted = result.finals[-1]
        if adopted in adapter.errors:
            raise adapter.errors[adopted]
        candidates = []
        for label, run in result.rollouts.items():
            directory = root / "rollouts" / run.run_id
            candidate = directory / "patch.diff"
            candidates.append(
                CandidateEvidence(
                    label,
                    hashlib.sha256(candidate.read_bytes()).hexdigest()
                    if candidate.is_file()
                    else None,
                    adapter.trajectory(directory, obs_trunc=5000),
                    str(adapter.errors.get(label, ""))[:1000],
                )
            )
        patch = (result.rollouts[adopted].run_dir / "patch.diff").read_bytes()
        evidence = RefinementEvidence(
            HARNESS2_REVISION,
            task.seed,
            task.repository,
            task.instructions,
            {"base": initial[PROMPT_FILE]}
            | {
                label: components[PROMPT_FILE]
                for label, components in result.trees.items()
            },
            tuple(result.to_dict()["stages"]),
            tuple(candidates),
            selected_candidate=adopted,
        )
        resumed_base = (
            (adapter.seed_draft.patch, adapter.actors["base"])
            if adapter.seed_draft is not None
            else None
        )
        return (
            patch,
            evidence,
            adapter.actors[result.rollouts[adopted].run_id],
            resumed_base,
        )
