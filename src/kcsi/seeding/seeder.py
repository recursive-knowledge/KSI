"""Population seeding from distilled knowledge bundles."""

from __future__ import annotations

import logging
from typing import Any

from ..models import AgentState
from ..runtime.seeding import is_canonical_distillation_bundle

log = logging.getLogger(__name__)


class PopulationSeeder:
    """Create a new agent population seeded with distilled bundles."""

    def seed(
        self,
        *,
        num_agents: int,
        generation: int,
        task_labels: list[str] | None = None,
        cross_task_bundle: dict[str, Any] | None = None,
        knowledge_store: Any | None = None,
        experiment: str | None = None,
        skip_per_task_labels: set[str] | frozenset[str] | None = None,
        cross_task_target_conditioning: bool = False,
    ) -> list[AgentState]:
        # ``generation`` is the generation that *produced* the bundles
        # (the phase-4 callsite in the engine passes the source generation);
        # the new agents belong to the following generation.
        next_gen = generation + 1

        labels = list(task_labels or [])
        # Hold-out probe: these labels never receive a per-task bundle, even
        # if a stale one exists in the DB (e.g. a resumed/repurposed
        # experiment where the task used to be a training task). They still
        # get the cross-task bundle like everyone else.
        skip_per_task = set(skip_per_task_labels or ())

        # Task-mode path: deterministic per-task assignment. Every label gets
        # the single broadcast ``cross_task_bundle``.
        if labels:
            per_task_labels = [
                labels[i] if i < len(labels) and str(labels[i]).strip() else f"task-{i}" for i in range(num_agents)
            ]
            # Batch-load every per-task bundle in ONE query instead of a
            # locked load_distillation() per agent. Hold-out
            # labels are excluded from the lookup so they never receive one.
            bundles_by_label = _load_per_task_bundles(
                knowledge_store=knowledge_store,
                generation=generation,
                task_ids=[lab for lab in per_task_labels if lab not in skip_per_task],
                experiment=experiment,
            )
            # Target-conditioning: each label gets ITS OWN cross-task bundle
            # (keyed by task id, scope='cross_task'), replacing the single
            # broadcast bundle. Hold-out labels still get cross-task guidance;
            # only same-task per-task bundles are excluded for hold-outs.
            cross_task_by_label: dict[str, dict[str, Any]] = {}
            if cross_task_target_conditioning:
                cross_task_by_label = _load_cross_task_bundles(
                    knowledge_store=knowledge_store,
                    generation=generation,
                    task_ids=per_task_labels,
                    experiment=experiment,
                )
            agents: list[AgentState] = []
            for i in range(num_agents):
                label = per_task_labels[i]
                agent_cross_task = (
                    cross_task_by_label.get(label) if cross_task_target_conditioning else cross_task_bundle
                )
                pkg = _build_task_seed_package(
                    label=label,
                    next_gen=next_gen,
                    cross_task_bundle=agent_cross_task,
                    per_task_bundle=None if label in skip_per_task else bundles_by_label.get(label),
                )
                agents.append(
                    AgentState(
                        id=f"agent-{i}",
                        generation=next_gen,
                        workstream=label,
                        workstream_description=label,
                        seed_package=pkg,
                    )
                )
            return agents

        agents: list[AgentState] = []
        for i in range(num_agents):
            pkg: dict[str, Any] = {}
            if cross_task_bundle is not None:
                pkg["generation"] = next_gen
                pkg["cross_task_bundle"] = cross_task_bundle
            agents.append(AgentState(id=f"agent-{i}", generation=next_gen, seed_package=pkg))
        return agents


def _build_task_seed_package(
    *,
    label: str,
    next_gen: int,
    cross_task_bundle: dict[str, Any] | None,
    per_task_bundle: dict[str, Any] | None,
) -> dict[str, Any]:
    pkg: dict[str, Any] = {
        "generation": next_gen,
        "workstream_name": label,
        "workstream_description": label,
        "insight_bundle": [],
        "evidence_refs": [],
    }
    if per_task_bundle is not None:
        pkg["per_task_bundle"] = per_task_bundle
    if cross_task_bundle is not None:
        pkg["cross_task_bundle"] = cross_task_bundle
    return pkg


def _load_per_task_bundles(
    *,
    knowledge_store: Any | None,
    generation: int,
    task_ids: list[str],
    experiment: str | None,
) -> dict[str, dict[str, Any]]:
    """Batch-load canonical per-task bundles keyed by task id.

    Non-canonical bundles are dropped (absent from the map = no bundle),
    matching the per-task filter this replaced.
    """
    if knowledge_store is None or not task_ids:
        return {}
    try:
        raw = knowledge_store.load_distillations_batch(
            generation=generation,
            task_ids=task_ids,
            scope="per_task",
            experiment=experiment,
        )
    except (ValueError, TypeError):
        # Programming error — should not silently degrade.
        raise
    except Exception as exc:  # pragma: no cover — defensive
        log.warning(
            "[SEEDER] Failed to batch-load per-task bundles (gen=%s): %s",
            generation,
            exc,
        )
        return {}
    return {tid: bundle for tid, bundle in raw.items() if is_canonical_distillation_bundle(bundle)}


def _load_cross_task_bundles(
    *,
    knowledge_store: Any | None,
    generation: int,
    task_ids: list[str],
    experiment: str | None,
) -> dict[str, dict[str, Any]]:
    """Batch-load canonical per-task cross-task bundles keyed by task id
    (scope='cross_task'). Mirrors :func:`_load_per_task_bundles`; used only when
    target-conditioning is on. Absent id = no bundle."""
    if knowledge_store is None or not task_ids:
        return {}
    try:
        raw = knowledge_store.load_distillations_batch(
            generation=generation,
            task_ids=task_ids,
            scope="cross_task",
            experiment=experiment,
        )
    except (ValueError, TypeError):
        raise
    except Exception as exc:  # pragma: no cover — defensive
        log.warning(
            "[SEEDER] Failed to batch-load cross-task bundles (gen=%s): %s",
            generation,
            exc,
        )
        return {}
    return {tid: bundle for tid, bundle in raw.items() if is_canonical_distillation_bundle(bundle)}


def _build_shared_seed_package(
    shared_insight_bundle: list[dict[str, Any]],
    *,
    generation: int,
    workstream_description: str = "Shared guidance bundle for task-mode execution",
) -> dict[str, Any]:
    """Build a shared seed package from an external insight bundle.

    Used only by the ``--seed-bundle-path`` gen-1 injection path (see
    :meth:`GenerationalOrchestrator._inject_seed_bundle`).  This is *not*
    the canonical path for per-task / cross-task bundles produced by
    the distillation phase service - those flow through
    ``seed(..., cross_task_bundle=..., knowledge_store=...)``.
    """
    bundle: list[dict[str, Any]] = []
    for item in shared_insight_bundle:
        normalized = _normalize_bundle_item(item)
        if normalized is not None:
            bundle.append(normalized)
    evidence_refs = _build_evidence_refs(bundle)
    return {
        "generation": generation,
        "workstream_name": "task-shared-bundle",
        "workstream_description": workstream_description,
        "insight_bundle": bundle,
        "shared_insight_bundle": bundle,
        "evidence_refs": evidence_refs,
    }


def _normalize_bundle_item(item: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    text = str(item.get("text") or "").strip()
    if not text:
        return None
    normalized = {
        "id": str(item.get("id") or item.get("insight_id") or "").strip(),
        "text": text,
        "confidence": str(item.get("confidence") or "medium"),
        "source_task_id": item.get("source_task_id"),
        "author_agent_id": str(item.get("author_agent_id") or "").strip(),
        "generation": int(item.get("generation") or 0),
        "workstream": str(item.get("workstream") or "").strip(),
    }
    source_insight_ids = item.get("source_insight_ids")
    if isinstance(source_insight_ids, list):
        normalized["source_insight_ids"] = [str(value).strip() for value in source_insight_ids if str(value).strip()]
    return normalized


def _build_evidence_refs(bundle: list[dict[str, Any]]) -> list[str]:
    evidence_refs: list[str] = []
    for item in bundle:
        source_task_id = item.get("source_task_id")
        if isinstance(source_task_id, str) and source_task_id.strip():
            evidence_refs.append(f"task:{source_task_id.strip()}")
    return evidence_refs
