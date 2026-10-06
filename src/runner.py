from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from src.base.result import ModelResult
from src.registry import ProviderConfig, get_provider


@dataclass(frozen=True)
class RunSummary:
    candidate_path: Path
    change_log_path: Path
    model_count: int
    changed_model_count: int
    warning_count: int


def run(
    project_root: Path,
    provider_keys: Iterable[str] = ("openai",),
    run_date: date | None = None,
) -> RunSummary:
    run_date = run_date or date.today()
    source_path = project_root / "models.json"
    official = _load_models(source_path)
    candidate = copy.deepcopy(official)
    candidate_models = candidate["models"]
    log_entries: list[str] = []
    changed_model_count = 0
    warning_count = 0

    for provider_key in provider_keys:
        config = get_provider(provider_key)
        source_models = [
            model for model in official["models"] if model.get("provider") == config.name
        ]
        if not source_models:
            log_entries.append(f"[WARNING]\n{config.name}\nno models found in models.json")
            warning_count += 1
            continue

        try:
            results = config.provider().run(config.url, source_models)
        except Exception as exc:  # Preserve the full candidate on source-level failure.
            message = f"official pricing source failed: {type(exc).__name__}: {exc}"
            log_entries.append(
                f"[WARNING]\n{config.name}\n{message}; "
                f"retained all {len(source_models)} existing models"
            )
            warning_count += 1
            continue

        try:
            result_map = _index_results(results, config)
        except ValueError as exc:
            log_entries.append(
                f"[WARNING]\n{config.name}\ninvalid provider results: {exc}; "
                f"retained all {len(source_models)} existing models"
            )
            warning_count += 1
            continue
        for index, old_model in enumerate(official["models"]):
            if old_model.get("provider") != config.name:
                continue
            key = _model_key(old_model)
            result = result_map.get(key)
            if result is None:
                log_entries.append(
                    _warning_entry(*key, "provider returned no result; retained existing model")
                )
                warning_count += 1
                continue

            candidate_models[index] = result.data
            changes = _collect_changes(old_model, result.data)
            if changes:
                changed_model_count += 1
                lines = [f"[PRICE]", _model_label(*key)]
                lines.extend(f"{path}: {_format_value(old)} -> {_format_value(new)}" for path, old, new in changes)
                log_entries.append("\n".join(lines))
            for warning in result.warnings:
                log_entries.append(_warning_entry(*key, warning))
                warning_count += 1

    result_dir = project_root / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = result_dir / f"{run_date.isoformat()}_models.json"
    _write_json(candidate_path, candidate)

    change_log_path = project_root / "change.log"
    _write_change_log(change_log_path, run_date, log_entries, changed_model_count)

    return RunSummary(
        candidate_path=candidate_path,
        change_log_path=change_log_path,
        model_count=len(candidate_models),
        changed_model_count=changed_model_count,
        warning_count=warning_count,
    )


def _load_models(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
        raise ValueError("models.json must be an object containing a models array")
    if not all(isinstance(model, dict) for model in data["models"]):
        raise ValueError("every item in models.json models must be an object")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _write_change_log(
    path: Path, run_date: date, entries: list[str], changed_model_count: int
) -> None:
    body: list[str] = [run_date.isoformat(), ""]
    if changed_model_count == 0:
        body.extend(["No pricing changes.", ""])
    if entries:
        body.append("\n\n".join(entries))
        body.append("")
    path.write_text("\n".join(body), encoding="utf-8")


def _index_results(
    results: list[ModelResult], config: ProviderConfig
) -> dict[tuple[str, str, str], ModelResult]:
    indexed: dict[tuple[str, str, str], ModelResult] = {}
    for result in results:
        key = (result.provider, result.model_api_id, result.model_region)
        if result.provider != config.name:
            raise ValueError(
                f"{config.name} provider returned a result for {result.provider}"
            )
        if key in indexed:
            raise ValueError(f"provider returned duplicate result for {_model_label(*key)}")
        indexed[key] = result
    return indexed


def _model_key(model: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(model.get("provider", "<missing>")),
        str(model.get("model_api_id", "<missing>")),
        str(model.get("model_region", "<missing>")),
    )


def _model_label(provider: str, model_id: str, region: str) -> str:
    suffix = "" if region == "Global" else f" / {region}"
    return f"{provider} / {model_id}{suffix}"


def _warning_entry(provider: str, model_id: str, region: str, message: str) -> str:
    return f"[WARNING]\n{_model_label(provider, model_id, region)}\n{message}"


def _collect_changes(
    old: Any, new: Any, path: tuple[str, ...] = ()
) -> list[tuple[str, Any, Any]]:
    if isinstance(old, dict) and isinstance(new, dict):
        changes: list[tuple[str, Any, Any]] = []
        keys = list(old)
        keys.extend(key for key in new if key not in old)
        for key in keys:
            changes.extend(_collect_changes(old.get(key), new.get(key), (*path, key)))
        return changes
    if isinstance(old, list) and isinstance(new, list):
        changes = []
        for index, (old_item, new_item) in enumerate(zip(old, new)):
            segment = _list_segment(old_item, index, len(old))
            changes.extend(_collect_changes(old_item, new_item, (*path, segment)))
        if len(old) != len(new):
            changes.append((_format_path(path), old, new))
        return changes
    if old != new:
        return [(_format_path(path), old, new)]
    return []


def _list_segment(item: Any, index: int, length: int) -> str:
    if isinstance(item, dict) and "context_min" in item:
        minimum = item.get("context_min")
        maximum = item.get("context_max")
        thinking = item.get("thinking")
        suffix = (
            f", thinking={str(thinking).lower()}"
            if isinstance(thinking, bool)
            else ""
        )
        return f"[{minimum}-{maximum if maximum is not None else '∞'}{suffix}]"
    return "" if length == 1 else f"[{index}]"


def _format_path(path: tuple[str, ...]) -> str:
    # per_million is the stored unit rather than a useful administrator-facing
    # path component.
    useful = path[:-1] if path and path[-1] == "per_million" else path
    rendered = ""
    for segment in useful:
        if not segment:
            continue
        if segment.startswith("["):
            rendered += segment
        else:
            rendered += ("." if rendered else "") + segment
    return rendered


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return format(value, ".15g")
    return str(value)
