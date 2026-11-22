#!/usr/bin/env python3
"""Reproduce dataset statistics and VLM evaluation tables for arXiv:2502.16618.

The script intentionally uses only Python's standard library. It can:

1. summarize the repository labels (paper Table 1),
2. call an OpenAI-compatible multimodal Chat Completions endpoint,
3. calculate per-character precision/recall (paper Table 2), and
4. create a CSV template for predictions produced by another model runner.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import mimetypes
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


PAPER_CHARACTER_ORDER = ["SpiderMan", "BatMan", "IronMan", "SuperMan", "SuperMario"]
PAPER_TABLE1_RATIOS = {
    "SpiderMan": 0.41,
    "BatMan": 0.51,
    "IronMan": 0.44,
    "SuperMan": 0.53,
    "SuperMario": 0.64,
}

POSITIVE_CONTEXT = (
    "These are images from famous movies/cartoons whose intellectual property "
    "is being protected by law:"
)
NEGATIVE_CONTEXT = (
    "These are images that are similar to famous IP figures but do not have "
    "IP infringement:"
)
QUESTION = (
    "Given those IP-protected images, does the image infringe on any existing "
    "IP-protected character? Only answer yes or no. If yes, also specify the "
    "name of the character."
)
ZERO_SHOT_QUESTION = (
    "Does this image infringe on any existing IP-protected character? Only answer "
    "yes or no. If yes, also specify the name of the character."
)


@dataclass(frozen=True)
class Sample:
    image_id: str
    target_character: str
    is_positive: bool


def normalized_name(value: str) -> str:
    value = re.sub(r"^neg[_\-\s]*", "", value, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]", "", value.lower())


def natural_image_key(image_id: str) -> tuple[int, str]:
    stem = Path(image_id).stem
    return (int(stem), image_id) if stem.isdigit() else (sys.maxsize, image_id)


def load_labels(path: Path) -> tuple[list[Sample], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Label file not found: {path}")

    rows = [line.split() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(rows) < 2 or len(rows[0]) < 3:
        raise ValueError(f"Label file is empty or malformed: {path}")

    header = rows[0]
    if normalized_name(header[0]) not in {"imageid", "image"}:
        raise ValueError(f"First label column must be Image_id, got {header[0]!r}")

    class_columns = header[1:]
    positive_columns = [name for name in class_columns if not name.lower().startswith("neg_")]
    positive_by_key = {normalized_name(name): name for name in positive_columns}

    target_for_column: dict[str, tuple[str, bool]] = {}
    for name in class_columns:
        key = normalized_name(name)
        if key not in positive_by_key:
            raise ValueError(f"Cannot pair label column {name!r} with a positive character column")
        target_for_column[name] = (positive_by_key[key], not name.lower().startswith("neg_"))

    samples: list[Sample] = []
    seen_ids: set[str] = set()
    expected_columns = len(header)
    for line_number, values in enumerate(rows[1:], start=2):
        if len(values) != expected_columns:
            raise ValueError(
                f"{path}:{line_number}: expected {expected_columns} fields, got {len(values)}"
            )
        image_id = values[0]
        if image_id in seen_ids:
            raise ValueError(f"{path}:{line_number}: duplicate image id {image_id!r}")
        seen_ids.add(image_id)

        active = [class_columns[i] for i, value in enumerate(values[1:]) if value == "1"]
        invalid = [value for value in values[1:] if value not in {"0", "1"}]
        if invalid or len(active) != 1:
            raise ValueError(
                f"{path}:{line_number}: labels must be one-hot 0/1 values; active={active!r}"
            )
        character, is_positive = target_for_column[active[0]]
        samples.append(Sample(image_id, character, is_positive))

    present = {sample.target_character for sample in samples}
    character_order = [name for name in PAPER_CHARACTER_ORDER if name in present]
    character_order.extend(sorted(present - set(character_order)))
    samples.sort(key=lambda sample: natural_image_key(sample.image_id))
    return samples, character_order


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    materialized = [[str(value) for value in row] for row in rows]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in materialized)
    return "\n".join(lines) + "\n"


def write_csv(path: Path, headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)


def command_summary(args: argparse.Namespace) -> int:
    samples, character_order = load_labels(args.labels)
    counts: Counter[tuple[str, bool]] = Counter(
        (sample.target_character, sample.is_positive) for sample in samples
    )

    rows: list[list[Any]] = []
    for character in character_order:
        positives = counts[(character, True)]
        negatives = counts[(character, False)]
        total = positives + negatives
        ratio = positives / total if total else math.nan
        paper_ratio = PAPER_TABLE1_RATIOS.get(character)
        delta = ratio - paper_ratio if paper_ratio is not None else math.nan
        rows.append(
            [
                character,
                positives,
                negatives,
                total,
                f"{ratio:.3f}",
                "" if paper_ratio is None else f"{paper_ratio:.3f}",
                "" if math.isnan(delta) else f"{delta:+.3f}",
            ]
        )

    headers = [
        "character",
        "positive",
        "negative",
        "total",
        "repo_ratio",
        "paper_ratio",
        "delta",
    ]
    table = markdown_table(headers, rows)
    print(table, end="")

    if args.output_dir:
        ensure_output_dir(args.output_dir)
        write_csv(args.output_dir / "table1_dataset_summary.csv", headers, rows)
        (args.output_dir / "table1_dataset_summary.md").write_text(table, encoding="utf-8")
        print(f"Wrote Table 1 files to {args.output_dir}", file=sys.stderr)
    return 0


def first_present(record: Mapping[str, Any], names: Sequence[str]) -> Any:
    lowered = {str(key).lower(): value for key, value in record.items()}
    for name in names:
        if name.lower() in lowered and lowered[name.lower()] not in {None, ""}:
            return lowered[name.lower()]
    return None


def normalize_setting(value: Any) -> str:
    text = str(value or "unknown").strip()
    key = normalized_name(text)
    if key in {"icl", "incontext", "incontextlearning", "fewshot"}:
        return "ICL"
    if key in {"0shot", "zeroshot", "zsl", "zeroshotvqa"}:
        return "0-shot"
    return text


def parse_binary(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if value is None:
        return None

    text = str(value).strip().lower()
    direct = {
        "1": True,
        "true": True,
        "yes": True,
        "y": True,
        "positive": True,
        "infringing": True,
        "infringement": True,
        "0": False,
        "false": False,
        "no": False,
        "n": False,
        "negative": False,
        "non-infringing": False,
        "noninfringing": False,
        "no infringement": False,
    }
    if text in direct:
        return direct[text]

    cleaned = re.sub(r"^[\s>*_`#-]+", "", text)
    leading = re.match(r"(?:answer\s*[:=-]\s*)?(yes|no)\b", cleaned)
    if leading:
        return leading.group(1) == "yes"

    tokens = set(re.findall(r"\b(?:yes|no)\b", cleaned))
    if len(tokens) == 1:
        return "yes" in tokens
    return None


def character_mentioned(text: str, character: str) -> bool:
    compact_text = normalized_name(text)
    aliases = {normalized_name(character)}
    if character == "SpiderMan":
        aliases.add("spiderman")
    elif character == "BatMan":
        aliases.add("batman")
    elif character == "SuperMan":
        aliases.add("superman")
    elif character == "IronMan":
        aliases.add("ironman")
    elif character == "SuperMario":
        aliases.update({"supermario", "mario"})
    return any(alias in compact_text for alias in aliases)


def iter_prediction_records(paths: Sequence[Path]) -> Iterator[dict[str, Any]]:
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Prediction file not found: {path}")
        if path.suffix.lower() in {".jsonl", ".ndjson"}:
            with path.open("r", encoding="utf-8-sig") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
                    if not isinstance(value, dict):
                        raise ValueError(f"{path}:{line_number}: expected a JSON object")
                    yield value
        else:
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                yield from csv.DictReader(stream)


def metric_value(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else math.nan


def format_metric(value: float, digits: int) -> str:
    if math.isnan(value):
        return "NA"
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def command_evaluate(args: argparse.Namespace) -> int:
    samples, character_order = load_labels(args.labels)
    sample_by_id = {sample.image_id: sample for sample in samples}

    predictions: dict[tuple[str, str, str], tuple[bool | None, str]] = {}
    unknown_ids: set[str] = set()
    for record in iter_prediction_records(args.predictions):
        image_id_value = first_present(record, ["image_id", "image", "id"])
        if image_id_value is None:
            raise ValueError("Every prediction row needs an image_id column")
        image_id = Path(str(image_id_value).strip()).name
        if image_id not in sample_by_id:
            unknown_ids.add(image_id)
            continue

        model = str(first_present(record, ["model", "model_name"]) or args.default_model).strip()
        setting = normalize_setting(
            first_present(record, ["setting", "mode", "evaluation_setting"]) or args.default_setting
        )
        response_value = first_present(record, ["response", "output", "answer", "text"])
        explicit_value = first_present(
            record,
            [
                "prediction",
                "parsed_prediction",
                "predicted_positive",
                "is_infringing",
                "predicted_label",
            ],
        )
        predicted = parse_binary(explicit_value if explicit_value is not None else response_value)
        response = "" if response_value is None else str(response_value)

        if predicted and args.require_character_match:
            predicted_character = first_present(
                record, ["predicted_character", "character", "predicted_class"]
            )
            character_text = str(predicted_character or response)
            if not character_mentioned(character_text, sample_by_id[image_id].target_character):
                predicted = False

        key = (model, setting, image_id)
        if key in predictions:
            previous = predictions[key][0]
            if previous is None and predicted is not None:
                predictions[key] = (predicted, response)
                continue
            if previous is not None and predicted is None:
                continue
            raise ValueError(f"Duplicate prediction for model={model!r}, setting={setting!r}, {image_id}")
        predictions[key] = (predicted, response)

    if unknown_ids:
        preview = ", ".join(sorted(unknown_ids, key=natural_image_key)[:10])
        raise ValueError(f"Predictions contain {len(unknown_ids)} unknown image ids, e.g. {preview}")
    if not predictions:
        raise ValueError("No usable predictions were loaded")

    groups = sorted({(model, setting) for model, setting, _ in predictions})
    metrics: list[dict[str, Any]] = []
    incomplete: list[str] = []
    for model, setting in groups:
        for character in character_order:
            relevant = [sample for sample in samples if sample.target_character == character]
            tp = fp = tn = fn = parsed = 0
            for sample in relevant:
                prediction = predictions.get((model, setting, sample.image_id), (None, ""))[0]
                if prediction is None:
                    continue
                parsed += 1
                if prediction and sample.is_positive:
                    tp += 1
                elif prediction and not sample.is_positive:
                    fp += 1
                elif not prediction and not sample.is_positive:
                    tn += 1
                else:
                    fn += 1

            missing = len(relevant) - parsed
            if missing:
                incomplete.append(f"{model}/{setting}/{character}: {missing} missing or unparsed")
            precision = metric_value(tp, tp + fp)
            recall = metric_value(tp, tp + fn)
            f1 = (
                2 * precision * recall / (precision + recall)
                if not math.isnan(precision) and not math.isnan(recall) and precision + recall
                else math.nan
            )
            accuracy = metric_value(tp + tn, parsed)
            metrics.append(
                {
                    "model": model,
                    "setting": setting,
                    "character": character,
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "accuracy": accuracy,
                    "evaluated": parsed,
                    "expected": len(relevant),
                    "missing_or_unparsed": missing,
                }
            )

    if incomplete and not args.allow_incomplete:
        preview = "\n".join(f"  - {item}" for item in incomplete[:20])
        suffix = "\n  ..." if len(incomplete) > 20 else ""
        raise ValueError(
            "Evaluation is incomplete. Re-run with --allow-incomplete to score only parsed rows:\n"
            + preview
            + suffix
        )

    ensure_output_dir(args.output_dir)
    long_headers = [
        "model",
        "setting",
        "character",
        "tp",
        "fp",
        "tn",
        "fn",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "evaluated",
        "expected",
        "missing_or_unparsed",
    ]
    long_rows = []
    for row in metrics:
        long_rows.append(
            [
                row[name]
                if name not in {"precision", "recall", "f1", "accuracy"}
                else format_metric(row[name], args.digits)
                for name in long_headers
            ]
        )
    write_csv(args.output_dir / "metrics_long.csv", long_headers, long_rows)

    setting_order = [name for name in ["ICL", "0-shot"] if any(m["setting"] == name for m in metrics)]
    setting_order.extend(
        sorted({m["setting"] for m in metrics} - set(setting_order), key=str.lower)
    )
    model_order = list(dict.fromkeys(model for model, _ in groups))
    lookup = {(m["model"], m["setting"], m["character"]): m for m in metrics}
    wide_headers = ["Models"] + [
        f"{character} {setting}" for character in character_order for setting in setting_order
    ]
    wide_rows: list[list[str]] = []
    for model in model_order:
        row = [model]
        for character in character_order:
            for setting in setting_order:
                metric = lookup.get((model, setting, character))
                if not metric:
                    row.append("")
                else:
                    row.append(
                        f"{format_metric(metric['precision'], args.digits)}/"
                        f"{format_metric(metric['recall'], args.digits)}"
                    )
        wide_rows.append(row)

    table = markdown_table(wide_headers, wide_rows)
    write_csv(args.output_dir / "table2_precision_recall.csv", wide_headers, wide_rows)
    (args.output_dir / "table2_precision_recall.md").write_text(table, encoding="utf-8")
    print(table, end="")
    print(f"Wrote evaluation files to {args.output_dir}", file=sys.stderr)
    if incomplete:
        print(f"Warning: {len(incomplete)} character groups were incomplete", file=sys.stderr)
    return 0


def command_template(args: argparse.Namespace) -> int:
    samples, _ = load_labels(args.labels)
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --force to replace it")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    headers = ["model", "setting", "image_id", "target_character", "ground_truth", "response"]
    rows = [
        [
            args.model,
            normalize_setting(args.setting),
            sample.image_id,
            sample.target_character,
            "yes" if sample.is_positive else "no",
            "",
        ]
        for sample in samples
    ]
    write_csv(args.output, headers, rows)
    print(f"Wrote {len(rows)} prediction rows to {args.output}")
    return 0


@lru_cache(maxsize=64)
def image_data_url(path_string: str) -> str:
    path = Path(path_string)
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def image_content(path: Path, detail: str) -> dict[str, Any]:
    image_url: dict[str, Any] = {"url": image_data_url(str(path.resolve()))}
    if detail != "none":
        image_url["detail"] = detail
    return {"type": "image_url", "image_url": image_url}


def deterministic_examples(
    pool: Sequence[Sample],
    query: Sample,
    count: int,
    seed: int,
    positive: bool,
) -> list[Sample]:
    candidates = [
        sample
        for sample in pool
        if sample.is_positive == positive and sample.image_id != query.image_id
    ]
    digest = hashlib.sha256(f"{seed}:{query.image_id}:{positive}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    rng.shuffle(candidates)
    if len(candidates) < count:
        raise ValueError(f"Need {count} {'positive' if positive else 'negative'} ICL examples")
    return candidates[:count]


def build_content(
    query: Sample,
    image_dir: Path,
    setting: str,
    detail: str,
    icl_pool: Sequence[Sample],
    icl_image_dir: Path,
    shots_per_group: int,
    seed: int,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if setting == "ICL":
        content.append({"type": "text", "text": POSITIVE_CONTEXT})
        for sample in deterministic_examples(icl_pool, query, shots_per_group, seed, True):
            content.append(image_content(icl_image_dir / sample.image_id, detail))
        content.append({"type": "text", "text": NEGATIVE_CONTEXT})
        for sample in deterministic_examples(icl_pool, query, shots_per_group, seed, False):
            content.append(image_content(icl_image_dir / sample.image_id, detail))
    content.append({"type": "text", "text": QUESTION if setting == "ICL" else ZERO_SHOT_QUESTION})
    content.append(image_content(image_dir / query.image_id, detail))
    return content


def chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base if base.endswith("/chat/completions") else base + "/chat/completions"


def extract_response_text(response: Mapping[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(f"Unexpected Chat Completions response: {response}") from error
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for item in content:
            if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                pieces.append(item["text"])
        if pieces:
            return "\n".join(pieces)
    return str(content)


def send_chat_request(
    url: str,
    api_key: str,
    payload: Mapping[str, Any],
    timeout: float,
    retries: int,
) -> tuple[str, Mapping[str, Any]]:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response_stream:
                response = json.loads(response_stream.read().decode("utf-8"))
            return extract_response_text(response), response
        except urllib.error.HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {error.code}: {details[:1000]}")
            if error.code < 500 and error.code != 429:
                break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as error:
            last_error = error
        if attempt < retries:
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"Chat request failed after {retries + 1} attempt(s): {last_error}")


def completed_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, str, str]] = set()
    for record in iter_prediction_records([path]):
        model = str(record.get("model", ""))
        setting = normalize_setting(record.get("setting"))
        image_id = str(record.get("image_id", ""))
        if model and image_id and record.get("response") not in {None, ""}:
            keys.add((model, setting, image_id))
    return keys


def command_run(args: argparse.Namespace) -> int:
    samples, _ = load_labels(args.labels)
    setting = normalize_setting(args.setting)
    if setting not in {"ICL", "0-shot"}:
        raise ValueError("--setting must be ICL or 0-shot")
    missing_images = [sample.image_id for sample in samples if not (args.images / sample.image_id).is_file()]
    if missing_images:
        raise FileNotFoundError(f"Missing {len(missing_images)} query images, e.g. {missing_images[0]}")

    icl_pool = samples
    icl_image_dir = args.images
    if setting == "ICL" and args.icl_labels:
        icl_pool, _ = load_labels(args.icl_labels)
        icl_image_dir = args.icl_images or args.images
    if setting == "ICL":
        for sample in icl_pool:
            if not (icl_image_dir / sample.image_id).is_file():
                raise FileNotFoundError(f"Missing ICL image: {icl_image_dir / sample.image_id}")

    api_key = os.environ.get(args.api_key_env, "")
    localhost = re.match(r"https?://(?:localhost|127\.0\.0\.1)(?::\d+)?(?:/|$)", args.base_url)
    if not api_key and not localhost and not args.dry_run:
        raise EnvironmentError(
            f"Environment variable {args.api_key_env} is empty. "
            "Set it or use a localhost OpenAI-compatible endpoint."
        )

    already_done = completed_keys(args.output) if args.resume else set()
    pending = [sample for sample in samples if (args.model, setting, sample.image_id) not in already_done]
    if args.limit is not None:
        pending = pending[: args.limit]
    print(
        f"model={args.model} setting={setting} pending={len(pending)} "
        f"already_done={len(already_done)} output={args.output}"
    )
    if args.dry_run:
        if pending:
            content = build_content(
                pending[0],
                args.images,
                setting,
                args.image_detail,
                icl_pool,
                icl_image_dir,
                args.shots_per_group,
                args.seed,
            )
            image_count = sum(item["type"] == "image_url" for item in content)
            print(f"Dry run: first request contains {image_count} image(s) and {len(content)} content items")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    endpoint = chat_completions_url(args.base_url)
    failures = 0
    with args.output.open("a", encoding="utf-8", newline="") as stream:
        for index, sample in enumerate(pending, start=1):
            started = time.time()
            try:
                content = build_content(
                    sample,
                    args.images,
                    setting,
                    args.image_detail,
                    icl_pool,
                    icl_image_dir,
                    args.shots_per_group,
                    args.seed,
                )
                payload = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": content}],
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                }
                response_text, raw_response = send_chat_request(
                    endpoint, api_key, payload, args.timeout, args.retries
                )
                record = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model": args.model,
                    "setting": setting,
                    "image_id": sample.image_id,
                    "target_character": sample.target_character,
                    "ground_truth": "yes" if sample.is_positive else "no",
                    "response": response_text,
                    "parsed_prediction": parse_binary(response_text),
                    "elapsed_seconds": round(time.time() - started, 3),
                    "request_id": raw_response.get("id"),
                }
            except Exception as error:  # Preserve the row so a resumed run can diagnose it.
                failures += 1
                record = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model": args.model,
                    "setting": setting,
                    "image_id": sample.image_id,
                    "target_character": sample.target_character,
                    "ground_truth": "yes" if sample.is_positive else "no",
                    "response": "",
                    "parsed_prediction": None,
                    "elapsed_seconds": round(time.time() - started, 3),
                    "error": str(error),
                }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            print(
                f"[{index}/{len(pending)}] {sample.image_id} {sample.target_character} "
                f"truth={'yes' if sample.is_positive else 'no'} response={record['response']!r}"
            )
            if failures >= args.max_failures:
                raise RuntimeError(f"Stopped after {failures} failed requests; see {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce Tables 1 and 2 from the Benchmark-Dataset repository."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary = subparsers.add_parser("summary", help="Calculate repository dataset ratios (Table 1)")
    summary.add_argument(
        "--labels", type=Path, default=Path("test/label/updated_list_identity.txt")
    )
    summary.add_argument("--output-dir", type=Path, default=Path("results"))
    summary.set_defaults(func=command_summary)

    template = subparsers.add_parser("template", help="Create a CSV template for external predictions")
    template.add_argument(
        "--labels", type=Path, default=Path("test/label/updated_list_identity.txt")
    )
    template.add_argument("--model", default="my-vlm")
    template.add_argument("--setting", default="0-shot")
    template.add_argument("--output", type=Path, default=Path("predictions/template.csv"))
    template.add_argument("--force", action="store_true")
    template.set_defaults(func=command_template)

    evaluate = subparsers.add_parser("evaluate", help="Calculate precision/recall (Table 2)")
    evaluate.add_argument(
        "--labels", type=Path, default=Path("test/label/updated_list_identity.txt")
    )
    evaluate.add_argument("--predictions", type=Path, nargs="+", required=True)
    evaluate.add_argument("--output-dir", type=Path, default=Path("results"))
    evaluate.add_argument("--default-model", default="model")
    evaluate.add_argument("--default-setting", default="0-shot")
    evaluate.add_argument("--digits", type=int, default=3)
    evaluate.add_argument("--allow-incomplete", action="store_true")
    evaluate.add_argument(
        "--require-character-match",
        action="store_true",
        help="Count Yes as positive only when the response also names the target character",
    )
    evaluate.set_defaults(func=command_evaluate)

    run = subparsers.add_parser("run", help="Run a VLM through an OpenAI-compatible endpoint")
    run.add_argument(
        "--labels", type=Path, default=Path("test/label/updated_list_identity.txt")
    )
    run.add_argument("--images", type=Path, default=Path("test/img"))
    run.add_argument("--model", required=True)
    run.add_argument("--setting", choices=["ICL", "0-shot", "icl", "zero-shot"], required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    run.add_argument("--api-key-env", default="OPENAI_API_KEY")
    run.add_argument("--icl-labels", type=Path)
    run.add_argument("--icl-images", type=Path)
    run.add_argument("--shots-per-group", type=int, default=3)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--image-detail", choices=["none", "auto", "low", "high"], default="none")
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--max-tokens", type=int, default=64)
    run.add_argument("--timeout", type=float, default=180.0)
    run.add_argument("--retries", type=int, default=3)
    run.add_argument("--max-failures", type=int, default=10)
    run.add_argument("--limit", type=int)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(func=command_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
