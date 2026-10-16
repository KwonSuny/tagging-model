"""清洗原始 PKU-SEGPOS 数据，保留完整段落并生成独立测试集。"""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import random
import shutil
import tempfile


TAG_NAMES = {
    "a": "形容词", "ad": "副形词", "an": "名形词", "b": "区别词",
    "c": "连词", "d": "副词", "dq": "源语料扩展副词类", "e": "叹词",
    "f": "方位词", "g": "语素", "h": "前接成分", "i": "成语",
    "j": "简称", "k": "后接成分", "l": "习用语", "m": "数词",
    "n": "名词", "nr": "人名", "ns": "地名", "nt": "机构团体名",
    "nx": "外文字符", "nz": "其他专名", "o": "拟声词", "p": "介词",
    "q": "量词", "r": "代词", "s": "处所词", "t": "时间词",
    "u": "助词", "v": "动词", "vd": "副动词", "vn": "名动词",
    "w": "标点符号", "x": "非语素字", "y": "语气词", "z": "状态词",
    "Ag": "形容语素", "Bg": "区别语素", "Dg": "副语素", "Eg": "叹词语素",
    "Mg": "数语素", "Ng": "名语素", "Qg": "量语素", "Rg": "代语素",
    "Tg": "时间语素", "Ug": "助语素", "Vg": "动语素", "Yg": "语气语素",
    "Zg": "状态语素",
}
TAG_ALIASES = {
    "dk": "d", "ds": "d", "dy": "d", "ly": "l", "mj": "m",
    "n1": "n", "n2": "n", "n3": "n", "n6": "n", "fs": "f",
    "wyy": "w", "wky": "w", "wj": "w",
    **{tag.lower(): tag for tag in TAG_NAMES if tag.endswith("g") and len(tag) == 2},
}
PHRASE_TAGS = {"ns", "nt", "nz", "i", "l", "ly"}
PIPELINE = "pku-segpos-clean-v2"


def text_hash(tokens):
    return hashlib.sha256("".join(tokens).encode("utf-8")).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_tag(tag, surface):
    if not isinstance(tag, str):
        raise ValueError("non_string_tag")
    parts = tag.split("]")
    if any(part not in PHRASE_TAGS for part in parts[1:]):
        raise ValueError("unknown_phrase_tag")
    base = parts[0]
    if base == "〓" and surface == "〓":
        base = "w"
    normalized = TAG_ALIASES.get(base, base)
    if normalized not in TAG_NAMES:
        raise ValueError("unknown_pos_tag")
    return normalized, parts[1:]


@dataclass
class Record:
    source: str
    source_line: int
    item: dict | None = None
    repairs: list = field(default_factory=list)
    reasons: list = field(default_factory=list)
    source_text_sha256: str | None = None
    status: str = "candidate"
    output_line: int | None = None

    def reject(self, reason):
        self.status = "rejected"
        if reason not in self.reasons:
            self.reasons.append(reason)


def validate_item(item):
    if not isinstance(item, dict):
        raise ValueError("record_not_object")
    tokens, spans = item.get("tokens"), item.get("span_list")
    if not isinstance(tokens, list) or not tokens:
        raise ValueError("empty_or_invalid_tokens")
    if any(not isinstance(token, str) or len(token) != 1 for token in tokens):
        raise ValueError("tokens_not_single_characters")
    if not isinstance(spans, list) or not spans:
        raise ValueError("empty_or_invalid_spans")
    cursor = 0
    for span in spans:
        if not isinstance(span, dict):
            raise ValueError("invalid_span_object")
        start, end = span.get("start"), span.get("end")
        if type(start) is not int or type(end) is not int:
            raise ValueError("non_integer_span")
        if not 0 <= start <= end < len(tokens):
            raise ValueError("invalid_span_range")
        if start != cursor:
            raise ValueError("coverage_gap_overlap_or_order")
        if not isinstance(span.get("type"), str) or span["type"] not in TAG_NAMES:
            raise ValueError("unregistered_tag")
        cursor = end + 1
    if cursor != len(tokens):
        raise ValueError("coverage_gap_overlap_or_order")


def apply_reviewed_repairs(record):
    tokens, spans = record.item["tokens"], record.item["span_list"]
    text = "".join(tokens)
    removed = set()
    revised = []
    for span in spans:
        start, end, tag = span["start"], span["end"], span["type"]
        surface = text[start:end + 1]
        if tag == "Mg" and surface == "卯" and text[max(0, start - 1):end + 3] == "榫卯工艺":
            record.repairs.append({
                "kind": "reviewed_morpheme_tag", "start": start, "end": end,
                "surface": surface, "before": "Mg", "after": "Ng",
                "reason": "2026-10-10 用户确认榫卯工艺中的卯为名词性语素",
            })
            revised.append({**span, "type": "Ng"})
        elif tag == "Ng" and surface == "长!B":
            # 用户明确只删除 B；保留感叹号并独立标为标点。
            removed.add(end)
            revised.extend([
                {"start": start, "end": start, "type": "Ng"},
                {"start": start + 1, "end": start + 1, "type": "w"},
            ])
            record.repairs.append({
                "kind": "reviewed_remove_stray_B", "start": start, "end": end,
                "before": "长!B/Ng", "after": "长/Ng !/w", "removed_position": end,
                "reason": "2026-10-10 用户确认三处长!B源数据错误，应删除B",
            })
        else:
            revised.append(dict(span))
    removed_before = [0]
    for position in range(len(tokens)):
        removed_before.append(removed_before[-1] + (position in removed))
    record.item = {
        "tokens": [token for position, token in enumerate(tokens) if position not in removed],
        "span_list": [{"start": span["start"] - removed_before[span["start"]],
                       "end": span["end"] - removed_before[span["end"] + 1],
                       "type": span["type"]} for span in revised],
    }


def clean_item(item, source="train.jsonline", source_line=1):
    record = Record(source, source_line)
    if not isinstance(item, dict):
        record.reject("record_not_object")
        return record
    tokens, spans = item.get("tokens"), item.get("span_list")
    if not isinstance(tokens, list) or not tokens or any(
        not isinstance(token, str) or len(token) != 1 for token in tokens
    ):
        record.reject("empty_or_invalid_character_tokens")
        return record
    record.source_text_sha256 = text_hash(tokens)
    if not isinstance(spans, list) or not spans:
        record.reject("empty_or_invalid_spans")
        return record

    normalized = []
    stack = []
    remove_positions = set()
    cursor = 0
    for span_index, span in enumerate(spans):
        if not isinstance(span, dict):
            record.reject("invalid_span_object")
            continue
        start, end = span.get("start"), span.get("end")
        if type(start) is not int or type(end) is not int:
            record.reject("non_integer_span")
            continue
        # 两条空区间不承载字符，删除后仍须通过完整覆盖检查。
        if 0 <= start <= len(tokens) and end == start - 1:
            record.repairs.append({"kind": "remove_empty_span", "span_index": span_index, "before": span})
            continue
        if not 0 <= start <= end < len(tokens):
            record.reject("invalid_span_range")
            continue
        if start != cursor:
            record.reject("coverage_gap_overlap_or_order")
        cursor = end + 1
        surface = "".join(tokens[start:end + 1])
        try:
            tag, closers = normalize_tag(span.get("type"), surface)
        except ValueError as error:
            record.reject(str(error))
            continue
        if tag != span["type"]:
            record.repairs.append({
                "kind": "normalize_tag", "span_index": span_index,
                "before": span["type"], "after": tag,
            })
        for position in range(start, end + 1):
            if tokens[position] == "[":
                if position != start:
                    record.reject("phrase_marker_inside_word")
                stack.append(position)
        for _ in closers:
            if not stack:
                record.reject("unmatched_phrase_closing")
            else:
                remove_positions.add(stack.pop())
        normalized.append({"start": start, "end": end, "type": tag})
    if cursor != len(tokens):
        record.reject("coverage_gap_overlap_or_order")
    if stack:
        record.reject("unmatched_phrase_opening")
    if record.status == "rejected":
        return record

    # 前缀计数使每个位置的重映射只依赖确实匹配的元标记。
    removed_before = [0]
    for position in range(len(tokens)):
        removed_before.append(removed_before[-1] + (position in remove_positions))
    clean_tokens = [token for position, token in enumerate(tokens) if position not in remove_positions]
    clean_spans = []
    for span in normalized:
        start, end = span["start"], span["end"]
        new_start = start - removed_before[start]
        new_end = end - removed_before[end + 1]
        if new_start > new_end:
            record.reject("word_only_contains_phrase_marker")
            return record
        clean_spans.append({"start": new_start, "end": new_end, "type": span["type"]})
    if remove_positions:
        record.repairs.append({"kind": "remove_phrase_openings", "positions": sorted(remove_positions)})
    record.item = {"tokens": clean_tokens, "span_list": clean_spans}
    apply_reviewed_repairs(record)
    try:
        validate_item(record.item)
    except ValueError as error:
        record.reject(str(error))
    return record


def annotation_key(record):
    return tuple((span["start"], span["end"], span["type"]) for span in record.item["span_list"])


def deduplicate(records):
    groups = defaultdict(list)
    for record in records:
        if record.status == "candidate":
            groups[tuple(record.item["tokens"])].append(record)
    retained = []
    for group in groups.values():
        if len({annotation_key(record) for record in group}) > 1:
            for record in group:
                record.reject("conflicting_annotations_for_same_text")
            continue
        retained.append(group[0])
        for record in group[1:]:
            record.reject("duplicate_text")
    return retained


def tags_of(record):
    return {span["type"] for span in record.item["span_list"]}


def split_training(records, test_ratio, seed):
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")
    target = int(len(records) * test_ratio + 0.5)
    if target < 1 or target >= len(records):
        raise ValueError("not enough training paragraphs for requested split")
    support = Counter(tag for record in records for tag in tags_of(record))
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    selected = set()
    # 抽出测试段落时保留训练标签覆盖，避免把唯一的稀有类别留给测试集。
    for record in shuffled:
        if len(selected) == target:
            break
        tags = tags_of(record)
        if any(support[tag] <= 1 for tag in tags):
            continue
        selected.add(id(record))
        support.subtract(tags)
    if len(selected) != target:
        raise ValueError("requested split would remove training label coverage")
    train = [record for record in records if id(record) not in selected]
    test = [record for record in records if id(record) in selected]
    return train, test


def validate_splits(splits):
    text_sets = {}
    tag_sets = {}
    result = {}
    for name, records in splits.items():
        texts = set()
        tags = set()
        for record in records:
            validate_item(record.item)
            key = tuple(record.item["tokens"])
            if key in texts:
                raise ValueError(f"duplicate text in {name}")
            texts.add(key)
            tags.update(tags_of(record))
        text_sets[name], tag_sets[name] = texts, tags
    names = list(splits)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlap = len(text_sets[left] & text_sets[right])
            result[f"{left}_{right}_text_overlap"] = overlap
            if overlap:
                raise ValueError(f"text overlap between {left} and {right}")
    for name in ["dev", "test"]:
        missing = sorted(tag_sets[name] - tag_sets["train"])
        result[f"{name}_tags_missing_from_train"] = missing
        if missing:
            raise ValueError(f"unseen labels in {name}: {missing}")
    result["unregistered_tags"] = sorted(set.union(*tag_sets.values()) - TAG_NAMES.keys())
    result["complete_character_coverage"] = True
    result["valid_inclusive_span_ranges"] = True
    return result


def summarize(records):
    tags = Counter()
    lengths = []
    for record in records:
        lengths.append(len(record.item["tokens"]))
        tags.update(span["type"] for span in record.item["span_list"])
    return {
        "paragraphs": len(records), "characters": sum(lengths), "words": sum(tags.values()),
        "max_characters": max(lengths, default=0), "pos_types": len(tags),
        "tag_counts": dict(sorted(tags.items())),
    }


def write_json(path, value):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def prepare_data(source_dir, output_dir, test_ratio=0.1, seed=42):
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    if output_dir == source_dir or output_dir in source_dir.parents or source_dir in output_dir.parents:
        raise ValueError("output directory must be separate from the source directory")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}; use a new --output")
    records_by_source = {}
    raw_tags = Counter()
    source_info = {}
    for split in ["train", "dev"]:
        filename = split + ".jsonline"
        path = source_dir / filename
        records = []
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    record = Record(filename, line_number)
                    record.reject("invalid_json")
                else:
                    if isinstance(item, dict) and isinstance(item.get("span_list"), list):
                        for span in item["span_list"]:
                            if isinstance(span, dict):
                                raw_tags[str(span.get("type"))] += 1
                    record = clean_item(item, filename, line_number)
                records.append(record)
        records_by_source[split] = records
        source_info[filename] = {"sha256": file_hash(path), "rows": len(records)}
    public_path = source_dir / "test_public.jsonline"
    if public_path.exists():
        source_info[public_path.name] = {"sha256": file_hash(public_path), "used_for_labeled_splits": False}

    # 跨开发集的冲突也整组剔除；同标注重复文本优先保留官方开发集。
    groups = defaultdict(list)
    all_records = records_by_source["train"] + records_by_source["dev"]
    for record in all_records:
        if record.status == "candidate":
            groups[tuple(record.item["tokens"])].append(record)
    for group in groups.values():
        if len({annotation_key(record) for record in group}) > 1:
            for record in group:
                record.reject("conflicting_annotations_for_same_text")
    train_pool = deduplicate(records_by_source["train"])
    dev = deduplicate(records_by_source["dev"])
    dev_texts = {tuple(record.item["tokens"]) for record in dev}
    exclusive_pool = []
    for record in train_pool:
        if tuple(record.item["tokens"]) in dev_texts:
            record.reject("duplicate_with_official_dev")
        else:
            exclusive_pool.append(record)
    train, test = split_training(exclusive_pool, test_ratio, seed)
    splits = {"train": train, "dev": dev, "test": test}
    checks = validate_splits(splits)
    for name, records in splits.items():
        for line_number, record in enumerate(records, 1):
            record.status, record.output_line = name, line_number

    unknown_raw_tags = {}
    for tag, count in sorted(raw_tags.items()):
        try:
            normalize_tag(tag, "〓" if tag == "〓" else "")
        except ValueError:
            unknown_raw_tags[tag] = count
    observed = sorted({tag for records in splits.values() for record in records for tag in tags_of(record)})
    label_to_id = {"O": 0}
    for tag in observed:
        for prefix in ["B", "I"]:
            label_to_id[f"{prefix}-{tag}"] = len(label_to_id)
    catalog = {
        "registered_tags": TAG_NAMES, "tag_aliases": TAG_ALIASES,
        "phrase_suffixes": sorted(PHRASE_TAGS), "observed_tags": observed,
        "registered_but_absent_tags": sorted(TAG_NAMES.keys() - set(observed)),
        "label_to_id": label_to_id, "id_to_label": {str(value): key for key, value in label_to_id.items()},
        "scope": "词级词性；短语结束标记不作为词性，也不合并短语内的词",
        "references": [
            "PKU-SEGPOS/README-ZH.md",
            "https://www.tup.tsinghua.edu.cn/upload/books/yz/090505-01.pdf",
        ],
        "extension_note": "保留源语料中的合法语素子类与 nx；Eg/Zg 按语素命名规则登记，dq 保留源扩展名称。",
    }
    report = {
        "pipeline": PIPELINE, "source_files": source_info,
        "split": {"seed": seed, "requested_test_ratio": test_ratio, "clean_train_pool": len(exclusive_pool),
                  "actual_test_ratio": len(test) / len(exclusive_pool), "unit": "完整段落，按清洗后文本去重",
                  "policy": "保留原开发集来源，清洗后从原训练集划测试集；训练保留全部已出现类别"},
        "outputs": {name: summarize(records) for name, records in splits.items()},
        "raw_tag_counts": dict(sorted(raw_tags.items())), "unknown_raw_tags_rejected": unknown_raw_tags,
        "source_summary": {}, "checks": checks,
        "limitations": ["源 JSON 无文章 ID，无法保证文章级隔离", "测试集来自训练时期的语料，属于自建测试集",
                        "旧模型标签映射与新数据不同，应重新训练；根目录旧数据未改动"],
    }
    for name, records in records_by_source.items():
        report["source_summary"][name] = {
            "input_rows": len(records), "retained_rows": sum(record.status != "rejected" for record in records),
            "rejected_rows": sum(record.status == "rejected" for record in records),
            "rejection_reason_counts": dict(sorted(Counter(reason for record in records for reason in record.reasons).items())),
            "retained_repair_counts": dict(sorted(Counter(
                repair["kind"] for record in records if record.status != "rejected" for repair in record.repairs
            ).items())),
            "retained_removed_phrase_characters": sum(
                len(repair["positions"]) for record in records if record.status != "rejected"
                for repair in record.repairs if repair["kind"] == "remove_phrase_openings"
            ),
        }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pku-clean-", dir=output_dir.parent) as temp:
        stage = Path(temp)
        for name, records in splits.items():
            path = stage / (name + ".jsonline")
            with path.open("w", encoding="utf-8", newline="\n") as stream:
                for record in records:
                    stream.write(json.dumps(record.item, ensure_ascii=False, separators=(",", ":")) + "\n")
            report["outputs"][name]["sha256"] = file_hash(path)
        write_json(stage / "label_catalog.json", catalog)
        with (stage / "cleaning_audit.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for record in all_records:
                audit = {
                    "source": record.source, "source_line": record.source_line, "status": record.status,
                    "output_line": record.output_line, "reasons": record.reasons, "repairs": record.repairs,
                    "source_text_sha256": record.source_text_sha256,
                    "clean_text_sha256": text_hash(record.item["tokens"]) if record.item else None,
                }
                stream.write(json.dumps(audit, ensure_ascii=False, separators=(",", ":")) + "\n")
        # 落盘后重新读取并校验，报告与实际交付文件共用同一检查标准。
        reloaded = {}
        for name in splits:
            with (stage / (name + ".jsonline")).open(encoding="utf-8") as stream:
                reloaded[name] = [Record(name, index, json.loads(line)) for index, line in enumerate(stream, 1)]
        if validate_splits(reloaded) != checks:
            raise ValueError("written dataset verification failed")
        for filename, info in source_info.items():
            if file_hash(source_dir / filename) != info["sha256"]:
                raise ValueError("source dataset changed during cleaning")
        write_json(stage / "cleaning_report.json", report)
        # 完成报告最后发布；复制后校验摘要，失败时移除本次创建的文件。
        output_dir.mkdir()
        published = []
        try:
            paths = sorted(stage.iterdir(), key=lambda path: (path.name == "cleaning_report.json", path.name))
            for path in paths:
                destination = output_dir / path.name
                published.append(destination)
                shutil.copyfile(path, destination)
                if file_hash(path) != file_hash(destination):
                    raise OSError(f"published file checksum mismatch: {path.name}")
        except OSError:
            for path in published:
                path.unlink(missing_ok=True)
            output_dir.rmdir()
            raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("PKU-SEGPOS"))
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = prepare_data(args.source, args.output, args.test_ratio, args.seed)
    print(json.dumps({"output": str(args.output), "counts": {
        name: stats["paragraphs"] for name, stats in report["outputs"].items()
    }, "checks": report["checks"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
