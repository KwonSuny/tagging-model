"""冻结词级数据版本，导入人工审核，单独留出诊断集；不改写旧数据。"""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import random
import re
import shutil
import tempfile

from clean_data import TAG_NAMES, file_hash, text_hash, validate_item
from eval import load_file
from supplement_data import validate_internal_targets


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def import_reviews(markdown, examples, reviewer, date):
    sections = re.split(r"^### (v2-[a-z]+-\d+)\s*$", markdown, flags=re.M)
    if len(sections) == 1:
        raise ValueError("没有找到审核编号")
    bodies = {}
    for index in range(1, len(sections), 2):
        identifier, body = sections[index:index + 2]
        if identifier in bodies:
            raise ValueError(f"重复审核编号: {identifier}")
        bodies[identifier] = body
    if set(bodies) != {row["augmentation"]["id"] for row in examples}:
        raise ValueError("审核文件与 JSONL 编号不一致")
    imported, groups = [], defaultdict(list)
    for original in examples:
        validate_item(original)
        validate_internal_targets(original)
        item = deepcopy(original)
        meta = item["augmentation"]
        body = bodies[meta["id"]]
        marks = re.findall(r"^-\s*\[([^\]]*)\]\s*人工审核通过\s*$", body, flags=re.M)
        if len(marks) != 1:
            raise ValueError(f"{meta['id']}: 审核框缺失或重复")
        mark = marks[0].strip()
        if mark in {"√", "✓", "x", "X"}:
            status = "approved"
        elif mark in {"×", "✗"}:
            status = "rejected"
        elif not mark:
            status = "unmarked"
        else:
            raise ValueError(f"{meta['id']}: 无法识别审核符号 {mark!r}")
        note = re.search(r"^-\s*修改意见：(.*?)(?=^## |\Z)", body, flags=re.M | re.S)
        review = {"status": status, "reviewer": reviewer if mark else None,
                  "reviewed_at": date if mark else None, "mark": mark,
                  "note": note.group(1).strip() if note else ""}
        meta["human_review"] = review
        meta["training_ready"] = False
        meta["assigned_split"] = "morpheme_research_only"
        imported.append(item)
        groups[meta["review_group"]].append((meta["id"], review))
    conflicts = []
    for group, reviews in groups.items():
        statuses = {review["status"] for _, review in reviews}
        if "approved" in statuses and "rejected" in statuses:
            conflicts.append({"group": group, "reason": "同组存在通过与否决，不能推广为规则",
                              "ids": [identifier for identifier, _ in reviews]})
    report = {"counts": dict(Counter(row["augmentation"]["human_review"]["status"] for row in imported)),
              "conflicting_groups": conflicts, "used_in_training": 0,
              "policy": "保留原意见；空白不是通过；词内语素与正式词级任务隔离"}
    return imported, report


def correct_confirmed_error(train):
    changes = []
    for row, item in enumerate(train, 1):
        for span in item["span_list"]:
            if span["type"] == "Eg":
                surface = "".join(item["tokens"][span["start"]:span["end"] + 1])
                if row != 10651 or span["start"] != 75 or surface != "喏":
                    raise ValueError("出现未确认的 Eg，拒绝套用独立喏的修正规则")
                changes.append({"input_train_line": row, "start": 75, "end": 75,
                                "surface": surface, "old_tag": "Eg", "new_tag": "e",
                                "reason": "用户已确认：独立喏为叹词"})
                span["type"] = "e"
    if len(changes) != 1:
        raise ValueError("当前输入不是预期的待修正数据版本")
    return changes


def select_diagnostic(train, ratio=0.05, seed=42):
    """先满足稀有类与对照，再随机补足；稀有类一般留80%，2至4段时最多留出1段。"""
    count = round(len(train) * ratio)
    order = list(range(len(train)))
    random.Random(seed).shuffle(order)
    row_tags = [set(span["type"] for span in item["span_list"]) for item in train]
    support = Counter(tag for tags in row_tags for tag in tags)
    used = Counter()
    selected = set()

    def take(index):
        if index in selected or len(selected) >= count:
            return False
        for tag in row_tags[index]:
            total = support[tag]
            cap = max(1, int(total * 0.2)) if total > 1 else 0
            if used[tag] >= (cap if total < 50 else total - 1):
                return False
        selected.add(index)
        used.update(row_tags[index])
        return True

    goals = {"Rg": 1, "Ug": 2, "dq": 20, "Mg": 1, "e": 4,
             "Bg": 5, "Qg": 6, "Dg": 10, "h": 12}
    for tag, goal in goals.items():
        for index in order:
            if used[tag] >= goal:
                break
            if tag in row_tags[index]:
                take(index)
    for word, tags in [("发展", ("v", "vn")), ("研究", ("v", "vn")),
                       ("工作", ("v", "vn", "n")), ("学习", ("v", "vn"))]:
        for tag in tags:
            candidates = [index for index in order if any(
                span["type"] == tag and "".join(train[index]["tokens"][span["start"]:span["end"] + 1]) == word
                for span in train[index]["span_list"])]
            if len(candidates) > 1:
                for index in candidates:
                    if take(index):
                        break
    for index in order:
        take(index)
    if len(selected) != count:
        raise ValueError("无法在保留训练类别覆盖的前提下满足诊断集数量")
    remaining = set().union(*(row_tags[i] for i in range(len(train)) if i not in selected))
    if remaining != set(support):
        raise ValueError("诊断划分导致训练丢失类别")
    return selected


def summary(rows):
    counts = Counter(span["type"] for row in rows for span in row["span_list"])
    return {"paragraphs": len(rows), "characters": sum(len(row["tokens"]) for row in rows),
            "words": sum(counts.values()), "tag_counts": dict(sorted(counts.items())),
            "tag_paragraph_counts": dict(sorted(Counter(
                tag for row in rows for tag in {s["type"] for s in row["span_list"]}).items()))}


def make_review_queue(train, old_lines):
    # 同词异标不一定错误，保留上下文供人工判断，禁止按多数标签替换。
    words = {"公司", "农村", "大力", "政治局", "常委会", "人民法院", "生活"}
    counts, examples = defaultdict(Counter), defaultdict(list)
    for row, item in enumerate(train, 1):
        for span in item["span_list"]:
            start, end, tag = span["start"], span["end"], span["type"]
            surface = "".join(item["tokens"][start:end + 1])
            if surface not in words:
                continue
            counts[surface][tag] += 1
            bucket = examples[(surface, tag)]
            if len(bucket) < 5:
                bucket.append({"train_line": row, "input_train_line": old_lines[row - 1],
                               "word": surface, "tag": tag, "start": start, "end": end,
                               "context": "".join(item["tokens"][max(0, start - 30):end + 31]),
                               "text_sha256": text_hash(item["tokens"]), "review_status": "pending"})
    return {"status": "pending_human_review_not_applied",
            "word_tag_counts": {word: dict(count) for word, count in sorted(counts.items())},
            "examples": [example for bucket in examples.values() for example in bucket]}


def prepare(source, output, reviewer="sunyeu", date="2026-10-10", ratio=0.05, seed=42, review_dir=None):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有数据版本: {output}")
    if not 0 < ratio < 0.5:
        raise ValueError("诊断集比例必须在 0 与 0.5 之间")
    old_report = json.loads((source / "cleaning_report.json").read_text(encoding="utf-8"))
    for split in ("train", "dev", "test"):
        if file_hash(source / f"{split}.jsonline") != old_report["outputs"][split]["sha256"]:
            raise ValueError(f"{split} 文件与清洗报告不一致")
    train, dev = load_file(source / "train.jsonline"), load_file(source / "dev.jsonline")
    changes = correct_confirmed_error(train)
    selected = select_diagnostic(train, ratio, seed)
    lines = [i + 1 for i in range(len(train)) if i not in selected]
    diagnostic_lines = [i + 1 for i in sorted(selected)]
    splits = {"train": [train[i - 1] for i in lines], "dev": dev,
              "diagnostic_dev": [train[i - 1] for i in diagnostic_lines]}
    all_hashes = set()
    for name, rows in splits.items():
        for item in rows:
            validate_item(item)
            digest = text_hash(item["tokens"])
            if digest in all_hashes:
                raise ValueError(f"{name}: 集合内重复或集合间文本泄漏")
            all_hashes.add(digest)
    # 测试标签保持封存；从旧审计的文本摘要检查隔离，不以测试类别分布做选样。
    audit = load_file(source / "cleaning_audit.jsonl")
    test_hashes = {row["clean_text_sha256"] for row in audit if row["status"] == "test"}
    if all_hashes & test_hashes or len(test_hashes) != old_report["outputs"]["test"]["paragraphs"]:
        raise ValueError("与封存测试集重叠，或旧审计不完整")
    source_rows = {(r["status"], r["output_line"]): r for r in audit if r["status"] != "rejected"}
    stats = {name: summary(rows) for name, rows in splits.items()}
    observed = sorted(stats["train"]["tag_counts"])
    if any(set(s["tag_counts"]) - set(observed) for s in stats.values()):
        raise ValueError("验证类别不在训练中")
    catalog = json.loads((source / "label_catalog.json").read_text(encoding="utf-8"))
    labels = ["O"] + [f"{prefix}-{tag}" for tag in observed for prefix in ("B", "I")]
    catalog.update(observed_tags=observed, id_to_label={str(i): tag for i, tag in enumerate(labels)},
                   label_to_id={tag: i for i, tag in enumerate(labels)},
                   registered_but_absent_tags=sorted(set(TAG_NAMES) - set(observed)))
    review_dir = Path(review_dir) if review_dir is not None else source / "augmentation_pending_v2"
    reviewed, reviews = import_reviews((review_dir / "review.md").read_text(encoding="utf-8"),
                                      load_file(review_dir / "rare_morphemes.jsonline"), reviewer, date)
    inputs = {str(p): file_hash(p) for p in [source / "train.jsonline", source / "dev.jsonline",
              source / "test.jsonline", source / "label_catalog.json", source / "cleaning_audit.jsonl",
              source / "cleaning_report.json", review_dir / "review.md", review_dir / "rare_morphemes.jsonline"]}
    provenance = []
    for name, source_split, input_lines in [("train", "train", lines), ("diagnostic_dev", "train", diagnostic_lines),
                                            ("dev", "dev", list(range(1, len(dev) + 1)))]:
        for row, old_line in enumerate(input_lines, 1):
            origin = source_rows[(source_split, old_line)]
            provenance.append({"split": name, "output_line": row, "input_split": source_split,
                               "input_line": old_line, "source": origin["source"], "source_line": origin["source_line"],
                               "text_sha256": text_hash(splits[name][row - 1]["tokens"]),
                               "repairs": origin["repairs"] + [c for c in changes if source_split == "train" and c["input_train_line"] == old_line]})
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output.parent) as work:
        stage = Path(work) / "result"
        stage.mkdir()
        stage.resolve().relative_to(output.parent.resolve())
        for name, rows in splits.items():
            write_jsonl(stage / f"{name}.jsonline", rows)
        # 官方 dev 和封存 test 逐字节保留。
        shutil.copyfile(source / "dev.jsonline", stage / "dev.jsonline")
        shutil.copyfile(source / "test.jsonline", stage / "test.jsonline")
        write_json(stage / "label_catalog.json", catalog)
        write_jsonl(stage / "provenance.jsonl", provenance)
        write_jsonl(stage / "morpheme_research.jsonline", reviewed)
        write_json(stage / "morpheme_review.json", reviews)
        write_json(stage / "annotation_review_queue.json", make_review_queue(splits["train"], lines))
        checksums = {p.name: file_hash(p) for p in stage.iterdir() if p.is_file()}
        manifest = {"pipeline": "word-segpos-prepared-v3", "created_at": date, "seed": seed,
                    "input_sha256": inputs, "generator_sha256": file_hash(Path(__file__)),
                    "corrections": changes, "splits": stats,
                    "test": {"paragraphs": len(test_hashes), "sealed": True, "sha256": checksums["test.jsonline"]},
                    "diagnostic_policy": {"ratio": ratio, "unit": "完整段落", "rare_goals": "优先 Rg/Ug/dq 等，保留训练类别",
                                          "report_separately": True, "independent_only_after_retraining": True},
                    "review": reviews, "observed_tags": observed, "bio_labels": len(labels),
                    "official_dev_missing_tags": sorted(set(observed) - set(stats["dev"]["tag_counts"])),
                    "all_dev_missing_tags": sorted(set(observed) - set(stats["dev"]["tag_counts"]) - set(stats["diagnostic_dev"]["tag_counts"])),
                    "checks": {"full_span_coverage": True, "unknown_tags": [], "text_overlap": 0,
                               "test_bytes_unchanged": checksums["test.jsonline"] == inputs[str(source / "test.jsonline")],
                               "official_dev_bytes_unchanged": checksums["dev.jsonline"] == inputs[str(source / "dev.jsonline")]},
                    "limitations": ["源数据没有文章ID，仅保证段落级隔离", "同词异标仍待人工判定，未按多数标签改写",
                                    "词内语素研究材料未进入训练或验证", "87标签新模型必须从原始BERT重新训练，不恢复89标签旧检查点"],
                    "output_sha256": checksums}
        write_json(stage / "manifest.json", manifest)
        stage.rename(output)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data")
    parser.add_argument("--output", default="data/prepared_v3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--review-dir", default="data/augmentation_pending_v2")
    parser.add_argument("--diagnostic-ratio", type=float, default=0.05)
    args = parser.parse_args()
    result = prepare(args.source, args.output, ratio=args.diagnostic_ratio, seed=args.seed, review_dir=args.review_dir)
    print(json.dumps({"sizes": {k: v["paragraphs"] for k, v in result["splits"].items()},
                      "labels": result["bio_labels"], "review": result["review"]["counts"]}, ensure_ascii=False))
