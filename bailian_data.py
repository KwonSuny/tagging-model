"""百炼词级 SFT 数据导出与严格评测；本模块不调用云端或上传数据。"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import tempfile

from clean_data import file_hash, text_hash, validate_item
from eval import evaluate_data, load_file
from prepare_dataset import summary, write_json, write_jsonl


def system_prompt(tags):
    descriptions = "；".join(f"{tag}={name}" for tag, name in sorted(tags.items()))
    return ("对用户JSON中text的中文原文进行完整词分词和词级词性标注。"
            "按上下文确定词边界和词性，沿用本项目语料的单层词级分词口径。"
            "保留语料原有的词级语素标签，不新增词内部的另一层语素标注。"
            "标点也要标注，原文字符不得增删或改写。"
            "只输出JSON对象，格式为{\"words\":[[\"完整词\",\"词性\"],...]}。"
            "词按原文顺序排列，拼接后必须与text完全相同。不要解释、思考过程、Markdown或坐标。"
            "允许的词性：" + descriptions)


def make_messages(item, prompt, answer=True):
    validate_item(item)
    messages = [{"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps({"text": "".join(item["tokens"])}, ensure_ascii=False, separators=(",", ":"))}]
    if answer:
        words = [["".join(item["tokens"][span["start"]:span["end"] + 1]), span["type"]] for span in item["span_list"]]
        messages.append({"role": "assistant", "content": json.dumps({"words": words}, ensure_ascii=False, separators=(",", ":"))})
    return messages


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"重复JSON键: {key}")
        result[key] = value
    return result


def parse_response(content, tokens, allowed_tags):
    if not isinstance(content, str):
        raise ValueError("content 必须是字符串")
    obj = json.loads(content, object_pairs_hook=reject_duplicate_keys)
    if not isinstance(obj, dict) or set(obj) != {"words"} or not isinstance(obj["words"], list):
        raise ValueError("答案必须是仅含 words 数组的 JSON 对象")
    cursor, spans, words = 0, [], []
    for entry in obj["words"]:
        if not isinstance(entry, list) or len(entry) != 2:
            raise ValueError("每项必须是 [完整词, 词性]")
        word, tag = entry
        if not isinstance(word, str) or not word or not isinstance(tag, str) or tag not in allowed_tags:
            raise ValueError("空词、非法词或未登记词性")
        spans.append({"start": cursor, "end": cursor + len(word) - 1, "type": tag})
        cursor += len(word)
        words.append(word)
    if "".join(words) != "".join(tokens):
        raise ValueError("输出词拼接与原文不一致，存在增删、改写或乱序")
    result = {"tokens": tokens, "span_list": spans}
    validate_item(result)
    return result


def select_pilot(rows, count, seed):
    if not 0 < count <= len(rows):
        raise ValueError("试验集大小必须为正且不大于原集合")
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    tags = [{s["type"] for s in row["span_list"]} for row in rows]
    support = Counter(tag for present in tags for tag in present)
    selected, covered = set(), set()
    for tag in sorted(support, key=lambda key: (support[key], key)):
        if tag not in covered:
            index = next(index for index in order if tag in tags[index])
            selected.add(index)
            covered.update(tags[index])
    if len(selected) > count:
        raise ValueError("试验集不足以覆盖全部训练类别")
    for index in order:
        if len(selected) == count:
            break
        selected.add(index)
    return sorted(selected)


def training_plan(reports):
    pilot_tokens = reports["pilot_train"].get("tokens", {})
    full_tokens = reports["train"].get("tokens", {})
    return {"status": "local_plan_only_no_cloud_action", "region": "北京", "model": "qwen3-8b",
            "training_type": "efficient_sft",
            "pilot": {"training_file": "pilot_train.jsonl", "validation_file": "pilot_dev.jsonl", "n_epochs": 1,
                      "estimated_training_yuan": pilot_tokens.get("estimated_training_yuan_per_epoch")},
            "full": {"training_file": "train.jsonl", "validation_file": "dev.jsonl", "n_epochs": 2,
                     "estimated_training_yuan": round(full_tokens["total"] * .006 / 1000 * 2, 2) if full_tokens else None},
            "suggested_hyperparameters": {"learning_rate": 1e-4, "batch_size": 16, "max_length": 8192,
                "lr_scheduler_type": "linear", "warmup_ratio": .05, "weight_decay": .01, "eval_steps": 50,
                "lora_rank": 8, "lora_alpha": 16, "lora_dropout": .1},
            "data_augmentation": False,
            "validation_policy": "选择独立验证数据集，关闭自动切分；diagnostic_dev不并入官方dev",
            "full_parameter_comparison": {"training_type": "sft", "learning_rate": 2e-5, "n_epochs": 2,
                "note": "不配置LoRA参数；平台当前单价相同，待先确认小试格式和效果后进行对照"},
            "checkpoint_policy": "按epoch保存并导出，后续按本地联合F1选取，不能默认最后一轮最佳",
            "caution": "起点方案不是已验证最优参数；各模型实际可选项、最终Token费用和部署费以控制台为准；没有数据集ID，此文件不是可直接提交的API请求"}


def export(source, output, template, tokenizer_path=None, max_length=8192, pilot_size=2048, seed=42):
    source, output, template = Path(source), Path(output), Path(template)
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有导出: {output}")
    version = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in version["output_sha256"].items():
        if file_hash(source / name) != digest:
            raise ValueError(f"数据版本摘要不一致: {name}")
    demo = load_file(template)
    for row in demo:
        if set(row) != {"messages"} or not isinstance(row["messages"], list):
            raise ValueError("官方示例并非 messages 格式")
        for message in row["messages"]:
            if set(message) != {"role", "content"} or message["role"] not in {"system", "user", "assistant"} or not isinstance(message["content"], str):
                raise ValueError("官方示例 role/content 结构异常")
    catalog = json.loads((source / "label_catalog.json").read_text(encoding="utf-8"))
    allowed = set(catalog["observed_tags"])
    prompt = system_prompt({tag: catalog["registered_tags"][tag] for tag in allowed})
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    sets = {name: load_file(source / f"{name}.jsonline") for name in ("train", "dev", "diagnostic_dev")}
    pilot = select_pilot(sets["train"], pilot_size, seed)
    pilot_dev = select_pilot(sets["dev"], min(512, len(sets["dev"])), seed)
    plans = {name: (rows, list(range(len(rows))), name) for name, rows in sets.items()}
    plans["pilot_train"] = ([sets["train"][i] for i in pilot], pilot, "train")
    plans["pilot_dev"] = ([sets["dev"][i] for i in pilot_dev], pilot_dev, "dev")
    tokenizer = None
    token_info = {"status": "not_measured", "caution": "需先用目标模型分词器或百炼控制台确认长度，禁止静默截断"}
    if tokenizer_path:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, use_fast=True)
        files = {p.name: file_hash(p) for p in Path(tokenizer_path).iterdir() if p.is_file()}
        token_info = {"status": "local_target_tokenizer_measured", "model": "Qwen/Qwen3-8B",
                      "files_sha256": files, "enable_thinking": False,
                      "caution": "含本地官方聊天模板开销；百炼具体模板和计费以控制台为准"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".bailian-", dir=output.parent) as work:
        stage = Path(work) / "result"
        stage.mkdir()
        stage.resolve().relative_to(output.parent.resolve())
        write_json(stage / "inference_config.json", {"candidate_model": "qwen3-8b", "enable_thinking": False,
                    "temperature": 0, "max_tokens": 8192, "system_prompt": prompt,
                    "system_prompt_sha256": prompt_hash, "response_schema": {"words": [["完整词", "词性"]]}})
        reports, provenance = {}, []
        for name, (rows, indices, origin) in plans.items():
            print(f"导出 {name}: {len(rows)} 段", flush=True)
            training, requests, lengths, answers = [], [], [], []
            for position, (item, index) in enumerate(zip(rows, indices), 1):
                identifier = f"{origin}:{index + 1}"
                digest = text_hash(item["tokens"])
                messages = make_messages(item, prompt)
                parsed = parse_response(messages[-1]["content"], item["tokens"], allowed)
                if parsed["span_list"] != item["span_list"]:
                    raise ValueError(f"{name}:{position}: 分词词性往返不一致")
                training.append({"messages": messages})
                requests.append({"id": identifier, "text_sha256": digest, "system_prompt_sha256": prompt_hash,
                                 "messages": messages[:2]})
                provenance.append({"export": name, "output_line": position, "id": identifier,
                                   "source_file": f"{origin}.jsonline", "source_line": index + 1, "text_sha256": digest})
            if tokenizer:
                # 成批编码，既核查最大长度，也核算整份数据的 Token 数。
                for start in range(0, len(training), 128):
                    batch = training[start:start + 128]
                    rendered = [tokenizer.apply_chat_template(r["messages"], tokenize=False,
                                add_generation_prompt=False, enable_thinking=False) for r in batch]
                    lengths.extend(map(len, tokenizer(rendered, add_special_tokens=False)["input_ids"]))
                    answers.extend(map(len, tokenizer([r["messages"][-1]["content"] for r in batch], add_special_tokens=False)["input_ids"]))
                if max(lengths) > max_length:
                    raise ValueError(f"{name}: 最大 {max(lengths)} tokens 超过 max_length={max_length}，拒绝截断")
            sft_file = stage / f"{name}.jsonl"
            write_jsonl(sft_file, training)
            if sft_file.stat().st_size > 200 * 1024 * 1024:
                raise ValueError(f"{name}: 超过百炼单文件200MB限制，请分片")
            write_jsonl(stage / f"{name}.requests.jsonl", requests)
            if name == "pilot_dev":
                write_jsonl(stage / "pilot_dev.gold.jsonline", rows)
            stats = summary(rows)
            if lengths:
                ordered = sorted(lengths)
                stats["tokens"] = {"total": sum(lengths), "maximum": max(lengths),
                                   "p95": ordered[min(len(ordered) - 1, int(len(ordered) * .95))],
                                   "max_answer": max(answers), "over_limit": 0,
                                   "estimated_training_yuan_per_epoch": round(sum(lengths) * .006 / 1000, 2)}
            reports[name] = stats
        plan = training_plan(reports)
        plan["suggested_hyperparameters"]["max_length"] = max_length
        write_json(stage / "training_plan.json", plan)
        write_jsonl(stage / "provenance.jsonl", provenance)
        checksums = {p.name: file_hash(p) for p in stage.iterdir() if p.is_file()}
        manifest = {"pipeline": "bailian-word-segpos-sft-v1", "source": str(source), "seed": seed,
                    "source_manifest_sha256": file_hash(source / "manifest.json"), "generator_sha256": file_hash(Path(__file__)),
                    "official_template": {"path": str(template), "sha256": file_hash(template), "examples": len(demo), "used_in_training": 0},
                    "prompt_sha256": prompt_hash, "max_length": max_length, "tokenizer": token_info,
                    "datasets": reports, "output_sha256": checksums,
                    "checks": {"all_answers_exact_roundtrip": True, "unknown_tags": [], "test_exported": False,
                               "synthetic_research_examples_used": 0, "cloud_actions_executed": False},
                    "pilot_policy": "试验数据为正式train/dev的固定子集，不能作为新增独立评测集",
                    "pricing_basis": {"checked_at": "2026-10-10", "qwen3_8b_yuan_per_1000_tokens": .006,
                                      "url": "https://help.aliyun.com/zh/model-studio/model-training-and-deployment-billing",
                                      "excludes": "混合数据、失败重跑、部署与推理费用；最终以平台计费为准"}}
        write_json(stage / "manifest.json", manifest)
        stage.rename(output)
    return manifest


def evaluate_responses(gold_path, response_path, catalog_path, output, request_path):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"拒绝覆盖评测报告: {output}")
    gold, requests = load_file(gold_path), load_file(request_path)
    responses = load_file(response_path) if Path(response_path).stat().st_size else []
    catalog = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    allowed = set(catalog["observed_tags"])
    if len(requests) != len(gold):
        raise ValueError("请求文件必须与 gold 集合段落数一致；子集评测必须使用对应 gold")
    identifiers = [r["id"] for r in requests]
    identifier_set = set(identifiers)
    if len(identifier_set) != len(identifiers):
        raise ValueError("请求 ID 重复")
    prompt_hashes = {r.get("system_prompt_sha256") for r in requests}
    if len(prompt_hashes) != 1 or None in prompt_hashes:
        raise ValueError("请求提示词版本不一致或缺失")
    by_id = {}
    for row in responses:
        identifier = row.get("id")
        if not isinstance(identifier, str) or identifier not in identifier_set or identifier in by_id:
            raise ValueError("响应 ID 多余、重复或缺失")
        by_id[identifier] = row
    predictions, failures = [], []
    for item, request in zip(gold, requests):
        digest = text_hash(item["tokens"])
        if request["text_sha256"] != digest or json.loads(request["messages"][1]["content"]) != {"text": "".join(item["tokens"])}:
            raise ValueError("请求文本与 gold 不一致")
        actual_prompt = hashlib.sha256(request["messages"][0]["content"].encode("utf-8")).hexdigest()
        if actual_prompt != request["system_prompt_sha256"]:
            raise ValueError("请求提示词摘要不一致")
        response = by_id.get(request["id"])
        try:
            if response is None:
                raise ValueError("missing_response")
            if response.get("text_sha256") != digest:
                raise ValueError("响应必须携带对应原文 text_sha256")
            if response.get("finish_reason") != "stop":
                raise ValueError("非正常完成（截断、拒绝或未提供 finish_reason）")
            prediction = parse_response(response.get("content"), item["tokens"], allowed)
        except (ValueError, TypeError) as error:
            failures.append({"id": request["id"], "error": str(error)})
            prediction = {"tokens": item["tokens"], "span_list": []}
        predictions.append(prediction)
    report = evaluate_data(gold, predictions, allowed)
    report.update(valid_response_rate=(len(gold) - len(failures)) / len(gold), invalid_responses=len(failures),
                  failures=failures, invalid_policy="整段无预测，所有gold词计入FN，绝不跳过",
                  prompt_sha256=next(iter(prompt_hashes)),
                  source_sha256={str(p): file_hash(Path(p)) for p in (gold_path, response_path, catalog_path, request_path)})
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    exp = commands.add_parser("export")
    exp.add_argument("--source", default="data/prepared_v3")
    exp.add_argument("--output", default="data/bailian_v1")
    exp.add_argument("--template", default="docs/templates/Trainingdata.jsonl")
    exp.add_argument("--tokenizer", default=None)
    exp.add_argument("--max-length", type=int, default=8192)
    exp.add_argument("--pilot-size", type=int, default=2048)
    score = commands.add_parser("evaluate")
    score.add_argument("--gold", required=True)
    score.add_argument("--responses", required=True)
    score.add_argument("--requests", required=True)
    score.add_argument("--catalog", default="data/prepared_v3/label_catalog.json")
    score.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "export":
        result = export(args.source, args.output, args.template, args.tokenizer, args.max_length, args.pilot_size)
        print(json.dumps({k: v.get("tokens", {}) for k, v in result["datasets"].items()}, ensure_ascii=False))
    else:
        result = evaluate_responses(args.gold, args.responses, args.catalog, args.output, args.requests)
        print(json.dumps({k: result[k] for k in ("f1", "valid_response_rate", "invalid_responses")}, ensure_ascii=False))
