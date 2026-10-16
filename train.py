import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import torch
from transformers import (
    AutoConfig, AutoTokenizer, BertForTokenClassification,
    DataCollatorForTokenClassification, EarlyStoppingCallback, Trainer,
    TrainingArguments, set_seed,
)

from eval import evaluate_data, load_file, validate_item
from main import WindowDataset, check_label_map, merge_window_logits, predictions_from_logits


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_training_data(train, dev, mapping):
    allowed = {label[2:] for label in mapping.values() if label != "O"}
    seen_sets = []
    supported = set()
    for name, data in [("train", train), ("dev", dev)]:
        seen = set()
        for row, item in enumerate(data, 1):
            if "augmentation" in item:
                metadata = item["augmentation"]
                review = metadata.get("human_review", {}) if isinstance(metadata, dict) else {}
                if (not isinstance(review, dict) or review.get("status") != "approved"
                        or not review.get("reviewer") or not review.get("reviewed_at")):
                    raise ValueError(f"{name}:{row}: 补充数据尚未完成人工审核")
                if metadata.get("training_ready") is False:
                    raise ValueError(f"{name}:{row}: 词内语素审核材料尚未转换为当前单层训练格式")
            try:
                spans = validate_item(item, allowed, require_coverage=True)
            except ValueError as error:
                raise ValueError(f"{name}:{row}: {error}") from error
            text = "".join(item["tokens"])
            if text in seen:
                raise ValueError(f"{name}:{row}: 重复文本")
            seen.add(text)
            if name == "train":
                supported.update(tag for _, _, tag in spans)
        seen_sets.append(seen)
    if seen_sets[0] & seen_sets[1]:
        raise ValueError("训练与开发集存在文本泄漏")
    if supported != allowed:
        raise ValueError(f"标签表与训练类别不一致: {supported ^ allowed}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="独立开发集、完整段落评测的 BERT 基线训练")
    parser.add_argument("--train", default="data/prepared_v3/train.jsonline")
    parser.add_argument("--dev", default="data/prepared_v3/dev.jsonline")
    parser.add_argument("--catalog", default="data/prepared_v3/label_catalog.json")
    parser.add_argument("--base-model", default="bert-base-chinese")
    parser.add_argument("--output-dir", default="outputs/bert-run")
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decoder", choices=["argmax", "viterbi"], default="argmax")
    parser.add_argument("--precision", choices=["auto", "no", "bf16", "fp16"], default="auto")
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--resume", help="当前输出目录下的 checkpoint 路径")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-steps", type=int, default=-1, help="仅用于短运行验证")
    parser.add_argument("--eval-steps", type=int, default=0, help="0 为每轮评测；正数为更新步数间隔")
    parser.add_argument("--limit-train", type=int, default=0, help="仅用于短运行验证")
    parser.add_argument("--limit-dev", type=int, default=0, help="仅用于短运行验证")
    return parser.parse_args(argv)


def run_training(argv=None):
    args = parse_args(argv)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("当前完整段落评测只支持单进程；请直接运行 python train.py")
    if min(args.batch_size, args.eval_batch_size, args.gradient_accumulation, args.patience) < 1:
        raise ValueError("batch size、梯度累积和 patience 必须为正")
    if args.epochs <= 0 or args.learning_rate <= 0 or min(args.limit_train, args.limit_dev, args.eval_steps) < 0:
        raise ValueError("训练参数非法")
    if args.max_steps > 0 and args.eval_steps > args.max_steps:
        raise ValueError("评测间隔不能大于短运行的总更新步数")
    output = Path(args.output_dir).resolve()
    resume = Path(args.resume).resolve() if args.resume else None
    if resume:
        if resume.parent != output or not (resume / "trainer_state.json").is_file():
            raise ValueError("--resume 必须指向当前输出目录内的有效 checkpoint")
        state = json.loads((resume / "trainer_state.json").read_text(encoding="utf-8"))
        if ((args.max_steps > 0 and state["global_step"] >= args.max_steps)
                or (args.max_steps <= 0 and (state.get("epoch") or 0) >= args.epochs)):
            raise ValueError("该 checkpoint 已达到训练预算，无需恢复；新实验请使用新输出目录")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output}: 请使用新的输出目录或显式 --resume")
    precision = args.precision
    if precision == "auto":
        precision = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if torch.cuda.is_available() else "no"
    if precision != "no" and not torch.cuda.is_available():
        raise ValueError("混合精度训练需要 CUDA")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("当前 GPU 不支持 BF16")
    set_seed(args.seed)
    catalog = json.loads(Path(args.catalog).read_text(encoding="utf-8"))
    mapping = check_label_map(catalog["id_to_label"])
    label_to_id = {label: index for index, label in mapping.items()}
    if catalog["label_to_id"] != label_to_id:
        raise ValueError("标签表的两个映射不一致")
    train_data, dev_data = load_file(args.train), load_file(args.dev)
    validate_training_data(train_data, dev_data, mapping)
    if args.limit_train:
        train_data = train_data[:args.limit_train]
    if args.limit_dev:
        dev_data = dev_data[:args.limit_dev]
    manifest = {
        "pipeline": "bert-segpos-v1",
        "arguments": {key: value for key, value in vars(args).items()
                      if key not in ("resume", "output_dir", "local_files_only")},
        "resolved_precision": precision,
        "source_sha256": {name: sha256(path) for name, path in
                          [("train", args.train), ("dev", args.dev), ("catalog", args.catalog)]},
        "versions": {name: importlib.metadata.version(name)
                     for name in ("torch", "transformers", "accelerate", "numpy")},
        "smoke_run": bool(args.limit_train or args.limit_dev or args.max_steps > 0),
    }
    manifest_path = output / "run_manifest.json"
    base_config = AutoConfig.from_pretrained(args.base_model, local_files_only=args.local_files_only)
    if base_config.model_type != "bert" or any("TokenClassification" in arch for arch in (base_config.architectures or [])):
        raise ValueError("请从原始 BERT 预训练权重开始；已微调模型可能污染测试集")
    revision = getattr(base_config, "_commit_hash", None)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True,
                                              local_files_only=args.local_files_only, revision=revision)
    manifest["base_revision"] = revision
    manifest["tokenizer_sha256"] = hashlib.sha256(
        tokenizer.backend_tokenizer.to_str().encode("utf-8")).hexdigest()
    if resume and (not manifest_path.exists() or json.loads(manifest_path.read_text(encoding="utf-8")) != manifest):
        raise ValueError("恢复训练的配置、数据、分词器或依赖版本与原运行不一致")
    if args.max_len > base_config.max_position_embeddings:
        raise ValueError("窗口超过模型位置编码上限")
    print("缓存字符编码与窗口……", flush=True)
    train_dataset = WindowDataset(train_data, tokenizer, args.max_len, args.overlap, label_to_id)
    dev_dataset = WindowDataset(dev_data, tokenizer, args.max_len, args.overlap, label_to_id)
    print(f"train: {len(train_data)} 段 / {len(train_dataset)} 窗；"
          f"dev: {len(dev_data)} 段 / {len(dev_dataset)} 窗；"
          f"标签 {len(mapping)}；精度 {precision}", flush=True)
    model = BertForTokenClassification.from_pretrained(
        str(resume) if resume else args.base_model, local_files_only=args.local_files_only,
        num_labels=len(mapping), id2label=mapping, label2id=label_to_id,
        revision=None if resume else revision,
    )
    model.config.segpos = {"pipeline": "bert-segpos-v1", "max_len": args.max_len,
                          "overlap": args.overlap, "decoder": args.decoder,
                          "catalog_sha256": manifest["source_sha256"]["catalog"]}
    output.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def compute_metrics(evaluation):
        logits = evaluation.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        merged = merge_window_logits(dev_dataset, logits)
        reports = {}
        for decoder in ("argmax", "viterbi"):
            predictions, illegal = predictions_from_logits(dev_data, merged, mapping, decoder)
            report = evaluate_data(dev_data, predictions, set(catalog["observed_tags"]), args.max_len - 2)
            report["illegal_bio_transitions"] = illegal
            report["input_character_coverage"] = 1.0
            reports[decoder] = report
        (output / "last_dev_report.json").write_text(
            json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        selected = reports[args.decoder]
        return {"joint_f1": selected["f1"], "segmentation_f1": selected["segmentation"]["f1"],
                "character_coverage": selected["character_coverage"],
                "illegal_bio_transitions": selected["illegal_bio_transitions"],
                "argmax_joint_f1": reports["argmax"]["f1"],
                "viterbi_joint_f1": reports["viterbi"]["f1"]}

    strategy = "steps" if args.eval_steps else "epoch"
    training_args = TrainingArguments(
        output_dir=str(output), num_train_epochs=args.epochs, max_steps=args.max_steps,
        learning_rate=args.learning_rate, per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        weight_decay=0.01, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
        warmup_ratio=0.1, lr_scheduler_type="linear", max_grad_norm=1.0,
        bf16=precision == "bf16", fp16=precision == "fp16",
        gradient_checkpointing=args.gradient_checkpointing,
        eval_strategy=strategy, save_strategy=strategy,
        eval_steps=args.eval_steps or None, save_steps=args.eval_steps or 500,
        save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="joint_f1", greater_is_better=True,
        seed=args.seed, data_seed=args.seed, logging_steps=10,
        eval_accumulation_steps=8, dataloader_num_workers=0,
        report_to="none", optim="adamw_torch",
        restore_callback_states_from_checkpoint=True,
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=dev_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()
    best = output / "best_model"
    trainer.save_model(str(best))
    tokenizer.save_pretrained(best)
    trainer.save_metrics("dev", trainer.evaluate())
    (output / "best_dev_report.json").write_bytes((output / "last_dev_report.json").read_bytes())
    (best / "label_catalog.json").write_bytes(Path(args.catalog).read_bytes())
    print(f"最佳模型: {best}\n开发集报告: {output / 'best_dev_report.json'}", flush=True)
    return trainer


if __name__ == "__main__":
    run_training()
