import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertForTokenClassification, DataCollatorForTokenClassification

from eval import load_file, validate_item

load_jsonline = load_file


def check_label_map(id_to_label):
    mapping = {int(key): value for key, value in id_to_label.items()}
    if sorted(mapping) != list(range(len(mapping))) or len(set(mapping.values())) != len(mapping):
        raise ValueError("模型标签编号必须连续且唯一")
    labels = set(mapping.values())
    if "O" not in labels or any(
        label != "O" and (not label.startswith(("B-", "I-")) or not label[2:])
        for label in labels
    ):
        raise ValueError("模型缺少实际 BIO 标签映射；不能用 LABEL_n 或新数据解释旧模型")
    if any("I-" + label[2:] not in labels for label in labels if label.startswith("B-")) or any(
        "B-" + label[2:] not in labels for label in labels if label.startswith("I-")
    ):
        raise ValueError("BIO 标签缺少配对")
    return mapping


class WindowDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=256, overlap=64, label_to_id=None):
        if not data:
            raise ValueError("输入数据为空")
        if not tokenizer.is_fast or tokenizer.padding_side != "right":
            raise ValueError("需要右侧填充的 fast tokenizer")
        if not 3 <= max_len <= tokenizer.model_max_length or not 0 <= overlap < max_len - 2:
            raise ValueError("窗口长度或重叠长度非法")
        if tokenizer.num_special_tokens_to_add() != 2 or tokenizer.build_inputs_with_special_tokens([7]) != [
            tokenizer.cls_token_id, 7, tokenizer.sep_token_id
        ]:
            raise ValueError("本项目窗口编码需要 BERT 的 CLS/SEP 格式")
        self.data = data
        self.features = []
        self.positions = []
        self.selected = []
        self.paragraph_ids = []
        self.window_counts = []
        self.max_len = max_len
        self.overlap = overlap
        allowed = {label[2:] for label in label_to_id if label != "O"} if label_to_id else None
        for paragraph_id, item in enumerate(data):
            tokens = item.get("tokens")
            if not isinstance(tokens, list) or not tokens or any(
                not isinstance(token, str) or len(token) != 1 for token in tokens
            ):
                raise ValueError(f"第 {paragraph_id + 1} 段 tokens 不是非空字符列表")
            gold_labels = None
            if label_to_id is not None:
                validate_item(item, allowed, require_coverage=True)
                gold_labels = [label_to_id["O"]] * len(tokens)
                for span in item["span_list"]:
                    for char in range(span["start"], span["end"] + 1):
                        prefix = "B-" if char == span["start"] else "I-"
                        gold_labels[char] = label_to_id[prefix + span["type"]]
            encoding = tokenizer(tokens, is_split_into_words=True, add_special_tokens=False,
                                 truncation=False, return_attention_mask=False,
                                 return_token_type_ids=False, verbose=False)
            word_ids = encoding.word_ids()
            groups = [[] for _ in tokens]
            for token_id, char in zip(encoding["input_ids"], word_ids):
                if char is None or not 0 <= char < len(tokens):
                    raise ValueError("分词器返回非法字符对齐")
                groups[char].append(token_id)
            if any(not group for group in groups):
                raise ValueError(f"第 {paragraph_id + 1} 段有字符被 tokenizer 丢弃")
            if any(len(group) > max_len - 2 for group in groups):
                raise ValueError("单字符的子词数超过窗口容量")
            prefix_lengths = np.cumsum([0] + [len(group) for group in groups]).tolist()
            first_window = len(self.features)
            owner = [-1] * len(tokens)
            best_distance = [-1] * len(tokens)
            start = 0
            while start < len(tokens):
                end = start
                while end < len(tokens) and prefix_lengths[end + 1] - prefix_lengths[start] <= max_len - 2:
                    end += 1
                ids = [tokenizer.cls_token_id]
                positions = [-1]
                for char in range(start, end):
                    positions.extend([char] + [-1] * (len(groups[char]) - 1))
                    ids.extend(groups[char])
                ids.append(tokenizer.sep_token_id)
                positions.append(-1)
                feature = {"input_ids": ids, "attention_mask": [1] * len(ids)}
                window_id = len(self.features)
                self.features.append(feature)
                self.positions.append(positions)
                self.paragraph_ids.append(paragraph_id)
                self.selected.append([])
                for char in range(start, end):
                    distance = min(prefix_lengths[char] - prefix_lengths[start],
                                   prefix_lengths[end] - prefix_lengths[char + 1])
                    if distance > best_distance[char]:
                        best_distance[char] = distance
                        owner[char] = window_id
                if end == len(tokens):
                    break
                next_start = end
                while next_start > start + 1 and prefix_lengths[end] - prefix_lengths[next_start - 1] <= overlap:
                    next_start -= 1
                start = next_start
            for window_id in range(first_window, len(self.features)):
                positions = self.positions[window_id]
                selected = [(pos, char) for pos, char in enumerate(positions)
                            if char >= 0 and owner[char] == window_id]
                if not selected:
                    raise ValueError("窗口没有有效监督字符，请降低 overlap")
                self.selected[window_id] = selected
                if gold_labels is not None:
                    labels = [-100] * len(positions)
                    for pos, char in selected:
                        labels[pos] = gold_labels[char]
                    self.features[window_id]["labels"] = labels
            self.window_counts.append(len(self.features) - first_window)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]


def merge_window_logits(dataset, logits):
    if len(logits) != len(dataset):
        raise ValueError("窗口预测数量不一致")
    width = logits[0].shape[-1]
    merged = [np.empty((len(item["tokens"]), width), dtype=np.float32) for item in dataset.data]
    covered = [np.zeros(len(item["tokens"]), dtype=np.int8) for item in dataset.data]
    for window_id, window_logits in enumerate(logits):
        paragraph = dataset.paragraph_ids[window_id]
        selected = dataset.selected[window_id]
        positions, chars = zip(*selected)
        merged[paragraph][list(chars)] = window_logits[list(positions)]
        covered[paragraph][list(chars)] += 1
    if any(np.any(count != 1) for count in covered):
        raise ValueError("滑窗未恰好覆盖每个字符一次")
    if any(not np.isfinite(array).all() for array in merged):
        raise ValueError("模型输出含非有限值")
    return merged


def illegal_bio_count(labels):
    errors = 0
    previous = "O"
    for label in labels:
        if label.startswith("I-") and previous not in ("B-" + label[2:], label):
            errors += 1
        previous = label
    return errors


def decode_logits(logits, id_to_label, decoder="argmax"):
    mapping = check_label_map(id_to_label)
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim != 2 or logits.shape[1] != len(mapping) or not np.isfinite(logits).all():
        raise ValueError("logits 形状或数值非法")
    if not len(logits):
        return []
    if decoder == "argmax":
        return [mapping[int(index)] for index in logits.argmax(axis=-1)]
    if decoder != "viterbi":
        raise ValueError(f"未知解码器: {decoder}")
    inverse = {label: index for index, label in mapping.items()}
    inside = np.array([index for index, label in mapping.items() if label.startswith("I-")], dtype=np.int32)
    begins = np.array([inverse["B-" + mapping[int(index)][2:]] for index in inside], dtype=np.int32)
    scores = logits[0].copy()
    scores[inside] = -np.inf
    pointers = np.zeros(logits.shape, dtype=np.int32)
    for char in range(1, len(logits)):
        best = int(scores.argmax())
        predecessors = np.full(len(mapping), best, dtype=np.int32)
        # I-X 只允许接在 B-X / I-X 之后，其余标签允许所有前驱。
        predecessors[inside] = np.where(scores[begins] >= scores[inside], begins, inside)
        pointers[char] = predecessors
        scores = scores[predecessors] + logits[char]
        scores -= scores.max()
    path = [int(scores.argmax())]
    for char in range(len(logits) - 1, 0, -1):
        path.append(int(pointers[char, path[-1]]))
    return [mapping[index] for index in reversed(path)]


def bio_to_spans(labels):
    spans = []
    start = None
    current = None
    for char, label in enumerate(list(labels) + ["O"]):
        if label == "O":
            if start is not None:
                spans.append({"start": start, "end": char - 1, "type": current})
            start, current = None, None
            continue
        prefix, tag = label.split("-", 1)
        if prefix not in ("B", "I") or not tag:
            raise ValueError(f"非法 BIO 标签: {label}")
        # argmax 中非法 I 作为新词起点处理，并另外报告原始非法转移数。
        if prefix == "B" or current != tag:
            if start is not None:
                spans.append({"start": start, "end": char - 1, "type": current})
            start, current = char, tag
    return spans


def predictions_from_logits(data, logits, mapping, decoder):
    if len(data) != len(logits):
        raise ValueError("段落预测数量不一致")
    predictions = []
    illegal = 0
    for item, array in zip(data, logits):
        if len(item["tokens"]) != len(array):
            raise ValueError("字符预测长度不一致")
        labels = decode_logits(array, mapping, decoder)
        illegal += illegal_bio_count(labels)
        predictions.append({**item, "span_list": bio_to_spans(labels)})
    if len(predictions) != len(data):
        raise ValueError("段落预测数量不一致")
    return predictions, illegal


def predict_logits(dataset, model, tokenizer, device, batch_size=16, precision="no"):
    if batch_size <= 0 or precision not in ("no", "bf16", "fp16"):
        raise ValueError("batch size 或精度非法")
    if precision != "no" and device.type != "cuda":
        raise ValueError("混合精度预测需要 CUDA")
    collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    merged = [np.empty((len(item["tokens"]), model.config.num_labels), dtype=np.float32)
              for item in dataset.data]
    coverage = [np.zeros(len(item["tokens"]), dtype=np.int8) for item in dataset.data]
    model.eval()
    offset = 0
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    with torch.inference_mode():
        for batch in loader:
            batch.pop("labels", None)
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=precision != "no"):
                outputs = model(**batch).logits.float().cpu().numpy()
            for row, window_logits in enumerate(outputs):
                window = offset + row
                paragraph = dataset.paragraph_ids[window]
                positions, chars = zip(*dataset.selected[window])
                merged[paragraph][list(chars)] = window_logits[list(positions)]
                coverage[paragraph][list(chars)] += 1
            offset += len(outputs)
    if offset != len(dataset) or any(np.any(count != 1) for count in coverage):
        raise ValueError("窗口预测未完整覆盖原文")
    if any(not np.isfinite(array).all() for array in merged):
        raise ValueError("模型输出含非有限值")
    return merged


def format_sentence_prediction(sentence, labels):
    if not isinstance(sentence, str) or not sentence:
        raise ValueError("请输入非空句子")
    if len(sentence) != len(labels):
        raise ValueError("字符预测长度不一致")
    spans = bio_to_spans(labels)
    characters = [{"character": char, "type": label[2:] if label != "O" else "O",
                   "bio": label} for char, label in zip(sentence, labels)]
    words = []
    cursor = 0
    for span in spans:
        while cursor < span["start"]:
            words.append({"text": sentence[cursor], "start": cursor, "end": cursor, "type": "O"})
            cursor += 1
        words.append({"text": sentence[span["start"]:span["end"] + 1], **span})
        cursor = span["end"] + 1
    while cursor < len(sentence):
        words.append({"text": sentence[cursor], "start": cursor, "end": cursor, "type": "O"})
        cursor += 1
    return {
        "text": sentence, "characters": characters, "words": words, "span_list": spans,
        "character_output": " ".join(f"{item['character']}({item['type']})" for item in characters),
        "word_output": " ".join(f"{item['text']}({item['type']})" for item in words),
        "bio_output": " ".join(f"{item['character']}({item['bio']})" for item in characters),
        "illegal_bio_transitions": illegal_bio_count(labels),
        "character_coverage": sum(span["end"] - span["start"] + 1 for span in spans) / len(sentence),
    }


def predict_sentence(sentence, model, tokenizer, device, decoder=None, batch_size=16, precision="no"):
    """复用批量预测的滑窗、字符对齐和解码，模型由调用方加载一次。"""
    if not isinstance(sentence, str) or not sentence:
        raise ValueError("请输入非空句子")
    mapping = check_label_map(model.config.id2label)
    options = getattr(model.config, "segpos", {})
    max_len, overlap = options.get("max_len", 256), options.get("overlap", 64)
    decoder = decoder or options.get("decoder", "argmax")
    if decoder not in ("argmax", "viterbi"):
        raise ValueError(f"未知解码器: {decoder}")
    if max_len > model.config.max_position_embeddings:
        raise ValueError("窗口长度超过模型位置编码上限")
    dataset = WindowDataset([{"tokens": list(sentence)}], tokenizer, max_len, overlap)
    logits = predict_logits(dataset, model, tokenizer, device, batch_size, precision)[0]
    result = format_sentence_prediction(sentence, decode_logits(logits, mapping, decoder))
    result.update({"decoder": decoder, "windows": len(dataset), "input_character_coverage": 1.0})
    return result


def predict_text_file(input_path, output_dir, model, tokenizer, device, *,
                      encoding="utf-8-sig", decoder=None, batch_size=16, precision="no"):
    """逐行预测普通文本，保留空白和原行坐标，失败时移除本次未完成的输出。"""
    from clean_data import TAG_NAMES

    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    mapping = check_label_map(model.config.id2label)
    unknown = {label[2:] for label in mapping.values() if label != "O"} - TAG_NAMES.keys()
    if unknown:
        raise ValueError(f"模型包含没有中文名称的词性: {sorted(unknown)}")
    destination = Path(output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()  # 输出目录必须新建，不能覆盖旧结果或输入文件。
    standard_path = destination / "predictions.jsonline"
    readable_path = destination / "words_zh.txt"
    counts = {"lines": 0, "records": 0, "characters": 0,
              "predicted_characters": 0, "tagged_characters": 0, "words": 0}
    try:
        with source.open(encoding=encoding, newline="") as original, \
                standard_path.open("x", encoding="utf-8", newline="\n") as standard, \
                readable_path.open("x", encoding="utf-8", newline="") as readable:
            for line_number, raw_line in enumerate(original, 1):
                counts["lines"] += 1
                text = raw_line.rstrip("\r\n")
                line_ending = raw_line[len(text):]
                spans, rendered, cursor = [], [], 0
                # 不将空白送入会丢弃空白的 BERT tokenizer，也不拼接两侧的词。
                for match in re.finditer(r"\S+", text):
                    try:
                        result = predict_sentence(
                            match.group(), model, tokenizer, device, decoder=decoder,
                            batch_size=batch_size, precision=precision,
                        )
                    except ValueError as error:
                        raise ValueError(f"{source}:第 {line_number} 行: {error}") from error
                    spans.extend({**span, "start": span["start"] + match.start(),
                                  "end": span["end"] + match.start()}
                                 for span in result["span_list"])
                    rendered.append(text[cursor:match.start()])
                    rendered.append(" ".join(
                        f"{word['text']}({TAG_NAMES[word['type']] if word['type'] != 'O' else '未标注'})"
                        for word in result["words"]
                    ))
                    cursor = match.end()
                    counts["predicted_characters"] += len(match.group())
                    counts["tagged_characters"] += sum(
                        span["end"] - span["start"] + 1 for span in result["span_list"]
                    )
                    counts["words"] += sum(word["type"] != "O" for word in result["words"])
                rendered.append(text[cursor:])
                readable.write("".join(rendered) + line_ending)
                # 真正的空行不生成空 tokens，靠 line_number 保留 JSONL 与源文件的对应关系。
                if text:
                    standard.write(json.dumps({"tokens": list(text), "span_list": spans,
                                               "line_number": line_number}, ensure_ascii=False) + "\n")
                    counts["records"] += 1
                    counts["characters"] += len(text)
        if not counts["predicted_characters"]:
            raise ValueError("输入文件不包含可预测的非空白文本")
    except BaseException:
        standard_path.unlink(missing_ok=True)
        readable_path.unlink(missing_ok=True)
        destination.rmdir()
        raise
    return {"input": str(source.resolve()), "standard_output": str(standard_path.resolve()),
            "readable_output": str(readable_path.resolve()), "encoding": encoding, **counts}


def write_predictions(data, labels_by_item, output_path):
    if len(data) != len(labels_by_item):
        raise ValueError("预测数量不一致")
    with open(output_path, "x", encoding="utf-8") as stream:
        for item, labels in zip(data, labels_by_item):
            if len(item["tokens"]) != len(labels):
                raise ValueError("预测未覆盖全文")
            stream.write(json.dumps({**item, "span_list": bio_to_spans(labels)}, ensure_ascii=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="完整文本的 BERT 分词和词性预测")
    parser.add_argument("--input", default="PKU-SEGPOS/test_public.jsonline")
    parser.add_argument("--output", default="outputs/predictions/PKUSEGPOS.jsonline")
    parser.add_argument("--model", default="outputs/bert-prepared-v3-lr3e5-seed42/best_model")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, help="默认读取模型保存的窗口设置")
    parser.add_argument("--overlap", type=int)
    parser.add_argument("--decoder", choices=["argmax", "viterbi"])
    parser.add_argument("--precision", choices=["no", "bf16", "fp16"], default="no")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only, use_fast=True)
    model = BertForTokenClassification.from_pretrained(args.model, local_files_only=args.local_files_only)
    mapping = check_label_map(model.config.id2label)
    options = getattr(model.config, "segpos", {})
    max_len = args.max_len if args.max_len is not None else options.get("max_len", 256)
    overlap = args.overlap if args.overlap is not None else options.get("overlap", 64)
    decoder = args.decoder or options.get("decoder", "argmax")
    if max_len > model.config.max_position_embeddings:
        raise ValueError("窗口长度超过模型位置编码上限")
    data = load_jsonline(args.input)
    started = time.perf_counter()
    dataset = WindowDataset(data, tokenizer, max_len, overlap)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logits = predict_logits(dataset, model, tokenizer, device, args.batch_size, args.precision)
    predictions, illegal = predictions_from_logits(data, logits, mapping, decoder)
    elapsed = time.perf_counter() - started
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "x", encoding="utf-8") as stream:
        for item in predictions:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps({"output": args.output, "paragraphs": len(data), "windows": len(dataset),
                      "decoder": decoder, "illegal_bio_transitions": illegal,
                      "input_character_coverage": 1.0, "seconds": elapsed,
                      "characters_per_second": sum(len(item["tokens"]) for item in data) / elapsed},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
