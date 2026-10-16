import argparse
import json
from collections import Counter
from pathlib import Path


def load_file(filename):
    data = []
    with open(filename, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"{filename}:{line_number}: 空行")
            data.append(json.loads(line))
    if not data:
        raise ValueError(f"{filename}: 数据为空")
    return data


def validate_item(item, allowed_tags=None, require_coverage=False):
    tokens = item.get("tokens")
    if not isinstance(tokens, list) or not tokens or any(
        not isinstance(token, str) or len(token) != 1 for token in tokens
    ):
        raise ValueError("tokens 必须是非空单字符列表")
    spans = item.get("span_list")
    if not isinstance(spans, list):
        raise ValueError("缺少 span_list 列表")
    result = set()
    previous_end = -1
    for span in spans:
        start, end, tag = span.get("start"), span.get("end"), span.get("type")
        if type(start) is not int or type(end) is not int or not 0 <= start <= end < len(tokens):
            raise ValueError(f"非法字符区间: {span}")
        if not isinstance(tag, str) or not tag or (allowed_tags is not None and tag not in allowed_tags):
            raise ValueError(f"未登记词性: {tag}")
        key = (start, end, tag)
        if key in result:
            raise ValueError(f"重复区间: {span}")
        if start <= previous_end:
            raise ValueError(f"重叠或错序区间: {span}")
        if require_coverage and start != previous_end + 1:
            raise ValueError(f"标注缺少字符覆盖: {span}")
        result.add(key)
        previous_end = end
    if require_coverage and previous_end != len(tokens) - 1:
        raise ValueError("标注未覆盖文本尾部")
    return result


def score(correct, predicted, gold):
    precision = correct / predicted if predicted else 0.0
    recall = correct / gold if gold else 0.0
    return {"precision": precision, "recall": recall,
            "f1": 2 * correct / (predicted + gold) if predicted + gold else 0.0,
            "pred_correct": correct, "total_pred": predicted, "total_golden": gold}


def evaluate_data(golden_data, pred_data, allowed_tags=None, long_threshold=254):
    if not golden_data or len(golden_data) != len(pred_data):
        raise ValueError("答案与预测的段落数不一致或为空")
    totals = Counter()
    categories = {}
    lengths = {"short": Counter(), "long": Counter()}
    for row, (gold, prediction) in enumerate(zip(golden_data, pred_data), 1):
        try:
            if gold.get("tokens") != prediction.get("tokens"):
                raise ValueError("预测文本或顺序与答案不一致")
            gs = validate_item(gold, allowed_tags, require_coverage=True)
            ps = validate_item(prediction, allowed_tags)
        except (ValueError, AttributeError) as error:
            raise ValueError(f"第 {row} 段: {error}") from error
        correct = gs & ps
        values = Counter(correct=len(correct), predicted=len(ps), gold=len(gs))
        totals.update(values)
        bucket = lengths["long" if len(gold["tokens"]) > long_threshold else "short"]
        bucket.update(values)
        bucket["paragraphs"] += 1
        gb = {(start, end) for start, end, _ in gs}
        pb = {(start, end) for start, end, _ in ps}
        totals.update(seg_correct=len(gb & pb), seg_predicted=len(pb), seg_gold=len(gb))
        totals["characters"] += len(gold["tokens"])
        covered = sum(end - start + 1 for start, end, _ in ps)
        totals["covered"] += covered
        totals["complete_paragraphs"] += covered == len(gold["tokens"])
        for key, spans in [("gold", gs), ("predicted", ps), ("correct", correct)]:
            for _, _, tag in spans:
                categories.setdefault(tag, Counter())[key] += 1
    result = score(totals["correct"], totals["predicted"], totals["gold"])
    result["segmentation"] = score(totals["seg_correct"], totals["seg_predicted"], totals["seg_gold"])
    result["per_tag"] = {tag: score(c["correct"], c["predicted"], c["gold"])
                         for tag, c in sorted(categories.items())}
    result["by_length"] = {name: {**score(c["correct"], c["predicted"], c["gold"]),
                                    "paragraphs": c["paragraphs"]} for name, c in lengths.items()}
    result.update(character_coverage=totals["covered"] / totals["characters"],
                  complete_paragraphs=totals["complete_paragraphs"], paragraphs=len(golden_data),
                  characters=totals["characters"], long_threshold=long_threshold)
    return result


def f1_metric(golden_file, pred_file, return_dict=True):
    result = evaluate_data(load_file(golden_file), load_file(pred_file))
    if return_dict:
        return result
    return result["precision"], result["recall"], result["f1"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="完整段落的分词与词性联合评测")
    parser.add_argument("prediction_file")
    parser.add_argument("golden_file")
    parser.add_argument("--catalog", default="data/label_catalog.json")
    parser.add_argument("--output", help="可选 JSON 评测报告")
    parser.add_argument("--long-threshold", type=int, default=254)
    args = parser.parse_args(argv)
    catalog = json.loads(Path(args.catalog).read_text(encoding="utf-8"))
    result = evaluate_data(load_file(args.golden_file), load_file(args.prediction_file),
                           set(catalog["registered_tags"]), args.long_threshold)
    report = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        path = Path(args.output)
        if path.exists():
            raise FileExistsError(path)
        path.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
