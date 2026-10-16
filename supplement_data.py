"""按用户的词内构词标准生成双层待审材料，不导入现有单层 BIO 训练。"""

import argparse
from collections import Counter
import json
from pathlib import Path

from clean_data import file_hash, text_hash, validate_item


TARGETS = ("Eg", "Rg", "Mg", "Ug")
REVIEW_DATE = "2026-10-10"
SCHEME = "word_and_internal_morpheme_review_v2"
# 整句保持词级分词；内部目标另外记录，不将合成词切碎。
EXAMPLES = {
    "Eg": [
        ("eg-haqi", "不要/d 对/v 着/u 人/n 哈气/v 。/w", "哈气", [(0, "哈")]),
        ("eg-haqi", "他/r 向/p 冻僵/v 的/u 双手/n 哈气/v 。/w", "哈气", [(0, "哈")]),
        ("eg-haqi", "女孩/n 往/p 镜子/n 上/f 哈气/v ，/w 镜面/n 蒙上/v 一/m 层/q 薄/a 雾/n 。/w", "哈气", [(0, "哈")]),
        ("eg-haqi", "寒风/n 中/f ，/w 工人/n 不断/d 哈气/v 。/w", "哈气", [(0, "哈")]),
        ("eg-xixi-haha", "大家/r 嘻嘻哈哈/z 地/u 走进/v 教室/n 。/w", "嘻嘻哈哈", [(0, "嘻"), (1, "嘻"), (2, "哈"), (3, "哈")]),
        ("eg-xixi-haha", "听到/v 笑话/n ，/w 孩子们/n 嘻嘻哈哈/z 。/w", "嘻嘻哈哈", [(0, "嘻"), (1, "嘻"), (2, "哈"), (3, "哈")]),
        ("eg-xixi-haha", "他们/r 在/p 院子/n 里/f 嘻嘻哈哈/z ，/w 十分/d 开心/a 。/w", "嘻嘻哈哈", [(0, "嘻"), (1, "嘻"), (2, "哈"), (3, "哈")]),
        ("eg-xixi-haha", "老师/n 进门/v 后/f ，/w 大家/r 不再/d 嘻嘻哈哈/z 。/w", "嘻嘻哈哈", [(0, "嘻"), (1, "嘻"), (2, "哈"), (3, "哈")]),
    ],
    "Rg": [
        ("rg-self", "她/r 每天/r 自学/v 一/m 小时/n 英语/n 。/w", "自学", [(0, "自")]),
        ("rg-self", "受困/v 的/u 居民/n 开始/v 自救/v 。/w", "自救", [(0, "自")]),
        ("rg-self", "学生/n 先/d 自查/v 作业/n ，/w 再/d 交给/v 老师/n 。/w", "自查", [(0, "自")]),
        ("rg-self", "充分/a 的/u 准备/vn 让/v 她/r 更加/d 自信/a 。/w", "自信", [(0, "自")]),
        ("rg-plural-person", "你们/r 可以/v 从/p 侧门/n 进入/v 。/w", "你们", [(0, "你")]),
        ("rg-plural-person", "我们/r 一起/d 修理/v 这/r 张/q 桌子/n 。/w", "我们", [(0, "我")]),
        ("rg-plural-person", "他们/r 已经/d 找到/v 新/a 的/u 住处/n 。/w", "他们", [(0, "他")]),
        ("rg-reciprocal", "邻居/n 之间/f 彼此/r 照应/v 。/w", "彼此", [(0, "彼"), (1, "此")]),
    ],
    "Mg": [
        ("mg-number-compound", "师傅/n 骑/v 三轮车/n 运送/v 蔬菜/n 。/w", "三轮车", [(0, "三")]),
        ("mg-number-compound", "孩子/n 画/v 了/u 一/m 个/q 三角形/n 。/w", "三角形", [(0, "三")]),
        ("mg-number-compound", "这/r 座/q 城市/n 四季/n 分明/a 。/w", "四季", [(0, "四")]),
        ("mg-number-compound", "锻炼/vn 可以/v 增强/v 四肢/n 的/u 力量/n 。/w", "四肢", [(0, "四")]),
        ("mg-number-compound", "画家/n 正在/d 描绘/v 人物/n 的/u 五官/n 。/w", "五官", [(0, "五")]),
        ("mg-double", "他/r 举起/v 双手/n 。/w", "双手", [(0, "双")]),
        ("mg-double", "双方/n 已经/d 达成/v 协议/n 。/w", "双方", [(0, "双")]),
        ("mg-second-hand", "她/r 买/v 了/u 一/m 辆/q 二手/b 自行车/n 。/w", "二手", [(0, "二")]),
    ],
    "Ug": [
        ("ug-shide", "孩子/n 像/v 小鸟/n 似的/u 跑进/v 院子/n 。/w", "似的", [(1, "的")]),
        ("ug-shide", "湖面/n 像/v 镜子/n 似的/u 平静/a 。/w", "似的", [(1, "的")]),
        ("ug-shide", "她/r 像/v 没/d 听见/v 似的/u ，/w 继续/v 低头/v 看/v 书/n 。/w", "似的", [(1, "的")]),
        ("ug-dehua", "你/r 同意/v 的话/u ，/w 我们/r 明天/t 就/d 出发/v 。/w", "的话", [(0, "的")]),
        ("ug-dehua", "天气/n 晴朗/a 的话/u ，/w 孩子们/n 可以/v 去/v 公园/n 。/w", "的话", [(0, "的")]),
        ("ug-dehua", "时间/n 充足/a 的话/u ，/w 请/v 再/d 检查/v 一/m 遍/q 。/w", "的话", [(0, "的")]),
        ("ug-laizhe", "你/r 刚才/t 说/v 什么/r 来着/u ？/w", "来着", [(1, "着")]),
        ("ug-laizhe", "我/r 昨天/t 在/p 这里/r 等/v 你/r 来着/u 。/w", "来着", [(1, "着")]),
    ],
}

REVIEW_NOTES = {
    "Eg": "按用户给定的哈气、嘻嘻哈哈口径构造；目标在词内。哈气中哈的呼气义与叹词义可能不同，Eg 属项目候选判断。嘻嘻哈哈的整词 z 与叠音语素粒度须人工确认；重复位置不代表不同的词根。",
    "Rg": "自保留自己的指代义；你/我/他进入复数代词，彼/此进入彼此。排除独立汝、尔、斯、甚，不按同字自动改标。",
    "Mg": "数字进入三轮车、三角形、四季、四肢、五官，双进入双手/双方，二进入二手；不使用独立枚举的甲乙丙丁。词汇化或序数义仍须人工核对。",
    "Ug": "的进入复合助词似的/的话，着进入来着；完整助词可独立于前词，内部的/着不是另一个句法词。排除句末独立矣和仅附着在词外的普通的、了、着。",
}

REFERENCES = [
    {"title": "sunyeu 的项目标注规则", "scope": "原词进入合成词内部；用户给定哈气、嘻嘻哈哈；独立喏应标e"},
    {"title": "香港教育大学：轻声词和轻声词教学，第3页", "url": "https://home.eduhk.hk/~cmsi/tm_item6_1/item7_1_7.pdf",
     "scope": "支持似的为比况助词、来着为时态助词；不提供Eg/Rg/Mg/Ug内部标签的直接依据"},
]


def validate_internal_targets(item):
    validate_item(item)
    targets = item.get("target_spans")
    if not isinstance(targets, list) or not targets:
        raise ValueError("缺少词内目标语素")
    previous_end = -1
    for target in targets:
        start, end, index = target["start"], target["end"], target["word_span_index_zero_based"]
        if any(type(value) is not int for value in (start, end, index)) or not 0 <= index < len(item["span_list"]):
            raise ValueError("非法目标坐标或所属词编号")
        word = item["span_list"][index]
        if not word["start"] <= start <= end <= word["end"] or (start == word["start"] and end == word["end"]):
            raise ValueError("目标必须是完整词内的真子区间")
        if start <= previous_end:
            raise ValueError("内部目标重叠或错序")
        previous_end = end
        if target["type"] not in TARGETS or target["unit"] != "".join(item["tokens"][start:end + 1]):
            raise ValueError("目标语素标签或词面不一致")
        if target["word_text"] != "".join(item["tokens"][word["start"]:word["end"] + 1]) or target["word_type"] != word["type"]:
            raise ValueError("所属完整词信息不一致")


def make_examples():
    result = []
    for tag, examples in EXAMPLES.items():
        for number, (group, annotated, host, components) in enumerate(examples, 1):
            tokens, spans, targets = [], [], []
            for part in annotated.split():
                word, pos = part.rsplit("/", 1)
                start = len(tokens)
                tokens.extend(word)
                span = {"start": start, "end": len(tokens) - 1, "type": pos}
                spans.append(span)
                if word == host:
                    for offset, unit in components:
                        targets.append({"start": start + offset, "end": start + offset + len(unit) - 1,
                                        "type": tag, "unit": unit, "word_span_index_zero_based": len(spans) - 1,
                                        "word_text": word, "word_type": pos})
            if len(targets) != len(components):
                raise ValueError("目标完整词缺失或重复")
            item = {"tokens": tokens, "span_list": spans, "target_spans": targets,
                    "augmentation": {"id": f"v2-{tag.lower()}-{number:03d}", "origin": "synthetic_agent_authored",
                                     "annotation_scheme": SCHEME, "training_ready": False,
                                     "target_tag": tag, "review_group": group, "assigned_split": None,
                                     "self_review": {"status": "completed", "reviewed_at": REVIEW_DATE, "note": REVIEW_NOTES[tag]},
                                     "human_review": {"status": "pending", "reviewer": None, "reviewed_at": None}}}
            validate_internal_targets(item)
            result.append(item)
    return result


def export_supplement(data_dir, output_dir):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"{output_dir}: 请换新目录，避免覆盖人工审核")
    examples = make_examples()
    catalog = json.loads((data_dir / "label_catalog.json").read_text(encoding="utf-8"))
    allowed = set(catalog["observed_tags"])
    source_hashes = set()
    with (data_dir / "cleaning_audit.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record["status"] in ("train", "dev", "test"):
                source_hashes.add(record["clean_text_sha256"])
    seen, counts = set(), Counter()
    for item in examples:
        validate_internal_targets(item)
        digest = text_hash(item["tokens"])
        if digest in seen or digest in source_hashes:
            raise ValueError("补充数据含重复文本或与已有集合重复")
        seen.add(digest)
        if any(span["type"] not in allowed for span in item["span_list"] + item["target_spans"]):
            raise ValueError("补充数据含标签表外的类别")
        counts.update(span["type"] for span in item["target_spans"])
    manifest = {
        "created_at": REVIEW_DATE, "annotation_scheme": SCHEME, "sentences": len(examples),
        "sentences_by_tag": {tag: len(EXAMPLES[tag]) for tag in TARGETS}, "target_occurrences": dict(counts),
        "origin": "synthetic_agent_authored", "self_review": "completed", "human_review": "pending",
        "training_ready": False, "merged_into_training": False,
        "checks": {"complete_word_coverage": True, "targets_strictly_inside_words": True,
                   "all_tags_in_catalog": True, "unique_sentences": True, "disjoint_from_existing_split_texts": True},
        "data_sha256": {name: file_hash(data_dir / name) for name in ("train.jsonline", "dev.jsonline", "cleaning_audit.jsonl", "label_catalog.json")},
        "generator_sha256": file_hash(Path(__file__)), "review_notes": REVIEW_NOTES, "references": REFERENCES,
        "source_correction": {"cleaned_file": "data/train.jsonline", "cleaned_line_one_based": 10651,
                              "original_file": "PKU-SEGPOS/train.jsonline", "original_line_one_based": 12065,
                              "start": 75, "end": 75, "unit": "喏", "before": "Eg", "after": "e",
                              "confirmed_by_user": True, "applied_to_training": False,
                              "note": "删除唯一Eg标注会改变现有训练类别覆盖；本轮只重做待审材料，正式数据与映射留待统一版本更新"},
        "self_review_changes": ["弃用v1的全部48句，独立喏、汝/尔/斯/甚、矣不再充当语素正例",
                                "逐句核查自然性、整词词性和词内目标，保留完整词边界",
                                "将词内真子区间作为必要条件，但不把位置条件冒充语义分类的充分条件",
                                "对照原训练/开发词面标注使用小时/n，简化暖手、赞成等与目标无关的边界歧义",
                                "排除词外普通助词；将字义、叠音粒度和完整词词性歧义写入审核提示"],
        "limitations": ["内部类别按用户构词口径拟定，属于待审候选，不是源语料或词典金标准",
                        "当前BERT单层BIO不能同时预测整词与内部语素；审核通过后也不能直接导入现有训练",
                        "同一review_group不能跨训练与验证；本轮没有分配集合"],
    }
    review = ["# 词内语素补充句人工审核（v2）", "",
              "共32句，每类8句。已做一次生成者自审，全部等待sunyeu审核，未并入任何集合。", "",
              "本轮按你的项目规则：原词进入合成词内部才标语素。旧48句不符合这条规则，已弃用。",
              "独立的喏应标e；源训练第10651行（原下载第12065行）的改标已记录为用户确认，尚未改写正式数据。", "",
              "每句保留整词词性，再用⟦字/语素标签⟧高亮词内候选；不把完整词拆成多个独立词。",
              "JSONL的span_list仅是完整词；target_spans是另一层内部语素。原BERT单层BIO不能同时表达这两层。",
              "即使人工通过，这份JSONL也不能直接导入现有训练：需先确定任务层次和训练转换方案。", "",
              "请核查：句子自然性、词边界、整个词的词性、内部成分意义和语素类型。位置在词内不意味着类别一定正确。",
              "Eg按你提供的哈气/嘻嘻哈哈造句；哈气中的哈字义、嘻嘻哈哈的整词z及叠音粒度仍请确认。", "",
              "Ug完整词参考[香港教育大学《轻声词和轻声词教学》第3页](https://home.eduhk.hk/~cmsi/tm_item6_1/item7_1_7.pdf)：似的为比况助词、来着为时态助词。内部Ug是本项目候选推断，资料未直接给出此类内部标签。", "",
              "可直接回复编号与修改意见；JSONL人工审核状态仍为pending。0基坐标、end包含在区间内。",
              "同一review_group的模板句将来必须留在同一集合。", ""]
    for tag in TARGETS:
        review.extend([f"## {tag}：{catalog['registered_tags'][tag]}", "", REVIEW_NOTES[tag], ""])
        for item in examples:
            if item["augmentation"]["target_tag"] != tag:
                continue
            target_starts = {span["start"]: span for span in item["target_spans"]}
            text, cursor, display = "".join(item["tokens"]), 0, []
            while cursor < len(text):
                target = target_starts.get(cursor)
                if target:
                    display.append(f"⟦{target['unit']}/{target['type']}⟧")
                    cursor = target["end"] + 1
                else:
                    display.append(text[cursor]); cursor += 1
            annotated = " ".join(f"{text[span['start']:span['end'] + 1]}({span['type']})" for span in item["span_list"])
            meta = item["augmentation"]
            review.extend([f"### {meta['id']}", "", "".join(display), "", "- 整词标注：" + annotated])
            for target in item["target_spans"]:
                review.append(f"- 内部候选：{target['unit']}/{target['type']} [{target['start']}:{target['end']}]；所属完整词：{target['word_text']}({target['word_type']})")
            review.extend([f"- 分组：`{meta['review_group']}`", "- [ ] 人工审核通过", "- 修改意见：", ""])
    output_dir.mkdir(parents=True)
    with (output_dir / "rare_morphemes.jsonline").open("x", encoding="utf-8", newline="\n") as stream:
        for item in examples:
            stream.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest["jsonline_sha256"] = file_hash(output_dir / "rare_morphemes.jsonline")
    (output_dir / "review.md").write_text("\n".join(review), encoding="utf-8", newline="\n")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data/augmentation_pending_v2"))
    args = parser.parse_args()
    report = export_supplement(args.data, args.output)
    print(json.dumps({key: report[key] for key in ("sentences", "sentences_by_tag", "target_occurrences")}, ensure_ascii=False, indent=2))
