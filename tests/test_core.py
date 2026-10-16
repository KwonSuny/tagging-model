"""分词、词性标注与数据处理的核心回归测试。"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from clean_data import clean_item, deduplicate, normalize_tag, validate_item as validate_clean_item, validate_splits
import itertools
import numpy as np
from transformers import BertTokenizerFast
from eval import evaluate_data, validate_item
from main import predict_text_file, WindowDataset, decode_logits, format_sentence_prediction, illegal_bio_count, merge_window_logits, predictions_from_logits
from train import validate_training_data
from bailian_data import make_messages, parse_response

def sample(text, tag='n'):
    return {'tokens': list(text), 'span_list': [{'start': 0, 'end': len(text) - 1, 'type': tag}]}

def paragraph(text='蜘蛛结网。', tags=('n', 'v', 'n', 'w')):
    return {'tokens': list(text), 'span_list': [{'start': 0, 'end': 1, 'type': tags[0]}, {'start': 2, 'end': 2, 'type': tags[1]}, {'start': 3, 'end': 3, 'type': tags[2]}, {'start': 4, 'end': 4, 'type': tags[3]}]}

class CleaningTests(unittest.TestCase):

    def test_phrase_suffix_preserves_morpheme_category(self):
        self.assertEqual(normalize_tag('Ng]nt', '院'), ('Ng', ['nt']))
        self.assertEqual(normalize_tag('Ng]ns', '村'), ('Ng', ['ns']))
        self.assertEqual(normalize_tag('ng', '村'), ('Ng', []))
        self.assertEqual(normalize_tag('Bg', '翠'), ('Bg', []))
        self.assertEqual(normalize_tag('nx', 'WTO'), ('nx', []))

    def test_unknown_labels_are_not_guessed_from_prefix(self):
        for tag in ['nGep', 'nMQNp', 'n2@', 'vt！2', 'Ng]unknown', 'Ng]']:
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                normalize_tag(tag, '字')
        self.assertEqual(normalize_tag('n2', '小麦'), ('n', []))
        self.assertEqual(normalize_tag('〓', '〓'), ('w', []))
        with self.assertRaises(ValueError):
            normalize_tag('〓', '字')

    def test_nested_phrase_removal_remaps_all_offsets(self):
        item = {'tokens': list('[中国[人民银行。'), 'span_list': [{'start': 0, 'end': 2, 'type': 'ns'}, {'start': 3, 'end': 5, 'type': 'n'}, {'start': 6, 'end': 7, 'type': 'n]nt]ns'}, {'start': 8, 'end': 8, 'type': 'w'}]}
        original = json.dumps(item)
        record = clean_item(item)
        self.assertEqual(record.status, 'candidate')
        self.assertEqual(''.join(record.item['tokens']), '中国人民银行。')
        self.assertEqual(record.item['span_list'], [{'start': 0, 'end': 1, 'type': 'ns'}, {'start': 2, 'end': 3, 'type': 'n'}, {'start': 4, 'end': 5, 'type': 'n'}, {'start': 6, 'end': 6, 'type': 'w'}])
        self.assertEqual(json.dumps(item), original)

    def test_conflicting_duplicate_group_is_removed(self):
        records = [clean_item(sample('中国')), clean_item(sample('中国', 'ns')), clean_item(sample('北京', 'ns'))]
        self.assertEqual(len(deduplicate(records)), 1)
        self.assertTrue(all((record.status == 'rejected' for record in records[:2])))

    def test_validators_detect_leakage_and_unknown_labels(self):
        record = clean_item(sample('中国'))
        with self.assertRaises(ValueError):
            validate_splits({'train': [record], 'dev': [], 'test': [record]})
        with self.assertRaises(ValueError):
            validate_clean_item(sample('字', 'unknown'))
        with self.assertRaises(ValueError):
            validate_clean_item(sample('字', []))

class PipelineTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        vocab = Path(cls.temporary.name) / 'vocab.txt'
        vocab.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n中\n国\n人\n年\n2\n0\n', encoding='utf-8')
        cls.tokenizer = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=False)
        cls.mapping = {0: 'O', 1: 'B-n', 2: 'I-n', 3: 'B-m', 4: 'I-m'}
        cls.inverse = {label: index for index, label in cls.mapping.items()}

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def item(self, text='中国人中国人中国人'):
        return {'tokens': list(text), 'span_list': [{'start': 0, 'end': len(text) - 1, 'type': 'n'}]}

    def test_window_coverage_and_global_inside_labels(self):
        item = self.item()
        dataset = WindowDataset([item], self.tokenizer, 6, 2, self.inverse)
        supervision = []
        for feature, selected in zip(dataset.features, dataset.selected):
            self.assertLessEqual(len(feature['input_ids']), 6)
            supervision.extend((char for _, char in selected))
            for position, char in selected:
                self.assertEqual(feature['labels'][position], 1 if char == 0 else 2)
        self.assertEqual(sorted(supervision), list(range(len(item['tokens']))))
        logits = np.zeros((len(dataset), 6, len(self.mapping)), dtype=np.float32)
        for window, selected in enumerate(dataset.selected):
            for position, char in selected:
                logits[window, position, 1 if char == 0 else 2] = 5
        merged = merge_window_logits(dataset, logits)
        predictions, errors = predictions_from_logits([item], merged, self.mapping, 'argmax')
        self.assertEqual(errors, 0)
        self.assertEqual(predictions, [item])

    def test_characters_keep_numeric_alignment(self):
        item = self.item('2020年中国')
        dataset = WindowDataset([item], self.tokenizer, 8, 2)
        self.assertEqual(sorted((char for selected in dataset.selected for _, char in selected)), list(range(7)))
        self.assertEqual(dataset.features[0]['input_ids'][1:5], self.tokenizer.convert_tokens_to_ids(list('2020')))

    def test_sentence_formats_keep_word_boundaries_and_unlabeled_characters(self):
        result = format_sentence_prediction('蜘蛛网。2020', ['B-n', 'I-n', 'I-n', 'O', 'B-m', 'I-m', 'I-m', 'I-m'])
        self.assertEqual(result['character_output'], '蜘(n) 蛛(n) 网(n) 。(O) 2(m) 0(m) 2(m) 0(m)')
        self.assertEqual(result['word_output'], '蜘蛛网(n) 。(O) 2020(m)')
        self.assertEqual(''.join((word['text'] for word in result['words'])), result['text'])
        self.assertEqual(result['character_coverage'], 7 / 8)
        self.assertEqual(format_sentence_prediction('中人', ['B-n', 'B-n'])['word_output'], '中(n) 人(n)')
        self.assertEqual(format_sentence_prediction('国中', ['O', 'I-n'])['illegal_bio_transitions'], 1)
        with self.assertRaises(ValueError):
            format_sentence_prediction('中', [])

    def test_multiple_subtokens_and_dropped_characters(self):

        class Encoding(dict):

            def word_ids(self):
                return self['words']

        class Tokenizer:
            is_fast, padding_side, model_max_length = (True, 'right', 512)
            cls_token_id, sep_token_id = (2, 3)

            def __init__(self, drop=False):
                self.drop = drop

            def num_special_tokens_to_add(self):
                return 2

            def build_inputs_with_special_tokens(self, ids):
                return [2] + ids + [3]

            def __call__(self, tokens, **kwargs):
                words = [char for char in range(len(tokens)) for _ in range(2)]
                if self.drop:
                    words = [word for word in words if word != 1]
                return Encoding(input_ids=[9] * len(words), words=words)
        dataset = WindowDataset([self.item()], Tokenizer(), 8, 2, self.inverse)
        self.assertEqual(sum((len(selected) for selected in dataset.selected)), 9)
        self.assertTrue(all((len(feature['input_ids']) <= 8 for feature in dataset.features)))
        with self.assertRaisesRegex(ValueError, '丢弃'):
            WindowDataset([self.item()], Tokenizer(drop=True), 8, 2)

    def test_viterbi_matches_brute_force(self):
        generator = np.random.default_rng(42)
        for _ in range(8):
            logits = generator.normal(size=(4, 5)).astype(np.float32)
            legal = []
            for sequence in itertools.product(range(5), repeat=4):
                labels = [self.mapping[index] for index in sequence]
                if illegal_bio_count(labels) == 0:
                    legal.append((sum((logits[char, index] for char, index in enumerate(sequence))), labels))
            expected = max(legal, key=lambda pair: pair[0])[1]
            self.assertEqual(decode_logits(logits, self.mapping, 'viterbi'), expected)

    def test_evaluation_joint_segmentation_and_coverage(self):
        gold = self.item('中国')
        wrong_tag = {**gold, 'span_list': [{'start': 0, 'end': 1, 'type': 'm'}]}
        report = evaluate_data([gold], [wrong_tag], {'n', 'm'}, long_threshold=1)
        self.assertEqual(report['f1'], 0)
        self.assertEqual(report['segmentation']['f1'], 1)
        self.assertEqual(report['character_coverage'], 1)
        self.assertEqual(report['by_length']['long']['paragraphs'], 1)
        partial = {**gold, 'span_list': [{'start': 0, 'end': 0, 'type': 'n'}]}
        self.assertEqual(evaluate_data([gold], [partial])['character_coverage'], 0.5)

    def test_training_rejects_split_leakage_and_map_mismatch(self):
        pending = {**self.item('中国'), 'augmentation': {'human_review': {'status': 'pending'}}}
        with self.assertRaisesRegex(ValueError, '人工审核'):
            validate_training_data([pending], [self.item('人')], self.mapping)
        review_only = {**self.item('中国'), 'augmentation': {'training_ready': False, 'human_review': {'status': 'approved', 'reviewer': 'test', 'reviewed_at': '2026-10-10'}}}
        with self.assertRaisesRegex(ValueError, '单层训练格式'):
            validate_training_data([review_only], [self.item('人')], self.mapping)
        mapping = {0: 'O', 1: 'B-n', 2: 'I-n'}
        with self.assertRaisesRegex(ValueError, '泄漏'):
            validate_training_data([self.item('中国')], [self.item('中国')], mapping)
        validate_training_data([self.item('中国')], [self.item('中国人')], mapping)
        with self.assertRaisesRegex(ValueError, '类别'):
            validate_training_data([self.item('中国')], [self.item('中国人')], self.mapping)

class CloudFormatTests(unittest.TestCase):

    def test_exact_roundtrip_and_repeated_word_offsets(self):
        item = paragraph()
        content = make_messages(item, 'prompt')[-1]['content']
        self.assertEqual(parse_response(content, item['tokens'], {'n', 'v', 'w'}), item)
        pred = parse_response('{"words":[["人","r"],["人","r"]]}', list('人人'), {'r'})
        self.assertEqual([s['start'] for s in pred['span_list']], [0, 1])

    def test_invalid_answers_are_not_repaired(self):
        bad = ['```json\n{"words":[["甲","n"]]}\n```', '{"words":[["甲","other"]]}', '{"words":[["乙","n"]]}', '{"words":[["甲甲","n"]]}', '{"words":[],"words":[["甲","n"]]}', '{"words":[["甲","n"]],"explanation":"文字"}', '{"words":[["","n"]]}', '{"words":[]}']
        for content in bad:
            with self.subTest(content=content), self.assertRaises(ValueError):
                parse_response(content, ['甲'], {'n'})


class TextFilePredictionTests(unittest.TestCase):
    def setUp(self):
        from types import SimpleNamespace
        self.model = SimpleNamespace(config=SimpleNamespace(id2label={
            0: "O", 1: "B-n", 2: "I-n", 3: "B-ns", 4: "I-ns",
        }))

    @staticmethod
    def fake_prediction(text, *args, **kwargs):
        labels = ["B-n"] + ["I-n"] * (len(text) - 1)
        if text == "北京#":
            labels = ["B-ns", "I-ns", "O"]
        return format_sentence_prediction(text, labels)

    def test_text_file_preserves_whitespace_offsets_and_unlabeled_characters(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "文章.txt", root / "result"
            raw = " \t蜘蛛 北京#\r\n\r\n\t \r\n珊瑚".encode("utf-8-sig")
            source.write_bytes(raw)
            with patch("main.predict_sentence", side_effect=self.fake_prediction) as predict:
                result = predict_text_file(source, output, self.model, None, None)
                self.assertEqual(predict.call_count, 3)
            records = [json.loads(line) for line in (output / "predictions.jsonline").read_text(
                encoding="utf-8").splitlines()]
            self.assertEqual([row["line_number"] for row in records], [1, 3, 4])
            self.assertEqual("".join(records[0]["tokens"]), " \t蜘蛛 北京#")
            self.assertEqual(records[0]["span_list"], [
                {"start": 2, "end": 3, "type": "n"}, {"start": 5, "end": 6, "type": "ns"},
            ])
            self.assertEqual(records[1]["span_list"], [])
            for record in records:
                validate_item(record, {"n", "ns"})
            self.assertEqual((output / "words_zh.txt").read_bytes().decode("utf-8"),
                             " \t蜘蛛(名词) 北京(地名) #(未标注)\r\n\r\n\t \r\n珊瑚(名词)")
            self.assertEqual((result["lines"], result["records"], result["predicted_characters"],
                              result["tagged_characters"], result["words"]), (4, 3, 7, 6, 3))
            with self.assertRaises(FileExistsError):
                predict_text_file(source, output, self.model, None, None)
            self.assertEqual(source.read_bytes(), raw)

    def test_text_file_supports_explicit_encoding_and_prediction_options(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "gb.txt"
            source.write_bytes("蜘蛛\n".encode("gb18030"))
            with patch("main.predict_sentence", side_effect=self.fake_prediction) as predict:
                predict_text_file(source, root / "out", self.model, None, None,
                                  encoding="gb18030", decoder="argmax", batch_size=2)
            self.assertEqual(predict.call_args.kwargs["decoder"], "argmax")
            self.assertEqual(predict.call_args.kwargs["batch_size"], 2)

    def test_text_file_rejects_empty_or_invalid_input_without_partial_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "input.txt", root / "out"
            for content, error_type in [(b" \r\n\t", ValueError), (b"\xff\xfe", UnicodeDecodeError)]:
                source.write_bytes(content)
                with self.assertRaises(error_type):
                    predict_text_file(source, output, self.model, None, None)
                self.assertFalse(output.exists())
            source.write_text("蜘蛛\n错误", encoding="utf-8")
            with patch("main.predict_sentence", side_effect=[self.fake_prediction("蜘蛛"), ValueError("丢字")]):
                with self.assertRaisesRegex(ValueError, "第 2 行.*丢字"):
                    predict_text_file(source, output, self.model, None, None)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
