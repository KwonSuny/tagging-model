# 基于 BERT 的中文分词与词性联合标注

使用 `bert-base-chinese` 对中文文本进行词级分词与词性联合标注，提供可追溯的数据清洗、独立数据划分、完整文本滑窗训练和严格评测。同时提供阿里云百炼千问 SFT 数据转换与生成结果评测工具。

当前任务采用单层词级 BIO 标注。词内部的语素候选作为独立研究材料保存，不直接并入词级训练。

## 功能

- 校验词性、字符坐标与完整覆盖，修复已确认的源数据错误；处理短语元标记、重复文本和冲突标注。
- 保留官方开发集，从原训练语料构建独立测试集，并留出单独报告的补充诊断集。
- 按字符对齐 BERT 子词，采用重叠滑窗处理完整段落；每个字符只参与一次监督。
- 支持 argmax 和 BIO 约束 Viterbi 解码，按完整段落的联合 F1 选择检查点。
- 报告分词 F1、词性联合 F1、逐标签指标、长段落表现与覆盖率。
- 交互式输入句子，输出 `字(属性)` 和 `完整词(属性)` 两种形式。
- 读取普通文本文件，保留原行空白与字符坐标，导出标准 JSONL 和 `词(中文词性)` 文本。
- 导出千问 ChatML/SFT 数据，严格校验生成答案并将失败响应纳入评分。

## 目录

```text
.
├── clean_data.py          # 原始语料清洗及测试集划分
├── supplement_data.py    # 语素研究材料生成，不自动加入训练
├── prepare_dataset.py    # 正式数据版本、审核导入和诊断划分
├── train.py              # BERT 训练入口
├── main.py               # 字符对齐、滑窗、解码和预测
├── eval.py               # 分词与词性联合评测
├── bailian_data.py       # 百炼 SFT 导出和生成结果评测
├── notebooks/
│   ├── trainmodel.ipynb
│   └── testmodel.ipynb
├── tests/test_core.py    # 17 项核心回归测试
├── docs/
│   ├── baseline_metrics.json
│   └── templates/Trainingdata.jsonl
├── requirements.txt
└── README.md
```

原始语料、`data/`、`outputs/`、模型权重、历史预测、个人审核记录及缓存不包含在源码发布包中。运行数据处理后会生成相应目录。运行命令应在项目根目录执行；Notebook 可以从根目录或 `notebooks/` 启动。

## 环境安装

已验证 Python **3.12**、PyTorch **2.6.0**、Transformers **4.49.0**、Accelerate **1.4.0**、NumPy **2.4.4**。

建议新建虚拟环境。Windows PowerShell 示例：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

根据硬件选择一种 PyTorch 安装方式，再安装项目依赖：

```shell
# CPU
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu

# 或 NVIDIA CUDA 11.8；需要兼容的显卡与驱动
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118

python -m pip install -r requirements.txt
```

其他平台和 CUDA 组合参考 [PyTorch 官方历史版本安装说明](https://pytorch.org/get-started/previous-versions/)。只运行命令行入口不需要 Jupyter；Notebook 需另行配置 Jupyter 或 VS Code 的 Python/Jupyter 内核。

## 数据准备

### 原始数据

本项目使用 PKU-SEGPOS 词性标注语料。请从有权使用的数据来源取得对应版本，并放入：

```text
PKU-SEGPOS/
├── train.jsonline         # 31,772 段
├── dev.jsonline           # 5,204 段
└── test_public.jsonline   # 4,764 段，无公开答案
```

每行是一个 JSON 对象，`tokens` 为单字符列表；`span_list` 中的 `start/end` 从 0 开始，**end 包含在区间内**：

```json
{"tokens":["蜘","蛛","结","网"],"span_list":[{"start":0,"end":1,"type":"n"},{"start":2,"end":2,"type":"v"},{"start":3,"end":3,"type":"n"}]}
```

脚本包含针对已确认下载版本的修复和位置检查，包括“卯”的类别、三处多余 B 和独立“喏”的词性。使用其他下载版本、不同的行顺序或自定义语料时，需要先复核这些规则；校验失败时应核对输入，不能直接绕过检查。原始文件不被改写。

### 清洗与冻结

按顺序执行，输出目录必须尚不存在：

```shell
python clean_data.py --source PKU-SEGPOS --output data/cleaned --test-ratio 0.1 --seed 42
python supplement_data.py --data data/cleaned --output data/augmentation_pending_v2
python prepare_dataset.py --source data/cleaned --review-dir data/augmentation_pending_v2 --output data/prepared_v3 --diagnostic-ratio 0.05 --seed 42
```

第二步生成独立的语素研究材料及审核框，初始状态为未审核，不进入训练；第三步导入状态和意见并独立保存。导入实际人工意见时，应通过 `prepare()` 的 `reviewer/date` 参数记录真实审核人和日期。源码包不包含原项目作者的个人审核记录，因此重建的审核元数据不等同于原实验记录。

对应输入版本下，正式词级数据规模如下：

| 集合 | 段落数 | 用途 |
| --- | ---: | --- |
| train | 26,593 | 梯度训练 |
| dev | 5,153 | 官方开发集，主要比较与选模 |
| diagnostic_dev | 1,400 | 独立留出的补充诊断，单独报告 |
| test | 3,110 | 从原训练集划出的最终测试集 |

有效词性为 **43 类、87 个 BIO 标签**。`label_catalog.json`、`provenance.jsonl`、`manifest.json` 记录映射、来源和 SHA256。没有可靠正例的 Eg 仍在登记表中，但不作为正式可训练类别。

划分以完整段落为单位，保证文本去重与集合隔离。源文件没有文章 ID，不能保证文章级隔离。自建测试集不等于官网隐藏测试集，官网公开输入不能用于计算有监督 F1。

## 训练

从原始 `bert-base-chinese` 预训练权重开始，基线配置示例：

```shell
python train.py --train data/prepared_v3/train.jsonline --dev data/prepared_v3/dev.jsonline --catalog data/prepared_v3/label_catalog.json --output-dir outputs/bert-prepared-v3-lr3e5-seed42 --learning-rate 3e-5 --epochs 6 --batch-size 16 --gradient-accumulation 4 --max-len 256 --overlap 64 --decoder viterbi --seed 42
```

首次运行可能需要联网下载 BERT；已有缓存可添加 `--local-files-only`。有效批次为 64，AdamW、weight decay 0.01、warmup 0.1、linear 调度、梯度裁剪 1.0；连续两次官方 dev 联合 F1 不提升则提前停止。精度自动选择支持的 BF16、CUDA FP16 或 CPU FP32。

训练输出包含 `best_model/`、检查点、`run_manifest.json`、`trainer_state.json` 与完整 dev 报告。实际标签映射随模型保存，不能用新数据推断旧模型的 `LABEL_n` 编号。不同标签映射不能互相恢复训练。

新实验需使用新输出目录。恢复同一未完成实验时使用 `--resume <当前实验的checkpoint目录>`，数据、配置和依赖必须一致；已完成预算的实验拒绝恢复。完整段落评测当前仅支持单进程训练，且应为检查点和最佳模型预留数 GB 空间。

## 预测和评测

```shell
python main.py --model outputs/bert-prepared-v3-lr3e5-seed42/best_model --input data/prepared_v3/dev.jsonline --output outputs/predictions/dev.jsonline --local-files-only
python eval.py outputs/predictions/dev.jsonline data/prepared_v3/dev.jsonline --catalog data/prepared_v3/label_catalog.json --output outputs/predictions/dev_metrics.json
```

补充诊断集应另行预测和评测，不能与官方 dev 合并成历史可比指标。完成调参与方案选择后再使用 test 一次验收。输出与指标文件拒绝覆盖。

主指标是精确匹配 `(start,end,type)` 的词级联合 micro-F1，同时报告仅匹配边界的分词 F1、逐标签 P/R/F1 和支持量、长段落表现及 span 覆盖率。gold 必须完整覆盖；漏标会降低召回。输入没有截断并不意味着预测没有漏标。

`notebooks/testmodel.ipynb` 提供独立的人工输入区，可调用：

```python
inspect_sentence("蜘蛛结了一张网。", show_bio=True)
```

输出两种形式，例如 `蜘(n) 蛛(n)` 和 `蜘蛛(n)`，也可显示 BIO。输出是模型预测。被 BERT tokenizer 丢弃的空格、换行等字符会明确报错，不静默修改原文。

### 普通文本文件预测

在 `notebooks/testmodel.ipynb` 中先运行“加载模型”单元格，再运行末尾的“普通文本文件预测与中文词性导出”。设置 `input_text_path`、`text_output_dir` 和 `text_encoding`；无需先运行有标签数据评测或人工输入。相对路径以项目根目录为准，输出目录必须尚不存在。

默认读取 UTF-8（兼容 BOM）；GBK/GB18030 文件可指定 `text_encoding = "gb18030"`，不自动猜测或忽略解码错误。模型只加载一次，后续文件复用该模型。每个文件生成：

- `predictions.jsonline`：标准 `tokens + span_list`，每个非空物理行一条记录，附原文件行号 `line_number`（从 1 开始）。tokens 保留原行空白，不包含行尾换行；span 使用原行从 0 开始、包含末端的字符坐标。
- `words_zh.txt`：中文词性格式，例如 `蜘蛛(名词) 北京(地名)`；O 显示为 `未标注`。保留缩进、空白、空行及换行方式，词之间增加空格方便阅读。示例仅说明格式，实际标签以模型预测为准。

文件按行、按空白分段，长段复用滑窗，不截断；模型上下文不会跨越分隔边界。空白不赋予词性，真正的空行不生成 JSONL 记录，纯空白行保留 tokens 和空 span_list；因此这些输出不能直接作为要求完整覆盖的训练数据。其他会被 tokenizer 丢弃的字符明确报错并提示行号；空文件、纯空白文件和解码错误均拒绝，失败时清理本次未完成的输出。

也可在已加载模型的 Python 环境中调用：

```python
from main import predict_text_file
result = predict_text_file(
    "inputs/example.txt", "outputs/predictions/text_file_run",
    inspection_model, inspection_tokenizer, inspection_device,
    encoding="utf-8-sig", decoder="viterbi",
)
```

## 基线结果

以下为本地已复核的参考实验记录，完整摘要见 [docs/baseline_metrics.json](docs/baseline_metrics.json)。训练使用上述配置，最佳模型来自第 5 轮；独立推理使用 FP32 和 Viterbi。

| 集合 | 联合 F1 | 分词 F1 |
| --- | ---: | ---: |
| 官方 dev | 94.1237% | 97.1661% |
| 补充诊断集 | 95.6962% | 97.8495% |

这些是已完成实验的记录，不表示下载源码即可获得相同模型；权重未随仓库提供。重新训练的结果会受输入版本、依赖、硬件与随机性影响。总体分数不能代表稀有类别效果；原语料的 n/ns 等标注口径仍有疑问，需要结合上下文审核，不按同词多数标签自动改写。

## 阿里云百炼数据准备

将词级标注转换为 system/user/assistant 三条消息的 JSONL，assistant 输出 `{"words":[["完整词","词性"],...]}`。本地按原文顺序还原字符区间，不把 BERT 标签 ID 或 BIO 直接上传。

```shell
python bailian_data.py export --source data/prepared_v3 --output data/bailian_v1 --template docs/templates/Trainingdata.jsonl
```

模板是本项目自建的格式示例，不属于正式训练数据。导出包含全量 train/dev、独立诊断、小试子集、无答案推理请求、版本摘要与训练方案；测试答案不导出。小试是原 train/dev 的子集，不能作为新增独立评测数据。

上面的命令只做格式转换。训练前应使用目标模型分词器核算长度，例如已有本地 Qwen3-8B 分词器时添加 `--tokenizer <本地分词器目录> --max-length 8192`。未提供分词器会明确记录“未测量”，不能据此认为样本不会超限。本项目没有捆绑分词器或千问权重，也没有自动上传或付费调用。

模型原始响应保存为 JSONL，每行携带对应请求的 `id`、`text_sha256`、`finish_reason` 与字符串 `content`，再评测：

```shell
python bailian_data.py evaluate --gold data/bailian_v1/pilot_dev.gold.jsonline --requests data/bailian_v1/pilot_dev.requests.jsonl --responses outputs/qwen_pilot_responses.jsonl --output outputs/qwen_pilot_metrics.json
```

缺失响应、截断、无效 JSON、未知标签和增删改字均作为整段无预测计入 FN，不从分母跳过；重复或额外 ID 被拒绝。平台 Validation Token Accuracy 不能替代本项目的词级联合 F1。具体训练模型、参数、数据上传限制及费用以百炼当时文档与控制台为准。

## 核心测试

```shell
python -B -m unittest discover -s tests -v
```

发布版保留 17 项核心回归测试，覆盖标签规范化、短语元标记处理、冲突去重、集合泄漏、完整字符覆盖、子词对齐、Viterbi、联合评分、生成答案解析，以及文本文件的空白坐标、编码和失败清理。测试使用合成的小型样本，不需要语料、预训练模型或云端调用。

## 数据来源与许可

PKU-SEGPOS 为北京大学提供的人民日报词性/分词语料，原作者为段慧明、常宝宝、穗志方。原始数据、预训练模型和第三方依赖遵循各自的使用条件与许可证，请从有权使用的来源获取数据。


参考：[Transformers 4.49 Trainer](https://huggingface.co/docs/transformers/v4.49.0/main_classes/trainer)、[Token classification](https://huggingface.co/docs/transformers/v4.49.0/tasks/token_classification)、[百炼数据上传规则](https://help.aliyun.com/zh/model-studio/text-generation-tuning-data-upload-rules)。
