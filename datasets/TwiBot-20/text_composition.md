# TwiBot-20 — 文本组成与预处理说明

此文档把论文中“将用户信息编码为文本序列”的描述与代码实现逐步对应。核心目标：让你明确每个字段、占位符、tokenizer 配置与训练时的数据流，方便复现或定制预处理。

概览：项目使用预处理后的字符串序列（存于 `norm_user_text.json`），该序列由三部分组成并用显式域标签分隔：

- `METADATA:`（用户元信息，字段以 `</s>` 分隔）
- `DESCRIPTION:`（用户个人简介）
- `TWEET:`（多条推文的拼接，tweet 间也用 `</s>` 分隔）

这些域标签与若干占位符（`@USER`, `HTTPURL`, `#HASHTAG`, `EMOJI`, `RT` 等）在 `model_building.py` 中注册为 tokenizer 的自定义 token，以保证 tokenization 的一致性与鲁棒性。

---

## 结构示例（来自 `norm_user_text.json`）

```
METADATA: Tue Nov 18 10:27:25 2008 </s> Orlando, FL </s> SHAQ </s> False </s> 15349596 </s> 692 </s> 9798 </s> 45568 </s> SHAQ </s> True
DESCRIPTION: VERY QUOTATIOUS , I PERFORM RANDOM ACTS OF SHAQNESS
TWEET: RT @USER : EMOJI Are you ready to see what our newest ship's name will be ? EMOJI ...
```

字段说明（示例）：
- `METADATA`：可能包含创建时间、位置、用户名、protected/verified、用户 id、followers、friends、listed、favourites、screen_name 等；用 `</s>` 分隔子项。
- `DESCRIPTION`：profile bio 文本。
- `TWEET`：若干条 tweet 的拼接文本，内部保留占位符（见下）并用 `</s>` 分隔。

---

## 预处理策略（论文 ↔ 代码映射）

1) 域标签串联（[M]/[D]/[T]）
    - 论文：把 metadata/description/tweets 按域标签串成单序列。
    - 代码：使用字面 token `METADATA:`, `DESCRIPTION:`, `TWEET:`，并在 `model_building.py` 通过 `LM_tokenizer.add_special_tokens` 将它们注册为 `additional_special_tokens`。

2) 噪声归一化（hashtags/mentions/URLs → 占位符）
    - 论文：把 `#...`、`@...`、URL 替换为 `#HASHTAG`、`@USER`、`HTTPURL`。
    - 代码/数据：`norm_user_text.json` 已包含这些占位符，说明该替换在预处理阶段完成（预处理脚本未包含在仓库源码里，但 README 提到可下载预处理数据）。

3) 轻量去噪（tokenization）
    - 论文参考了 TweetTokenizer2 风格的处理；在实践中可用 NLTK 的 `TweetTokenizer` 或等效工具做基本分词与规范化（示例代码在下方）。

4) tokenizer 配置
    - `model_building.py` 中将以下 token 注册到 tokenizer：

```py
special_tokens_dict = {'additional_special_tokens': ['DESCRIPTION:','METADATA:','TWEET:']}
LM_tokenizer.add_special_tokens(special_tokens_dict)
LM_tokenizer.add_tokens(["@USER", '#HASHTAG', 'HTTPURL', 'EMOJI', 'RT', 'None'])
LM_model.LM.resize_token_embeddings(len(LM_tokenizer))
```

5) 训练输入
    - `utils.load_raw_data()` 读取 `norm_user_text.json`；`dataloader.build_LM_dataloader()` 根据索引构造 DataLoader；训练时 `trainer.LM_Trainer.batch_to_tensor()` 调用 tokenizer（`add_special_tokens=False`，`truncation=True`，`max_length` 默认 512）将字符串转为模型输入 ids。

6) LM 表示
    - 在 `LM.py` 中，模型取 transformer 最后一层 hidden states 的均值作为用户表示（mean pooling）：

```py
out = self.LM(output_hidden_states=True, **tokenized_tensors)['hidden_states']
embedding = out[-1].mean(dim=1)
```

---

## 可复现的预处理示例（参考实现）

下面给出一个简单且可定制的预处理示例，演示如何把原始 tweets/description/metadata 变为 `norm_user_text` 中的格式：

```py
import re
from nltk.tokenize import TweetTokenizer

URL_RE = re.compile(r'https?://\S+|www\.\S+')
MENTION_RE = re.compile(r'@\w+')
HASHTAG_RE = re.compile(r'#\w+')

tknzr = TweetTokenizer(preserve_case=False, reduce_len=True)

def normalize_text(s: str) -> str:
     s = URL_RE.sub('HTTPURL', s)
     s = MENTION_RE.sub('@USER', s)
     s = HASHTAG_RE.sub('#HASHTAG', s)
     toks = tknzr.tokenize(s)
     return ' '.join(toks)

def build_user_text(metadata_items, description, tweets):
     meta_str = ' </s> '.join([str(i) for i in metadata_items if i is not None and i != ''])
     desc = normalize_text(description or '')
     tweets_norm = [normalize_text(t) for t in tweets]
     tweets_str = ' </s> '.join(tweets_norm)
     return f"METADATA: {meta_str} </s> DESCRIPTION: {desc} TWEET: {tweets_str}"
```

注意：实际预处理脚本可能包含更多的清洗步骤（emoji 归一化、多语言处理、停用词策略等），这里提供最小可运行示例以便快速复现论文中描述的核心思想。

---

## 实践检验清单（快速验证）

- `len(norm_user_text.json)` 应与 `labels.pt`.shape[0] 相等。
- `edge_index` 的最大/最小索引应在 `[0, N-1]`。
- 若新增占位符 token，请在 `model_building.py` 中调用 `add_tokens` 并 `resize_token_embeddings`。
- 训练时注意 `max_length`（512）会截断超长文本；可用 sliding-window 或更短的文本聚合策略应对长历史推文。

---
