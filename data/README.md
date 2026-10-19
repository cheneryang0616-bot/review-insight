# 数据说明（重要）

## 为什么仓库里没有完整数据

1. **个人信息**：原始 `meituan.xlsx` 含「评论用户姓名」「评论用户id」两列，涉及 **27,686 个用户**，
   属于《个人信息保护法》意义上的个人信息，不适合公开转分发。
2. **第三方内容**：评论正文是用户原创 UGC，著作权属于评论者本人；整份数据是爬取所得，
   公开转分发既不合适也没有必要。
3. **上游仓库无 LICENSE**：数据来自开源仓库
   [wq624915051/meituan_text_minning](https://github.com/wq624915051/meituan_text_minning)，
   该仓库未声明许可证，默认保留所有权利。

所以本仓库只放**脱敏样例** `sample_reviews.csv`（300 行，已删除姓名与用户 id，按星级分层抽样）。

## 两份数据分别是什么

| 场景 | 文件 | 来源 | 规模 |
|---|---|---|---|
| 评论归因 | `meituan_reviews.xlsx` | 开源仓库 wq624915051/meituan_text_minning | 42,038 条 |
| 信用风险 | `cs-training.csv` | Kaggle *Give Me Some Credit* | 150,000 条 |
| 信用风险（合规演示） | `german_credit.xlsx` | UCI *Statlog German Credit* | 1,000 条 |

仓库自带的是 `sample_reviews.csv`（225 行）与 `sample_credit.csv`（3,000 行）。

## 怎么拿到完整数据（想复现报告数字的人）

```bash
# 评论数据：克隆上游仓库，把文件拷过来
git clone https://github.com/wq624915051/meituan_text_minning.git /tmp/mtn
cp "/tmp/mtn/美团文本分析/meituan.xlsx" data/meituan_reviews.xlsx

# 信贷数据：Kaggle Give Me Some Credit → data/cs-training.csv
#           UCI German Credit → data/german_credit.xlsx
# 信贷数据用哪份：设置环境变量 RI_CREDIT_DATA=german 或 gmc

# 方式二：如果上游已失效，用自己的数据，但必须保持这 7 列
#   评论店铺 / 评论用户姓名 / 评论用户id / 评论用户星级 / 评论用户菜品 / 评论用户内容 / 评论时间
```

放好之后无需改代码：`data.load()` 会自动优先使用 `data/meituan_reviews.xlsx`，
不存在时回退到 `sample_reviews.csv`，并在 DataFrame 的 `attrs["source"]` 里标记来源。

## 两种数据下的差异

| | 完整数据 | 样例数据 |
|---|---|---|
| 行数 | 42,038 | 300 |
| `attrs["source"]` | `full` | `sample` |
| `results/REPORT.md` 的数字 | ✅ 由完整数据生成 | 不适用 |
| 单元测试中依赖规模的断言 | 执行 | 自动跳过 |
| 端到端问答 | 正常 | 正常（样本小，部分工具会因样本量门槛返回「无数据」） |

`results/REPORT.md` 里的所有数字都来自完整公开数据集，仓库里不带原始数据不影响报告的可信度 ——
但**任何人重跑报告都需要自己获取完整数据**，这一点写在 README 的快速开始里。

## 合规提醒

- 不要把这些评论正文再分发到别处
- 不要把本项目的分析结论说成任何公司的真实业务数据
- 如果要在公司/课程场景使用，请确认数据来源与使用范围
