# 仇恨言论检测系统 — 基于BERT与指针网络

本项目实现了一个端到端的仇恨言论检测系统，结合了BERT预训练编码器与指针网络，能够检测文本中的仇恨/非仇恨属性，并抽取目标（Target）和论点（Argument），同时对论点进行群体（Group）多标签分类。

## 项目结构

```
.
├── 模型训练.py              # 核心脚本，包含数据预处理、模型定义、训练、评估与推理
├── data/
│   ├── train.json           # 训练集数据（含 content 和 output）
│   └── test1.json           # 测试集数据，用于推理
├── roberta-wwm-ext-as-bert/ # 本地预训练模型目录，需下载并解压至本目录
├── final_pointer_model.pt   # 训练结束后保存的模型权重
└── predictions.txt          # 推理结果输出文件
```

## 环境依赖

- Python 3.8+
- torch
- transformers
- scikit-learn
- tqdm
- numpy

安装方法：

```bash
pip install torch transformers scikit-learn tqdm numpy
```

## 快速开始

1. **准备数据**
   - 将 `train.json` 和 `test1.json` 放入 `data/` 目录。
2. **下载预训练模型**
   - 下载 RoBERTa-wwm-ext（或与 `BertModel` 兼容的预训练模型），解压到项目根目录，目录名为 `roberta-wwm-ext-as-bert`。
3. **运行脚本**
   ```bash
   python 模型训练.py
   ```
   脚本会自动执行：
   - 基于 `data/train.json` 训练模型，并保存到 `final_pointer_model.pt`
   - 在训练集上评估并打印群体分类与仇恨分类的 Macro-F1 分数
   - 加载保存的模型，对 `data/test1.json` 进行推理，将结果写入 `predictions.txt`
4. **查看结果**
   - 模型权重：`final_pointer_model.pt`
   - 推理输出：`predictions.txt`

## 数据格式说明

```json
{
  "content": "待检测文本",
  "output": "target | argument | group | hate [END] [SEP] ..."
}
```

- `content`：文本字符串
- `output`：标注的四元组序列，每组由 `|` 分隔
  - `target`：攻击目标
  - `argument`：攻击论点
  - `group`：群体标签，多标签逗号分隔
  - `hate`：`hate` 或 `non-hate`
  - 每组以 `[END]` 结尾，组间用 `[SEP]` 分隔

## 参数配置

在 `if __name__ == "__main__"` 模块顶部，可修改：

- `train_json`：训练集路径
- `test_json`：测试集路径
- `LOCAL_ROBERTA_DIR`：本地预训练模型目录
- `NUM_EPOCHS`：训练轮次
- `LEARNING_RATE`：学习率

## 输出文件

- `final_pointer_model.pt`：训练好的模型参数
- `predictions.txt`：推理脚本生成的四元组预测结果

## 许可证

本项目使用 MIT 许可证，详情见 LICENSE 文件。

