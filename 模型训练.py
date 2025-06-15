# -*- coding: utf-8 -*-
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score


# 设备选择
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 群体分类和仇恨分类映射
GROUP2ID = {
    "Region": 0,
    "Racism": 1,
    "Sexism": 2,
    "LGBTQ": 3,
    "others": 4
}
ID2GROUP = {v: k for k, v in GROUP2ID.items()}
NUM_GROUPS = len(GROUP2ID)
HATEID2STR = {1: "hate", 0: "non-hate"}

# 超参数
MAX_LEN = 128
BATCH_SIZE = 8
NUM_EPOCHS = 4
LEARNING_RATE = 2e-5
WARMUP_PROPORTION = 0.1

class HateSpeechDataset(Dataset):
    """
    数据集类，负责：
    1. 读取 JSON 文件
    2. 将每条样本中的 output 字符串拆成多个 (tar, arg, group, hate) 四元组
    3. 将文本通过 tokenizer 转为 token_ids，并根据 tar/arg 文本计算对应的 token-span
    4. 返回一个包含所有特征信息的字典
    """
    def __init__(self, json_path, tokenizer, max_len=128, mode="train"):
        super().__init__()
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_len = max_len

        # 读取所有原始样本
        with open(json_path, "r", encoding="utf-8") as f:
            self.examples = json.load(f)

        # 把每个原始样本拆成若干个子样本（每个 quad 对应一个子样本）
        self.features = []
        for ex in self.examples:
            raw_output = ex.get("output", "").strip()
            quads = self._split_quads(raw_output)
            # 如果没有任何合法的四元组，则添加一个空四元组：让 arg_span 覆盖整句
            if len(quads) == 0:
                quads = [(None, None, "others", "non-hate")]

            for tar, arg, group, hate in quads:
                feat = self._preprocess_one(
                    content=ex["content"],
                    tar=tar,
                    arg=arg,
                    group=group,
                    hate=hate
                )
                self.features.append(feat)

    def _split_quads(self, raw_output: str):
        """
        将 output 字符串按 “[SEP]” 拆成若干个四元组 (tar, arg, group, hate)，
        每段必须以 “[END]” 结尾，并且 body 部分用 “|” 分成 4 个字段。否则忽略该段。
        返回形如 [(tar1, arg1, group1, hate1), (tar2, arg2, group2, hate2), ...]
        """
        quads = []
        if not raw_output:
            return quads
        parts = [segment.strip() for segment in raw_output.split("[SEP]")]
        for segment in parts:
            if not segment.endswith("[END]"):
                continue
            body = segment[:-5].strip()  # 去掉尾部的 "[END]"
            fields = [f.strip() for f in body.split("|")]
            if len(fields) != 4:
                continue
            tar_text, arg_text, group_text, hate_text = fields
            quads.append((tar_text, arg_text, group_text, hate_text))
        return quads

    def _preprocess_one(self, content: str, tar: str, arg: str, group: str, hate: str):
        """
        对单个四元组 (tar, arg, group, hate) 与 content 做处理：
        1) 用 tokenizer 对 content 编码，获取 input_ids, attention_mask, offsets
        2) 根据 tar 文本计算 target_span (token 级别)，若 tar=None 或对齐失败则 target_span=None
        3) 根据 arg 文本计算 arg_span (token 级别)，若 arg=None 或对齐失败则退化为整句
        4) 构造 group multi-hot 向量、hate id
        返回一个字典，包含所有需要的特征。
        """
        text = content
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_offsets_mapping=True
        )
        input_ids = encoding["input_ids"].squeeze(0)            # Tensor(max_len,)
        attention_mask = encoding["attention_mask"].squeeze(0)  # Tensor(max_len,)
        offsets = encoding["offset_mapping"].squeeze(0).tolist() # List[(char_s, char_e)]

        # —— 1. 处理 target_span ——
        target_span = None
        if tar is not None and tar.strip():
            pos = text.find(tar)
            if pos >= 0:
                pos_end = pos + len(tar)
                tok_s, tok_e = None, None
                for idx, (s, e) in enumerate(offsets):
                    if s <= pos < e:
                        tok_s = idx
                    if s < pos_end <= e:
                        tok_e = idx
                if tok_s is not None and tok_e is not None:
                    target_span = (tok_s, tok_e)
        # 如果 tar 是 None 或对齐失败，就保留 target_span = None

        # —— 2. 处理 arg_span ——
        # 无论如何，都要保证 arg_span 不为 None。即如果无法对齐，就令 arg_span 覆盖整句有效 token。
        last_valid_idx = 0
        for idx, (s, e) in enumerate(offsets):
            if e == 0:  # 第一次出现 (0,0) 表示后面全是 padding
                break
            last_valid_idx = idx

        arg_span = None
        if arg is not None and arg.strip():
            pos_a = text.find(arg)
            if pos_a >= 0:
                pos_a_end = pos_a + len(arg)
                tok_s, tok_e = None, None
                for idx, (s, e) in enumerate(offsets):
                    if s <= pos_a < e:
                        tok_s = idx
                    if s < pos_a_end <= e:
                        tok_e = idx
                if tok_s is not None and tok_e is not None:
                    arg_span = (tok_s, tok_e)
        # 如果 arg 文本对齐失败或 arg=None，就退化为整句
        if arg_span is None:
            arg_span = (0, last_valid_idx)

        # —— 3. 处理 group & hate 标签 ——
        # hate: 单标签，"hate"->1, "non-hate"->0
        hate_id = 0 if (hate is None or hate.lower() == "non-hate") else 1

        # group: 多标签，可以有逗号分隔多个子标签
        group_multi_hot = torch.zeros(NUM_GROUPS, dtype=torch.float)
        if group is None:
            group_multi_hot[GROUP2ID["others"]] = 1.0
        else:
            for g in group.split(","):
                g = g.strip()
                if g in GROUP2ID:
                    group_multi_hot[GROUP2ID[g]] = 1.0
            # 如果没有任何合法标签，强制把 "others" 置 1
            if group_multi_hot.sum().item() == 0:
                group_multi_hot[GROUP2ID["others"]] = 1.0

        return {
            "input_ids":     input_ids,          # Tensor(max_len,)
            "attention_mask": attention_mask,     # Tensor(max_len,)
            "target_span":    target_span,       # (tok_s, tok_e) 或 None
            "arg_span":       arg_span,          # (tok_s, tok_e)，永不为 None
            "group_label":    group_multi_hot,   # Tensor(NUM_GROUPS,)
            "hate_label":     hate_id,           # 0/1
            "orig_text":      text,              # 原始 content 字符串
            "offsets":        offsets            # List[(char_s, char_e)]
        }

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

def collate_fn(batch):
    """
    将 Dataset 返回的若干 features batch 化：
    - 拼接 input_ids, attention_mask
    - 构造 target_start_labels, target_end_labels, arg_start_labels, arg_end_labels
    - 构造 target_exists_mask, arg_exists_mask（后者全 True）
    - 拼接 group_labels, hate_labels
    - 同时返回 orig_texts、offsets（推理时需要）
    """
    batch_size = len(batch)
    input_ids = torch.stack([ex["input_ids"] for ex in batch], dim=0)         # (B, L)
    attention_mask = torch.stack([ex["attention_mask"] for ex in batch], dim=0)  # (B, L)

    target_start_labels = torch.zeros((batch_size,), dtype=torch.long)
    target_end_labels   = torch.zeros((batch_size,), dtype=torch.long)
    arg_start_labels    = torch.zeros((batch_size,), dtype=torch.long)
    arg_end_labels      = torch.zeros((batch_size,), dtype=torch.long)

    target_exists_mask = torch.zeros((batch_size,), dtype=torch.bool)
    arg_exists_mask    = torch.ones((batch_size,), dtype=torch.bool)  # arg_span 永远存在

    group_labels = torch.zeros((batch_size, NUM_GROUPS), dtype=torch.float)
    hate_labels  = torch.zeros((batch_size,), dtype=torch.long)

    orig_texts = []
    offsets_batch = []

    for i, ex in enumerate(batch):
        # target span
        if ex["target_span"] is not None:
            ts, te = ex["target_span"]
            target_start_labels[i] = ts
            target_end_labels[i]   = te
            target_exists_mask[i]  = True
        else:
            # 对于没有 target 的样本，labels 保持默认 (0,0)，并且 exists_mask=False
            target_start_labels[i] = 0
            target_end_labels[i]   = 0

        # argument span：一定有
        as_, ae = ex["arg_span"]
        arg_start_labels[i] = as_
        arg_end_labels[i]   = ae

        # group, hate
        group_labels[i, :] = ex["group_label"]
        hate_labels[i]     = ex["hate_label"]

        orig_texts.append(ex["orig_text"])
        offsets_batch.append(ex["offsets"])

    return {
        "input_ids":           input_ids,             # (B, L)
        "attention_mask":      attention_mask,        # (B, L)
        "target_start_labels": target_start_labels,   # (B,)
        "target_end_labels":   target_end_labels,     # (B,)
        "arg_start_labels":    arg_start_labels,      # (B,)
        "arg_end_labels":      arg_end_labels,        # (B,)
        "target_exists_mask":  target_exists_mask,    # (B,)
        "arg_exists_mask":     arg_exists_mask,       # (B,)
        "group_labels":        group_labels,          # (B, NUM_GROUPS)
        "hate_labels":         hate_labels,           # (B,)
        "orig_texts":          orig_texts,            # List[str]，推理时需要
        "offsets":             offsets_batch          # List[List[(char_s,char_e)]]
    }

class HateSpeechPointerModel(nn.Module):
    """
    Span-based 指针模型，包含：
    1) BERT 编码器
    2) Target 指针头（start + end）
    3) Argument 指针头（start + end），输入时在每个 token 上拼接 target span 特征
    4) Group & Hate 分类头，输入为 gold 或预测的 argument span 特征
    """
    def __init__(self, local_model_dir, hidden_size, num_groups, num_hate):
        super().__init__()
        self.bert = BertModel.from_pretrained(local_model_dir)

        # 1) Target 指针头
        self.target_start_head = nn.Linear(hidden_size, 1)
        self.target_end_head   = nn.Linear(hidden_size, 1)

        # 2) Argument 指针头（拼接 target span 特征）
        #    输入维度为 hidden_size * 3：BERT 输出 + target_span_feat (2*H)
        self.arg_start_head = nn.Linear(hidden_size * 3, 1)
        self.arg_end_head   = nn.Linear(hidden_size * 3, 1)

        # 3) 分类头：在 gold arg_span 或预测的 arg_span 上提取特征 (2H)，再做分类
        self.group_fc = nn.Linear(hidden_size * 2, num_groups)  # 多标签
        self.hate_fc  = nn.Linear(hidden_size * 2, num_hate)   # 二分类

        # 损失函数
        self.loss_fct_ce       = nn.CrossEntropyLoss()
        self.loss_fct_ce_hate  = nn.CrossEntropyLoss(reduction="none")
        self.loss_fct_bce      = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        input_ids,
        attention_mask,
        target_start_labels=None,
        target_end_labels=None,
        arg_start_labels=None,
        arg_end_labels=None,
        target_exists_mask=None,
        arg_exists_mask=None,
        group_labels=None,
        hate_labels=None
    ):
        # —— 1) BERT 编码 ——
        B, L = input_ids.size()
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = outputs.last_hidden_state  # (B, L, H)

        # —— 2) Target 指针 logits ——
        # (B, L, 1) -> squeeze -> (B, L)
        tgt_start_logits = self.target_start_head(seq_out).squeeze(-1)  # (B, L)
        tgt_end_logits   = self.target_end_head(seq_out).squeeze(-1)    # (B, L)

        # —— 3) 构造 target span 特征，拼接到每个 token 上，供 Argument 指针使用 ——
        span_feats = []
        for i in range(B):
            if target_exists_mask is not None and target_exists_mask[i]:
                ts = target_start_labels[i].item()
                te = target_end_labels[i].item()
                start_feat = seq_out[i, ts]  # (H,)
                end_feat   = seq_out[i, te]  # (H,)
                span_feats.append(torch.cat([start_feat, end_feat], dim=-1))  # (2H,)
            else:
                # target = NULL 时，使用全零向量占位
                span_feats.append(torch.zeros((seq_out.size(-1) * 2,), device=seq_out.device))
        # (B, 2H)
        span_feats = torch.stack(span_feats, dim=0)

        # (B, 1, 2H) -> expand -> (B, L, 2H)
        span_feats_exp = span_feats.unsqueeze(1).expand(-1, L, -1)
        # 拼接到 seq_out： (B, L, H) + (B, L, 2H) -> (B, L, 3H)
        concat_feats = torch.cat([seq_out, span_feats_exp], dim=-1)

        # —— 4) Argument 指针 logits ——
        arg_start_logits = self.arg_start_head(concat_feats).squeeze(-1)  # (B, L)
        arg_end_logits   = self.arg_end_head(concat_feats).squeeze(-1)    # (B, L)

        # —— 5) 用 gold arg_span 构造 arg_span_feats，用于群体 & 仇恨分类 ——
        arg_span_feats = []
        for i in range(B):
            # arg_exists_mask 恒为 True
            as_idx = arg_start_labels[i].item()
            ae_idx = arg_end_labels[i].item()
            start_feat = seq_out[i, as_idx]
            end_feat   = seq_out[i, ae_idx]
            arg_span_feats.append(torch.cat([start_feat, end_feat], dim=-1))  # (2H,)
        arg_span_feats = torch.stack(arg_span_feats, dim=0)  # (B, 2H)

        group_logits = self.group_fc(arg_span_feats)  # (B, NUM_GROUPS)
        hate_logits  = self.hate_fc(arg_span_feats)   # (B, 2)

        # —— 6) 计算总 Loss ——
        loss = torch.tensor(0.0, device=seq_out.device)

        # 6.1) Target 指针 Loss：仅对存在 target 的样本计算
        if target_start_labels is not None and target_exists_mask is not None and target_exists_mask.any():
            idxs = torch.nonzero(target_exists_mask, as_tuple=False).squeeze(-1)  # (num_with_tgt,)
            loss += self.loss_fct_ce(tgt_start_logits[idxs], target_start_labels[idxs])
            loss += self.loss_fct_ce(tgt_end_logits[idxs],   target_end_labels[idxs])

        # 6.2) Argument 指针 Loss：对所有样本都计算（arg_span 恒合法）
        loss += self.loss_fct_ce(arg_start_logits, arg_start_labels)
        loss += self.loss_fct_ce(arg_end_logits,   arg_end_labels)

        # 6.3) 仇恨分类 Loss：先算 reduction="none" 拿到每个样本的损失 (B,)
        hate_loss_per_example = self.loss_fct_ce_hate(hate_logits, hate_labels)  # (B,)
        hate_loss_mean = hate_loss_per_example.mean()  # scalar
        loss += hate_loss_mean

        # 6.4) 群体分类 Loss：多标签 BCE，先算 (B, NUM_GROUPS)，再对每条样本 sum，最后 mask 仅对 hate 样本计算
        group_loss_per_dim = self.loss_fct_bce(group_logits, group_labels)  # (B, NUM_GROUPS)
        group_loss_per_example = group_loss_per_dim.sum(dim=1)  # (B,)
        # 仅对 hate_labels == 1 的样本保留 group_loss，其它置 0
        group_loss_masked = group_loss_per_example * hate_labels.float()  # (B,)
        group_loss_mean = group_loss_masked.mean()  # scalar
        loss += group_loss_mean

        return {
            "loss": loss,
            "tgt_start_logits": tgt_start_logits,
            "tgt_end_logits":   tgt_end_logits,
            "arg_start_logits": arg_start_logits,
            "arg_end_logits":   arg_end_logits,
            "group_logits":     group_logits,
            "hate_logits":      hate_logits,
        }


def train_full(model, train_loader, num_epochs, lr):
    """
    在整个训练集上训练模型，训练结束后保存权重，并返回训练好的 model。
    """
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        int(WARMUP_PROPORTION * total_steps),
        total_steps
    )
    model.to(DEVICE)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            input_ids          = batch["input_ids"].to(DEVICE)
            attention_mask     = batch["attention_mask"].to(DEVICE)
            tgt_start_labels   = batch["target_start_labels"].to(DEVICE)
            tgt_end_labels     = batch["target_end_labels"].to(DEVICE)
            arg_start_labels   = batch["arg_start_labels"].to(DEVICE)
            arg_end_labels     = batch["arg_end_labels"].to(DEVICE)
            tgt_exists_mask    = batch["target_exists_mask"].to(DEVICE)
            arg_exists_mask    = batch["arg_exists_mask"].to(DEVICE)
            group_labels       = batch["group_labels"].to(DEVICE)
            hate_labels        = batch["hate_labels"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_start_labels=tgt_start_labels,
                target_end_labels=tgt_end_labels,
                arg_start_labels=arg_start_labels,
                arg_end_labels=arg_end_labels,
                target_exists_mask=tgt_exists_mask,
                arg_exists_mask=arg_exists_mask,
                group_labels=group_labels,
                hate_labels=hate_labels
            )
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"[Epoch {epoch+1}/{num_epochs}] 训练集平均 Loss: {avg_train_loss:.4f}")

    # 训练结束后保存模型
    torch.save(model.state_dict(), "final_pointer_model.pt")
    print(">>> 已保存最终模型到 final_pointer_model.pt")
    return model


def evaluate_f1(model, data_loader):
    """
    对 data_loader 中的样本做批量推理，收集预测的 group_pred、hate_pred，与真实标签计算 Macro-F1。
    返回 (group_f1, hate_f1)。
    """
    model.eval()
    all_group_preds = []
    all_group_trues = []
    all_hate_preds  = []
    all_hate_trues  = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids        = batch["input_ids"].to(DEVICE)
            attention_mask   = batch["attention_mask"].to(DEVICE)
            arg_start_labels = batch["arg_start_labels"].to(DEVICE)
            arg_end_labels   = batch["arg_end_labels"].to(DEVICE)
            tgt_exists_mask  = batch["target_exists_mask"].to(DEVICE)
            offsets_batch    = batch["offsets"]
            orig_texts_batch = batch["orig_texts"]
            group_labels     = batch["group_labels"].to(DEVICE)
            hate_labels      = batch["hate_labels"].to(DEVICE)

            B, L = input_ids.size()
            outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)
            seq_out = outputs.last_hidden_state  # (B, L, H)

            # —— 1) 预测 target span（仅取 argmax，不用阈值）——
            tgt_start_logits = model.target_start_head(seq_out).squeeze(-1)  # (B, L)
            tgt_end_logits   = model.target_end_head(seq_out).squeeze(-1)    # (B, L)
            pred_tgt_starts  = torch.argmax(tgt_start_logits, dim=-1)        # (B,)
            pred_tgt_ends    = torch.argmax(tgt_end_logits,   dim=-1)        # (B,)

            # —— 2) 用预测的 target span 构造 span_feats，供 argument 预测 ——
            span_feats = []
            for i in range(B):
                ts = pred_tgt_starts[i].item()
                te = pred_tgt_ends[i].item()
                if ts <= te:
                    start_feat = seq_out[i, ts]
                    end_feat   = seq_out[i, te]
                    span_feats.append(torch.cat([start_feat, end_feat], dim=-1))
                else:
                    span_feats.append(torch.zeros((seq_out.size(-1) * 2,), device=seq_out.device))
            span_feats = torch.stack(span_feats, dim=0)  # (B, 2H)

            span_feats_exp = span_feats.unsqueeze(1).expand(-1, L, -1)  # (B, L, 2H)
            concat_feats = torch.cat([seq_out, span_feats_exp], dim=-1)  # (B, L, 3H)

            # —— 3) 预测 argument span ——
            arg_start_logits = model.arg_start_head(concat_feats).squeeze(-1)  # (B, L)
            arg_end_logits   = model.arg_end_head(concat_feats).squeeze(-1)    # (B, L)
            pred_arg_starts  = torch.argmax(arg_start_logits, dim=-1)         # (B,)
            pred_arg_ends    = torch.argmax(arg_end_logits,   dim=-1)         # (B,)

            # —— 4) 根据预测的 arg_span 依次做群体 & 仇恨分类，收集预测结果 ——
            for i in range(B):
                arg_s = pred_arg_starts[i].item()
                arg_e = pred_arg_ends[i].item()
                # 如果 arg_s > arg_e，则当整句
                if arg_s > arg_e:
                    offsets = offsets_batch[i]
                    last_valid_idx = 0
                    for idx, (s, e) in enumerate(offsets):
                        if e == 0:
                            break
                        last_valid_idx = idx
                    arg_s, arg_e = 0, last_valid_idx

                start_feat = seq_out[i, arg_s]
                end_feat   = seq_out[i, arg_e]
                arg_span_feat = torch.cat([start_feat, end_feat], dim=-1).unsqueeze(0)  # (1, 2H)

                # —— a) Hate 预测 ——
                h_logits = model.hate_fc(arg_span_feat)  # (1, 2)
                pred_hate = torch.argmax(h_logits, dim=-1).item()
                all_hate_preds.append(pred_hate)
                all_hate_trues.append(hate_labels[i].item())

                # —— b) Group 预测（仅当 pred_hate==1 时才做多标签）——
                if pred_hate == 0:
                    # non-hate：不做 group，强制置为全 0 向量
                    grp_pred_vector = [0] * NUM_GROUPS
                else:
                    grp_logits = model.group_fc(arg_span_feat)  # (1, NUM_GROUPS)
                    grp_probs  = torch.sigmoid(grp_logits).squeeze(0).cpu().tolist()
                    # 阈值 0.5，若都 < 0.5，则取最大值的索引
                    grp_pred_vector = [1 if p >= 0.35 else 0 for p in grp_probs]
                    if sum(grp_pred_vector) == 0:
                        max_idx = grp_probs.index(max(grp_probs))
                        grp_pred_vector[max_idx] = 1

                all_group_preds.append(grp_pred_vector)
                all_group_trues.append(group_labels[i].cpu().long().tolist())

    group_f1 = f1_score(all_group_trues, all_group_preds, average="macro", zero_division=0)
    hate_f1  = f1_score(all_hate_trues, all_hate_preds, average="macro", zero_division=0)
    return group_f1, hate_f1


def match_spans(P_start, P_end, mask, threshold):
    """
    在长度 L 的序列中，用贪心方式从 P_start、P_end 概率里挑出所有满足阈值的 (i,j)：
      - 从 i=0 开始遍历，当 P_start[i] >= threshold 且 mask[i]==1 时，
        尝试在 [i..L) 中找到第一个满足 P_end[j]>=threshold 且 mask[j]==1 的 j。
      - 找到后就记录 (i,j)，然后令 i = j+1；否则 i += 1 继续找下一个 start。
    返回所有记录的 span 列表：[(i1,j1), (i2,j2), ...]。
    """
    spans = []
    L = P_start.size(0)
    i = 0
    while i < L:
        # 跳过被 mask 或 start 概率过低的位置
        if mask[i] == 0 or P_start[i].item() < threshold:
            i += 1
            continue

        # P_start[i] ≥ threshold 且 mask[i]==1，往后找第一个满足 end 条件的 j ≥ i
        j = i
        found = False
        while j < L:
            if mask[j] == 1 and P_end[j].item() >= threshold:
                spans.append((i, j))
                i = j + 1  # 匹配成功后跳过整个 span
                found = True
                break
            j += 1

        if not found:
            # 如果这一轮没找到任何合法 end，就从 i+1 继续
            i += 1

    return spans


def match_one_span(P_start, P_end, mask, threshold):
    """
    在长度 L 的序列中，用贪心方式只匹配第一个满足阈值的 (i,j)：
      - 从 i=0 开始遍历，当 P_start[i] >= threshold 且 mask[i]==1 时，
        尝试在 [i..L) 中找到第一个满足 P_end[j]>=threshold 且 mask[j]==1 的 j。
      - 找到后立即返回 (i,j)。如果找不到，返回 None。
    """
    L = P_start.size(0)
    i = 0
    while i < L:
        if mask[i] == 0 or P_start[i].item() < threshold:
            i += 1
            continue
        j = i
        while j < L:
            if mask[j] == 1 and P_end[j].item() >= threshold:
                return (i, j)
            j += 1
        i += 1
    return None


def inference_greedy(
    model, tokenizer,
    text,
    tgt_threshold=0.15,
    arg_threshold=0.15,
    max_len=128
):
    """
    对单条 content(text) 做一次推理，使用贪心算法从 start/end 概率里匹配多个 target span，
    然后对每个 target span，只匹配一个最合理的 argument span，最后做分类。
    如果从句子中找不到任何 target，就只提取一个最有可能的论点 span（tar=NULL）。
    返回一个列表，每个元素都是 “target | argument | group | hate” 样式的字符串，其中
    只有最后一个四元组末尾加 “ [END]”，前面都只留两个空格，不加 [END]。
    """
    model.eval()
    with torch.no_grad():
        # 1) Tokenize
        enc = tokenizer(
            text,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_offsets_mapping=True
        )
        input_ids  = enc["input_ids"].to(DEVICE)          # (1, L)
        attn_mask  = enc["attention_mask"].to(DEVICE)     # (1, L)
        offsets    = enc["offset_mapping"].squeeze(0).tolist()  # List[(char_s, char_e)]
        seq_out    = model.bert(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state.squeeze(0)  # (L, H)
        L, H       = seq_out.size()

        # 2) 预测 target 的 start/end logits → 转 softmax 概率
        tgt_start_logits = model.target_start_head(seq_out).squeeze(-1)  # (L,)
        tgt_end_logits   = model.target_end_head(seq_out).squeeze(-1)    # (L,)

        # 把被 mask 掉的位置 logits 置为 -1e9，保证它们的 softmax 概率 ~ 0
        mask_float = (attn_mask.squeeze(0) == 0).float() * (-1e9)  # (L,)
        s_logits = tgt_start_logits + mask_float  # (L,)
        e_logits = tgt_end_logits   + mask_float  # (L,)

        P_start_tgt = F.softmax(s_logits, dim=0)  # (L,)
        P_end_tgt   = F.softmax(e_logits, dim=0)  # (L,)

        # 3) 贪心匹配出所有 target span
        tgt_mask = attn_mask.squeeze(0)  # (L,), 0/1
        tgt_spans = match_spans(P_start_tgt, P_end_tgt, tgt_mask, tgt_threshold)

        quads = []

        if not tgt_spans:
            # —— 当找不到任何 target 时 ——
            # 只提取一句话里最有可能的一个论点 span（arg）
            zero_tgt_feat = torch.zeros((H * 2,), device=seq_out.device)  # (2H,)
            zero_tgt_feat_exp = zero_tgt_feat.unsqueeze(0).expand(L, -1)  # (L, 2H)
            arg_input_feats = torch.cat([seq_out, zero_tgt_feat_exp], dim=-1)  # (L, 3H)

            # 预测 argument start/end
            arg_start_logits = model.arg_start_head(arg_input_feats).squeeze(-1)  # (L,)
            arg_end_logits   = model.arg_end_head(arg_input_feats).squeeze(-1)    # (L,)

            a_s_logits = arg_start_logits + mask_float
            a_e_logits = arg_end_logits   + mask_float
            P_start_arg = F.softmax(a_s_logits, dim=0)  # (L,)
            P_end_arg   = F.softmax(a_e_logits, dim=0)  # (L,)

            # 只匹配一个最有可能的论点 span
            one_arg = match_one_span(P_start_arg, P_end_arg, tgt_mask, arg_threshold)
            if one_arg is None:
                # 完全没有论点 → "NULL [END]"
                quads.append("NULL [END]")
            else:
                a_u, a_v = one_arg
                char_su = offsets[a_u][0]
                char_ev = offsets[a_v][1]
                arg_text = text[char_su:char_ev] if char_su < char_ev else ""
                if not arg_text.strip():
                    # 假如切出来是空，就当作没有论点
                    quads.append("NULL [END]")
                else:
                    # 做 hate 分类
                    a_start_feat = seq_out[a_u]
                    a_end_feat   = seq_out[a_v]
                    arg_feat     = torch.cat([a_start_feat, a_end_feat], dim=-1).unsqueeze(0)  # (1, 2H)
                    h_logits     = model.hate_fc(arg_feat)  # (1, 2)
                    pred_hate    = torch.argmax(h_logits, dim=-1).item()
                    hate_str     = "hate" if pred_hate == 1 else "non-hate"

                    # 做 group 分类（仅当 pred_hate==1 时）
                    if pred_hate == 0:
                        group_str = "non-hate"
                    else:
                        g_logits   = model.group_fc(arg_feat)  # (1, NUM_GROUPS)
                        g_probs    = torch.sigmoid(g_logits).squeeze(0).cpu().tolist()  # (NUM_GROUPS,)
                        g_pred_vec = [1 if p >= 0.5 else 0 for p in g_probs]
                        if sum(g_pred_vec) == 0:
                            max_idx   = g_probs.index(max(g_probs))
                            group_str = ID2GROUP[max_idx]
                        else:
                            chosen = [ID2GROUP[idx] for idx, flag in enumerate(g_pred_vec) if flag == 1]
                            group_str = ",".join(chosen)

                    # 由于 target 为 NULL，固定四元组格式："NULL | arg_text | group_str | hate_str"
                    quad = f"NULL | {arg_text} | {group_str} | {hate_str}"
                    quads.append(quad)

        else:
            # —— 对每个 target span，只匹配一个最合理的 argument ——
            for (t_i, t_j) in tgt_spans:
                char_si = offsets[t_i][0]
                char_ej = offsets[t_j][1]
                tgt_text = text[char_si:char_ej] if char_si < char_ej else ""
                if not tgt_text.strip():
                    continue  # 如果切出来全是空或标点，就跳过

                # 用预测到的 target span 构造 特征，拼到 seq_out：(L,H) + (L,2H)->(L,3H)
                t_start_feat = seq_out[t_i]  # (H,)
                t_end_feat   = seq_out[t_j]  # (H,)
                tgt_feat     = torch.cat([t_start_feat, t_end_feat], dim=-1)  # (2H,)
                tgt_feat_exp = tgt_feat.unsqueeze(0).expand(L, -1)            # (L, 2H)
                arg_input_feats = torch.cat([seq_out, tgt_feat_exp], dim=-1)  # (L, 3H)

                # 预测 argument 的 start/end logits → 转 softmax 概率
                arg_start_logits = model.arg_start_head(arg_input_feats).squeeze(-1)  # (L,)
                arg_end_logits   = model.arg_end_head(arg_input_feats).squeeze(-1)    # (L,)

                a_s_logits = arg_start_logits + mask_float
                a_e_logits = arg_end_logits   + mask_float
                P_start_arg = F.softmax(a_s_logits, dim=0)  # (L,)
                P_end_arg   = F.softmax(a_e_logits, dim=0)  # (L,)

                # 贪心只匹配 “第一个满足阈值的 start + 它后面最近的满足阈值的 end”
                one_arg = match_one_span(P_start_arg, P_end_arg, tgt_mask, arg_threshold)

                if one_arg is None:
                    # 找不到 argument → 按照逻辑当作 argument="" 且 non-hate
                    arg_text = ""
                    hate_str = "non-hate"
                    group_str = "non-hate"
                    quad = f"{tgt_text} | {arg_text} | {group_str} | {hate_str}"
                    quads.append(quad)
                else:
                    a_u, a_v = one_arg
                    char_su = offsets[a_u][0]
                    char_ev = offsets[a_v][1]
                    arg_text = text[char_su:char_ev] if char_su < char_ev else ""
                    if not arg_text.strip():
                        # 如果切出来的 argument 全是空或标点，就按 non-hate 处理
                        hate_str = "non-hate"
                        group_str = "non-hate"
                        quad = f"{tgt_text} | {arg_text} | {group_str} | {hate_str}"
                        quads.append(quad)
                    else:
                        # 做 hate 分类
                        a_start_feat = seq_out[a_u]
                        a_end_feat   = seq_out[a_v]
                        arg_feat     = torch.cat([a_start_feat, a_end_feat], dim=-1).unsqueeze(0)  # (1, 2H)
                        h_logits     = model.hate_fc(arg_feat)  # (1, 2)
                        pred_hate    = torch.argmax(h_logits, dim=-1).item()
                        hate_str     = "hate" if pred_hate == 1 else "non-hate"

                        # 做 group 分类（仅当 pred_hate==1 时）
                        if pred_hate == 0:
                            group_str = "non-hate"
                        else:
                            g_logits   = model.group_fc(arg_feat)  # (1, NUM_GROUPS)
                            g_probs    = torch.sigmoid(g_logits).squeeze(0).cpu().tolist()  # (NUM_GROUPS,)
                            g_pred_vec = [1 if p >= 0.35 else 0 for p in g_probs]
                            if sum(g_pred_vec) == 0:
                                max_idx   = g_probs.index(max(g_probs))
                                group_str = ID2GROUP[max_idx]
                            else:
                                chosen = [ID2GROUP[idx] for idx, flag in enumerate(g_pred_vec) if flag == 1]
                                group_str = ",".join(chosen)

                        quad = f"{tgt_text} | {arg_text} | {group_str} | {hate_str}"
                        quads.append(quad)

        # —— 拼接输出格式 ——
        if not quads:
            # 理论上不会走到这里，上面已经处理了所有情况
            return ["NULL [END]"]

        # 只有最后一个四元组要加" [END]"，
        final_quads = []
        for idx, q in enumerate(quads):
            if idx == len(quads) - 1:
                final_quads.append(q + " [END]")

        # 用 " [SEP] "（前两个空格 + [SEP] + 一个空格）把所有片段串起来
        line = " [SEP] ".join(final_quads)
        return [line]


def inference(
    model,
    tokenizer,
    test_json,
    output_path="predictions.txt",
    tgt_threshold: float = 0.15,
    arg_threshold: float = 0.15,
    max_len: int = 128
):
    """
    对测试集做推理，将结果写入 output_path。每条 content 输出一行，格式如下：
      - tar1 | arg1 | group1 | hate1  [SEP] tar2 | arg2 | group2 | hate2  [SEP] ... tarN | argN | groupN | hateN [END]
      - 如果完全没匹配到任何四元组，就输出 "NULL [END]"
    """
    model.to(DEVICE)
    model.eval()

    with open(output_path, "w", encoding="utf-8") as fout:
        data = json.load(open(test_json, "r", encoding="utf-8"))
        for example in data:
            text = example["content"]
            quads = inference_greedy(
                model, tokenizer, text,
                tgt_threshold=tgt_threshold,
                arg_threshold=arg_threshold,
                max_len=max_len
            )
            # inference_greedy 本身会返回列表（长度为1），所以直接写 quads[0]
            fout.write(quads[0] + "\n")

    print(f"推理完成，结果已写入：{output_path}")


if __name__ == "__main__":
    # 1) 配置路径和参数
    train_json = "data/train.json"
    test_json  = "data/test1.json"
    LOCAL_ROBERTA_DIR = "./roberta-wwm-ext-as-bert"
    NUM_EPOCHS = 4
    LEARNING_RATE = 2e-5

    # 2) 加载 tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(LOCAL_ROBERTA_DIR)

    # 3) 构建完整数据集和 DataLoader
    full_dataset = HateSpeechDataset(train_json, tokenizer, max_len=MAX_LEN, mode="train")
    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # 4) 初始化模型（hidden_size=768，对应 BERT-base）
    model = HateSpeechPointerModel(
        local_model_dir=LOCAL_ROBERTA_DIR,
        hidden_size=768,
        num_groups=NUM_GROUPS,
        num_hate=2
    )
    model.to(DEVICE)

    # 5) 在训练集上训练
    model = train_full(model, train_loader, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE)

    # 6) 评估：直接用 full_dataset 做一次评测
    eval_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    group_f1, hate_f1 = evaluate_f1(model, eval_loader)
    print("=== 最终评估 F1 分数（在训练集上） ===")
    print(f"群体分类 Macro‐F1: {group_f1:.4f}")
    print(f"仇恨/非仇恨 Macro‐F1: {hate_f1:.4f}")

    # 7) 加载最优模型并对测试集做推理
    model.load_state_dict(torch.load("final_pointer_model.pt", map_location=DEVICE))
    inference(
        model, tokenizer, test_json,
        output_path="predictions.txt",
        tgt_threshold=0.15,
        arg_threshold=0.15,
        max_len=MAX_LEN
    )
