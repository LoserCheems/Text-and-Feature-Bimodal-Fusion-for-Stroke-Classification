from modeling.tokenization_cheems import CheemsTokenizer
from modeling.configuration_cheems import CheemsConfig
from modeling.textfeature_classification_modeing import CheemsForSequenceClassification

from scripts.Dataset import HealthCare_Dataset

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import os
import logging
import argparse
import transformers
transformers.logging.set_verbosity(logging.ERROR)

from transformers.models.bert.modeling_bert import BertConfig
from transformers.models.bert.modeling_bert import BertForSequenceClassification

from transformers.models.xlnet.modeling_xlnet import XLNetConfig
from transformers.models.xlnet.modeling_xlnet import XLNetForSequenceClassification

from transformers.models.llama.modeling_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForSequenceClassification

from transformers.models.phi.modeling_phi import PhiConfig
from transformers.models.phi.modeling_phi import PhiForSequenceClassification

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def evaluate_acc_precision_recall_f1(logits, labels):  
    preds = torch.argmax(logits, dim=1)
    labels = labels.cpu().numpy()
    preds = preds.cpu().numpy()
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    return acc, precision, recall, f1


# 训练函数
def trainer(
    model: torch.nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    device: torch.device,
    epochs: int,
    logger: logging.Logger, 
    pass_epoch: int = 0
) -> None:
    torch.manual_seed(233)
    model.to(device)
    # 如果是Linux, 编译模型
    if os.name == 'posix':
        torch.compile(model)
    # 记录模型参数数量
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(model)
    logger.info(f"Total Parameters: {num_parameters}")
    train_loss = []

    for epoch in range(epochs):
        model.train()
        # 如果epoch小于pass_epoch, 则只更新学习率
        if epoch < pass_epoch:
            train_loader_len = len(train_loader)
            for step in range(train_loader_len):
                scheduler.step()
                print(f"Epoch [{epoch+1}/{epochs}], Step [{step+1}/{train_loader_len}], lr: {scheduler.get_last_lr()[0]:.8f}")
            continue
        for step, batch in enumerate(train_loader):
            # 获取数据
            if model.__class__.__name__ == "CheemsForSequenceClassification":
                inputs = {
                    "text_ids": batch["text_ids"].to(device),
                    "text_attention_mask": batch["text_attention_mask"].to(device),
                    "feature_ids": batch["feature_ids"].to(device),
                    "feature_attention_mask": batch["feature_attention_mask"].to(device),
                    "labels": batch["label"].to(device)
                }
            else:
                inputs = {
                    "input_ids" : batch["text_ids"].to(device),
                    "attention_mask" : batch["text_attention_mask"].to(device),
                    "labels" : batch["label"].to(device)
                }
            
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            outputs = model(**inputs)
            loss = outputs.loss
            # 反向传播
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # 更新参数
            optimizer.step()
            # 更新学习率
            scheduler.step()
            # 记录损失
            train_loss.append(loss.item())

            if (step+1) % (len(train_loader)//10) == 0:
                train_loss = sum(train_loss) / len(train_loss)
                logger.info(f"Epoch [{epoch+1}/{epochs}], Step [{step+1}/{len(train_loader)}], lr: {scheduler.get_last_lr()[0]:.8f}, Loss: {train_loss}")

                train_loss = []

        # 模型评估 ACC
        if (epoch+1) % 4 == 0:
            model.eval()
            accs = []
            precisions = []
            recalls = []
            f1s = []
            for step, batch in enumerate(eval_loader):
                if model.__class__.__name__ == "CheemsForSequenceClassification":
                    inputs = {
                        "text_ids": batch["text_ids"].to(device),
                        "text_attention_mask": batch["text_attention_mask"].to(device),
                        "feature_ids": batch["feature_ids"].to(device),
                        "feature_attention_mask": batch["feature_attention_mask"].to(device),
                        "labels": batch["label"].to(device)
                    }
                else:
                    inputs = {
                        "input_ids" : batch["text_ids"].to(device),
                        "attention_mask" : batch["text_attention_mask"].to(device),
                        "labels" : batch["label"].to(device)
                    }

                outputs = model(**inputs)
                logits = outputs.logits
                labels = inputs["labels"]
                acc, precision, recall, f1 = evaluate_acc_precision_recall_f1(logits, labels)
                accs.append(acc)
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)
            acc = sum(accs) / len(accs)
            precision = sum(precisions) / len(precisions)
            recall = sum(recalls) / len(recalls)
            f1 = sum(f1s) / len(f1s)
            logger.info(f"\nEpoch [{epoch+1}/{epochs}], Eval ACC: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n")
            
            # 保存模型
            model.save_pretrained(f"./models/{model.__class__.__name__}_epoch_{epoch+1}")


def set_logger(model_name: str):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(f"./logs/train_{model_name}.log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def get_optimizer_and_scheduler(
    model: torch.nn.Module,
    lrate: callable,
):
    no_decay = ["bias", "LayerNorm.weight"]
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and not p.requires_grad == False],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and not p.requires_grad == False],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1, betas=(0.9, 0.98), eps=1e-8)
    scheduler = LambdaLR(optimizer, lr_lambda=lrate)
    return optimizer, scheduler

if __name__ == "__main__":
    
  
    parser = argparse.ArgumentParser()
    # lr
    parser.add_argument("--lr", type=float, default=2e-5)
    # epochs
    parser.add_argument("--epochs", type=int, default=64)
    args = parser.parse_args()
    
    # 加载数据集
    tokenizer = CheemsTokenizer('./modeling/cheems_tokenizer.model')
    train_dataset = HealthCare_Dataset('./data/healthcare_stroke', 'train', tokenizer, max_length=128)
    eval_dataset = HealthCare_Dataset('./data/healthcare_stroke', 'eval', tokenizer, max_length=128)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 学习率
    def lrate(
        step: int
    ) -> float:
        if step == 0:
            return 0.0
        return args.lr

    # Cheems模型
    config = CheemsConfig()
    config.text_vocab_size = tokenizer.vocab_size
    config.feature_vocab_size = 2
    config.num_labels = 2
    model = CheemsForSequenceClassification(config)
    optimizer, scheduler = get_optimizer_and_scheduler(model, lrate)
    logger = set_logger(model.__class__.__name__)

    trainer(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        logger=logger
    )

    # Bert模型
    config = BertConfig()
    config.vocab_size = tokenizer.vocab_size
    config.hidden_size = 1024
    config.intermediate_size = 1024*4
    config.num_attention_heads = 16
    config.num_labels = 2
    model = BertForSequenceClassification(config)
    optimizer, scheduler = get_optimizer_and_scheduler(model, lrate)
    logger = set_logger(model.__class__.__name__)

    trainer(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        logger=logger
    )

    # XLNet模型
    config = XLNetConfig()
    config.vocab_size = tokenizer.vocab_size
    config.d_model = 1024
    config.d_inner = 1024*4
    config.num_labels = 2
    model = XLNetForSequenceClassification(config)
    optimizer, scheduler = get_optimizer_and_scheduler(model, lrate)
    logger = set_logger(model.__class__.__name__)

    trainer(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        logger=logger
    )

    # Llama模型
    config = LlamaConfig()
    config.vocab_size = tokenizer.vocab_size
    config.pad_token_id = tokenizer.pad_token_id
    config.num_hidden_layers = 12
    config.hidden_size = 1024
    config.intermediate_size = 1024*4
    config.num_labels = 2
    model = LlamaForSequenceClassification(config)
    optimizer, scheduler = get_optimizer_and_scheduler(model, lrate)
    logger = set_logger(model.__class__.__name__)

    trainer(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        logger=logger
    )

    # Phi模型
    config = PhiConfig()
    config.vocab_size = tokenizer.vocab_size
    config.pad_token_id = tokenizer.pad_token_id
    config.num_hidden_layers = 12
    config.hidden_size = 1024
    config.intermediate_size = 1024*4
    config.num_labels = 2
    model = PhiForSequenceClassification(config)
    optimizer, scheduler = get_optimizer_and_scheduler(model, lrate)
    logger = set_logger(model.__class__.__name__)

    trainer(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        logger=logger
    )
