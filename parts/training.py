import torch
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from .tokenzer import get_tokenizer
from .models import get_model
from .dataset_loader import prepare_dataset
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
import logging

logger = logging.getLogger(__name__)
model_id = "klue/roberta-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokenize_function(examples):
    return get_tokenizer(model_id)(
        examples["title"], padding="max_length", truncation=True
    )


def make_dataloader(dataset, batch_size, shuffle=True):
    dataset = dataset.map(tokenize_function, batched=True).with_format("torch")

    # 데이터셋에 토큰화 수행
    dataset = dataset.rename_column("label", "labels")
    dataset = dataset.remove_columns(column_names=["title"])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def do_training():
    # 데이터셋 취득
    train_, valid_, test_ = prepare_dataset("klue", "ynat")

    # 모델과 토크나이저 불러오기
    model = get_model(model_id, len(train_.features["label"].names))

    model.to(device)

    train_dataloader = make_dataloader(train_, batch_size=8, shuffle=True)
    valid_dataloader = make_dataloader(valid_, batch_size=8, shuffle=True)
    test_dataloader = make_dataloader(test_, batch_size=8, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 1
    # 학습 루프
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1} / {num_epochs}")
        train_loss = train_epoch(
            model=model, data_loader=train_dataloader, optimizer=optimizer
        )
        logger.info(f"Training loss: {train_loss}")
        valid_loss, valid_accuracy = evaluate(model, valid_dataloader)
        logger.info(f"validation loss: {valid_loss}, accuracy: {valid_accuracy}")

    # 테스트
    _, test_accuracy = evaluate(model, test_dataloader)
    logger.info(f"Test accuracy: {test_accuracy}")


# 훈련
def train_epoch(model, data_loader, optimizer):
    model.train()
    scaler = GradScaler("cuda")
    total_loss = 0

    for batch in tqdm(data_loader):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with autocast("cuda"):
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        # 모델업데이트
        # GPU 활용도가 올라가고, 큰 모델 학습 시 OOM(Out Of Memory) 위험이 줄어듬.
        scaler.scale(loss).backward()
        scaler.step(optimizer=optimizer)
        scaler.update()
        total_loss += loss.detach().item()
    return total_loss / len(data_loader)


# 검증
def evaluate(model, data_loader):
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    # 기울기추척 (연산그래프 작성) 비활성.
    with torch.no_grad():
        for batch in tqdm(data_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast("cuda"):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits
                loss = outputs.loss
                total_loss += loss.detach().item()

            preds = torch.argmax(logits, dim=-1)
            predictions.extend(
                preds.cpu().numpy()
            )  # gpu에 있는 값을 cpu쪽으로 복사해옴
            true_labels.extend(labels.cpu().numpy())
        avg_loss = total_loss / len(data_loader)
        accuracy = np.mean(np.array(predictions) == np.array(true_labels))
        return avg_loss, accuracy
