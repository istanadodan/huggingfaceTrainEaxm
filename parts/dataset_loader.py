from datasets import load_dataset, Dataset
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def get_dataset() -> "Dataset":
    klue_mrc_dataset = load_dataset("klue", "mrc", split="train")

    logger.info(f"dataset: {klue_mrc_dataset}")

    my_dict = {
        "a": [1, 2, 3],
        "b": [4, 4, 5],
    }
    dataset = Dataset.from_dict(my_dict)

    logger.info(f"{dataset}")

    df = pd.DataFrame(
        {
            "a": [
                1,
                2,
            ]
        }
    )
    dataset = Dataset.from_pandas(df)

    logger.info(f"{dataset}")


def prepare_dataset(path: str, name: str):
    klue_tc_train = load_dataset(path, name, split="train")
    klue_tc_valid = load_dataset(path, name, split="validation")
    # logger.info(klue_tc_train)

    klue_tc_train = klue_tc_train.remove_columns(["guid", "url", "date"])
    klue_tc_valid = klue_tc_valid.remove_columns(["guid", "url", "date"])
    logger.info(klue_tc_train)

    def make_str_label(batch):
        batch["label_str"] = klue_tc_train.features["label"].int2str(batch["label"])
        return batch

    klue_tc_train = klue_tc_train.map(make_str_label, batched=True, batch_size=1000)

    # 학습/검증/테스트 데이터셋 분할 - 만개만 추출
    train_dataset = klue_tc_train.train_test_split(
        test_size=10000, shuffle=True, seed=42
    )["test"]
    dataset = klue_tc_valid.train_test_split(test_size=1000, shuffle=True, seed=42)
    test_dataset = dataset["test"]
    valid_dataset = dataset["train"].train_test_split(
        test_size=1000, shuffle=True, seed=42
    )["test"]

    logger.info(f"{klue_tc_train[0]}")
    return train_dataset, valid_dataset, test_dataset


from typing import Any


def do_training_using_Trainer():
    import torch
    import numpy as np
    from transformers import (
        Trainer,
        TrainingArguments,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )
    from .tokenzer import get_tokens, get_tokenizer
    from .models import get_model

    model_id = "klue/roberta-base"

    # 데이터셋 취득
    train_, valid_, test_ = prepare_dataset("klue", "ynat")

    def tokenize_function(examples):
        return get_tokenizer(model_id)(
            examples["title"], padding="max_length", truncation=True
        )

    train_dataset = train_.map(tokenize_function, batched=True)
    valid_dataset = valid_.map(tokenize_function, batched=True)
    test_dataset = test_.map(tokenize_function, batched=True)

    model = get_model(model_id, len(train_dataset.features["label"].names))
    # model = get_model(model_id, len(train_dataset[next(iter(train_dataset))]))
    tokenizer = get_tokenizer(model_id)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="steps",
        eval_steps=30,
        learning_rate=5e-3,
        push_to_hub=False,
    )

    def compute_metrics(eval_pred):
        # from sklearn.metrics import accuracy_score, f1_score

        logits, labels = eval_pred
        preditions = np.argmax(logits, axis=-1)
        logger.info(f"preditions:{preditions}, lables:{labels}")
        # preditions = logits.argmax(axis=-1)
        return {"accuracy": (preditions == labels).mean()}
        # return {
        #     "accuracy": accuracy_score(labels, preds),
        #     "f1": f1_score(labels, preds, average="weighted")
        # }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    eval_results = trainer.evaluate(test_dataset)
    logger.info(eval_results)
