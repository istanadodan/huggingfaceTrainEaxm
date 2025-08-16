from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


def get_model(model_id: str = None, num_labels: int = 0) -> "Model":
    return AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=num_labels
    )
