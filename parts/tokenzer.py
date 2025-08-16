from transformers import AutoTokenizer
import logging
from typing import Union

logger = logging.getLogger(__name__)
# model_id = "klue/roberta-base"


def get_tokenizer(model_id) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_id)


def get_tokens(texts: Union[str, list[str]]) -> "tokenized_data":
    tokenizer = get_tokenizer()
    tokenized = tokenizer(texts)
    # print(tokenized)

    # print(tokenizer.convert_ids_to_tokens(tokenized["input_ids"]))

    # print(tokenizer.decode(tokenized["input_ids"]))

    # print(tokenizer.decode(tokenized["input_ids"], skip_special_tokens=True))

    # batch_tokenized = tokenizer(["첫번째 문장", "두번째 문장"])

    # logger.info(batch_tokenized)

    logger.info(f"input:{texts}, output:{tokenized}")

    return tokenized
