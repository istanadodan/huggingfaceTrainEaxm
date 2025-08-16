from config.logging import setup_logging, logging

# from parts.tokenzer import get_tokens
from parts.dataset_loader import get_dataset, prepare_dataset, do_training_using_Trainer
from parts.training import do_training

logger = setup_logging(level=logging.INFO)


# logger.info(get_tokens("토큰나이저"))

# do_training_using_Trainer()

do_training()
