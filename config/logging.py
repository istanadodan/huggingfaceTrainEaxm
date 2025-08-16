import logging
from pathlib import Path
import datetime


# 로그출력 포맷 작성
log_format = "%(asctime)s|%(levelname)s|[%(name)20s]- %(message)s"
datetime_format = "%Y-%m-%d %H-%M-%S"
date_format = "%Y-%m-%d"
root_dir = Path("./logs")


def setup_logging(level=logging.DEBUG) -> logging.Logger:
    from logging.handlers import RotatingFileHandler

    # 로깅폴더 초기화
    root_dir.mkdir(exist_ok=True)

    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    file_log_handler = RotatingFileHandler(
        filename=str((root_dir / _log_filename()).resolve()),
        backupCount=30,
        maxBytes=10 * 1024 * 1024,  # 10MB
        encoding="utf-8",
    )
    file_log_handler.setFormatter(_logging_formatter())

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(_logging_formatter())

    # 출력 설정; 기존에 추가된 핸들러가 있다면 제거 (중복 로깅 방지)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.addHandler(file_log_handler)
    root_logger.addHandler(console_handler)

    # (Safety Net) `sys.excepthook`: 미들웨어가 잡지 못하는 영역(백그라운드, 시작/종료 등)에서 발생하는 예외를 놓치지 않고
    #   로그로 남기기 위한 안전망(Safety Net)으로 함께 사용합니다.
    # Uncaught Exception에 대한 로깅 핸들러 설정
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # KeyboardInterrupt는 기본 동작을 따르도록 함
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        root_logger.critical(
            "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
        )

    import sys

    sys.excepthook = handle_exception

    return root_logger


def _logging_formatter() -> logging.Formatter:
    return logging.Formatter(
        log_format,
        datefmt=datetime_format,
    )


def _log_filename() -> str:
    return f"prj_{datetime.datetime.now().strftime(date_format)}.log"
