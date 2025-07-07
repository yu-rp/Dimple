import logging
import os
from transformers.utils import logging as hf_logging
from accelerate import Accelerator

_main_logger_initialized = False

def setup_logging_for_main_process():
    """
    Call this at the start of the script (only once), before importing other modules that use logging.
    """
    global _main_logger_initialized
    accelerator = Accelerator()
    if accelerator.is_main_process and not _main_logger_initialized:
        logging.basicConfig(
            level=logging.INFO,
            # level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        # hf_logging.set_verbosity_debug()
        hf_logging.set_verbosity_info()
        _main_logger_initialized = True
    else:
        logging.getLogger().setLevel(logging.ERROR)
        hf_logging.set_verbosity_error()


def add_file_handler(logger: logging.Logger, output_folder: str, filename: str = "training_log.txt"):
    """
    Add a file handler to the given logger. Should be called after OUTPUT_FOLDER is known.
    Safe to call multiple times – it won't add duplicate handlers.
    """

    log_path = os.path.join(output_folder, filename)

    # Avoid adding the same file handler multiple times
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_path) for h in logger.handlers):
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        file_handler.setFormatter(formatter)
        # file_handler.setLevel(logging.DEBUG) 
        file_handler.setLevel(logging.INFO) 
        logger.addHandler(file_handler)
