from ai_lang.ai_lang import process_ai_file_sync
from loguru import logger
import sys

try:
    results = process_ai_file_sync("test.ai")
    logger.info(f"Successfully processed {len(results)} requests")
except Exception as e:
    logger.exception(f"Failed to process file: {e}")
    sys.exit(1)
