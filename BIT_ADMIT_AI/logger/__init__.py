import logging
import os
from datetime import datetime
from from_root import from_root

# logging file setup
log_dir = "logs"
log_file_name = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"
log_dir_path = os.path.join(from_root(), log_dir)
log_file_path = os.path.join(log_dir_path, log_file_name)

os.makedirs(log_dir_path, exist_ok=True)

logging.basicConfig(
    filename=log_file_path,
    format="[ %(levelname)s ] - %(asctime)s - %(name)s - %(message)s",
    level=logging.DEBUG,
)
