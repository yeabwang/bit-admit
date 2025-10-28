import logging
import os
from datetime import datetime

from from_root import from_root
from rich.console import Console
from rich.logging import RichHandler

# logging file setup
log_dir = "logs"
log_file_name = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"
log_dir_path = os.path.join(from_root(), log_dir)
log_file_path = os.path.join(log_dir_path, log_file_name)

os.makedirs(log_dir_path, exist_ok=True)

console = Console()

file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(
    logging.Formatter("[ %(levelname)s ] - %(asctime)s - %(name)s - %(message)s")
)

rich_handler = RichHandler(
    console=console,
    rich_tracebacks=True,
    markup=False,
    show_time=False,
    show_path=False,
)
rich_handler.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[file_handler, rich_handler],
)
