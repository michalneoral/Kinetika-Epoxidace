import logging
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.WARNING, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

_LOGGER = logging.getLogger('niceGUI')
# _LOGGER.setLevel(logging.ERROR)
