from pathlib import Path

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])
ROOT = Path.home()
DATA_DIR = Path.home() / ".local" / "orcbench"
MODEL_DIR = DATA_DIR / "models"
URL = "https://rcs.uwaterloo.ca/~ryan/files/models"


# Model Creation Globals
RAW_DATA_URL = "https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/azurefunctions_dataset2019/azurefunctions-dataset2019.tar.xz"

RAW_DATA_DIR = DATA_DIR / "raw_data"
POISSON_SEED = 1337
NUM_DATA_FILES = 40
DEFAULT_DAYS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

STARTTIME = 720
CAPTURE_WINDOW = 30
INVOCATION_CUTOFF = 10


# Logger
FORMAT = "[%(asctime)s] %(message)s"
