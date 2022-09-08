from pathlib import Path

TM_DIR = Path(__file__).parent
RESOURCES_DIR = TM_DIR / 'test' / 'resources'
PROJECT_DIR = TM_DIR.parent.parent.parent.parent / 'master_thesis'
URI_PREFIX = "http://localhost:41193/"

PATIENCE = 3
MIN_DELTA = 0.01
