import importlib
import sys

from local_yolos.yolov8.ultralytics.utils import LOGGER

# Set modules in sys.modules under their old name
sys.modules['ultralytics.yolo.engine'] = importlib.import_module('ultralytics.engine')

LOGGER.warning("WARNING ⚠️ 'ultralytics.yolo.engine' is deprecated since '8.0.136' and will be removed in '8.1.0'. "
               "Please use 'ultralytics.engine' instead.")
