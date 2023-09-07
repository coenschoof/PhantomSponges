# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = '8.0.172'

from local_yolos.yolov8.ultralytics.models import RTDETR, SAM, YOLO
from local_yolos.yolov8.ultralytics.models.fastsam import FastSAM
from local_yolos.yolov8.ultralytics.models.nas import NAS
from local_yolos.yolov8.ultralytics.utils import SETTINGS as settings
from local_yolos.yolov8.ultralytics.utils.checks import check_yolo as checks
from local_yolos.yolov8.ultralytics.utils.downloads import download

__all__ = '__version__', 'YOLO', 'NAS', 'SAM', 'FastSAM', 'RTDETR', 'checks', 'download', 'settings'
