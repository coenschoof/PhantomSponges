# Ultralytics YOLO 🚀, AGPL-3.0 license

from local_yolos.yolov8.ultralytics.models.yolo.segment import SegmentationValidator
from local_yolos.yolov8.ultralytics.utils.metrics import SegmentMetrics


class FastSAMValidator(SegmentationValidator):

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize SegmentationValidator and set task to 'segment', metrics to SegmentMetrics."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = 'segment'
        self.args.plots = False  # disable ConfusionMatrix and other plots to avoid errors
        self.metrics = SegmentMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
