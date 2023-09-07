from local_yolos.yolov8.ultralytics.models.yolo.detect import DetectionPredictor
from local_yolos.yolov8.ultralytics.utils import ops
from local_yolos.yolov8.ultralytics.nn.autobackend import AutoBackend
from local_yolos.yolov8.ultralytics.utils.torch_utils import select_device

class CustomPredictor(DetectionPredictor): 
    def postprocess(self, preds):
        #print(preds[0].shape)
        """Post-processes predictions for an image and returns them."""
        post_nms_preds = ops.non_max_suppression(prediction = preds,
                                        conf_thres = 0.25,
                                        iou_thres = 0.45,
                                        max_det=300)
        return post_nms_preds
        
    def __call__(self, source=None, model=None, stream=False, *args, **kwargs):
        return list(self.stream_inference(source, model, *args, **kwargs))  # merge list of Result into one
    
    
    def stream_inference(self, source=None, model=None, *args, **kwargs):
        """Streams real-time inference on camera feed and saves results to file."""

        # Setup model
        if not self.model:
            self.setup_model(model, verbose=False)

        # Setup source every time predict is called
        self.setup_source(source if source is not None else self.args.source)

        # Warmup model
        if not self.done_warmup:
            self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
            self.done_warmup = True

        self.seen, self.windows, self.batch, profilers = 0, [], None, (ops.Profile(), ops.Profile(), ops.Profile())
        self.run_callbacks('on_predict_start')
        for batch in self.dataset:
            self.run_callbacks('on_predict_batch_start')
            self.batch = batch
            _, im0s, _, _ = batch

            # Preprocess
            with profilers[0]:
                im = self.preprocess(im0s)

            # Inference
            with profilers[1]:
                preds = self.inference(im, *args, **kwargs)

            # Postprocess
            with profilers[2]:
                self.results = self.postprocess(preds)
            return preds
        

    def setup_model(self, model, verbose=True):
        """Initialize YOLO model with given parameters and set it to evaluation mode."""
        self.model = AutoBackend(model or self.args.model,
                                 device=select_device(self.args.device, verbose=verbose),
                                 dnn=self.args.dnn,
                                 data=self.args.data,
                                 fp16=self.args.half,
                                 fuse=True,
                                 verbose=verbose)
        self.device = self.model.device  # update device
        self.args.half = self.model.fp16  # update half
        self.model.eval()
