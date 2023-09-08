from local_yolos.yolov8.ultralytics.models.yolo.detect import DetectionPredictor
from local_yolos.yolov8.ultralytics.utils import ops
from local_yolos.yolov8.ultralytics.nn.autobackend import AutoBackend
from local_yolos.yolov8.ultralytics.utils.torch_utils import select_device
import torch

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
        # Setup model
        if not self.model:
            print(model)
            self.setup_model(model, verbose=False)

        #print(type(self.model.zero_grad()))
        #print(type(self.model))
        preds = self.model(source)
        
        #print(self.model.model(source))
        preds[0] = preds[0].permute(0,2,1) #make sure its similar to yolov5's output (torch.Size([8, 25200, 85]))
        # Create a new tensor with the desired shape 'torch.Size([3, 8400, 85])'

        new_shape = list(preds[0].shape)
        new_shape[-1] += 1
        new_tensor = torch.zeros(new_shape)
        #print(new_tensor.shape)

        # Copy the data from the original tensor to the new tensor, shifting values to the right
        new_tensor[:, :, 4] = 1 
        #print(new_tensor.shape)
        new_tensor[:, :, 0:4], new_tensor[:, :, 5:] = preds[0][:, :, 0:4], preds[0][:, :, 4:] 
        #print(new_tensor.shape)
        return (new_tensor, )
        #print('NEW TENSOR SHAPE: ', new_tensor.shape, preds.shape)
        #return list(self.stream_inference(source, model, *args, **kwargs))  # merge list of Result into one
    
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
        #print(type(self.model))
        self.model.eval()
    
    # def stream_inference(self, source=None, model=None, *args, **kwargs):
    #     """Streams real-time inference on camera feed and saves results to file."""

    #     # Setup model
    #     if not self.model:
    #         self.setup_model(model, verbose=False)

    #     # Setup source every time predict is called
    #     self.setup_source(source if source is not None else self.args.source)

    #     # Warmup model
    #     if not self.done_warmup:
    #         self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
    #         self.done_warmup = True

    #     self.seen, self.windows, self.batch, profilers = 0, [], None, (ops.Profile(), ops.Profile(), ops.Profile())
    #     self.run_callbacks('on_predict_start')
    #     for batch in self.dataset:
    #         self.run_callbacks('on_predict_batch_start')
    #         self.batch = batch
    #         _, im0s, _, _ = batch

    #         # Preprocess
    #         with profilers[0]:
    #             im = self.preprocess(im0s)

    #         # Inference
    #         with profilers[1]:
    #             preds = self.inference(im, *args, **kwargs)

    #         # Postprocess
    #         with profilers[2]:
    #             self.results = self.postprocess(preds)


    #         preds[0] = preds[0].permute(0,2,1) #make sure its similar to yolov5's output (torch.Size([8, 25200, 85]))
    #         # Create a new tensor with the desired shape 'torch.Size([3, 8400, 85])'

    #         new_shape = list(preds[0].shape)
    #         new_shape[-1] += 1
    #         new_tensor = torch.zeros(new_shape)

    #         # Copy the data from the original tensor to the new tensor, shifting values to the right
    #         new_tensor[:, :, 4] = 1 
    #         new_tensor[:, :, 0:4], new_tensor[:, :, 5:] = preds[0][:, :, 0:4], preds[0][:, :, 4:] 
    #         #print('NEW TENSOR SHAPE: ', new_tensor.shape, preds.shape)
    #         return new_tensor, _
        


