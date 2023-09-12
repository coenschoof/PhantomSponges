from local_yolos.yolov8.ultralytics.nn.autobackend import AutoBackend
import torch

class CustomPredictor(AutoBackend): 
    def __call__(self, source=None, model=None, stream=False, *args, **kwargs):
        #Setup model
        if not self.model:
            #print(model)
            self.setup_model(model, verbose=False)

        preds = list(self.model(source))
        #print(self.model.model(source))
        preds[0] = preds[0].permute(0,2,1) #make sure its similar to yolov5's output (torch.Size([8, 25200, 85]))
        # Create a new tensor with the desired shape 'torch.Size([3, 8400, 85])'

        new_shape = list(preds[0].shape)
        new_shape[-1] += 1
        new_tensor = torch.zeros(new_shape).to("cuda")
        #print(new_tensor.shape)

        # Copy the data from the original tensor to the new tensor, shifting values to the right
        new_tensor[:, :, 4] = 1 
        #print(new_tensor.shape)
        new_tensor[:, :, 0:4], new_tensor[:, :, 5:] = preds[0][:, :, 0:4], preds[0][:, :, 4:] 
        #print(new_tensor.shape)
        return (new_tensor, )




import torch
import torch.nn as nn

# Define a wrapper class for the DetectionModel
class CustomDetectionModelWrapper(nn.Module):
    def __init__(self, detection_model, device = "cuda"):
        super(CustomDetectionModelWrapper, self).__init__()
        self.detection_model = detection_model.to(device)

    def forward(self, input):
        # Perform the forward pass using the original model
        output = self.detection_model(input)

        # Add your post-processing logic here
        processed_output = self.custom_post_processing(output)

        return processed_output

    def custom_post_processing(self, output, device = "cuda"):
        preds = list(output)
        preds[0] = preds[0].permute(0,2,1) #make sure its similar to yolov5's output (torch.Size([8, 25200, 85]))
        new_shape = list(preds[0].shape)
        new_shape[-1] += 1
        new_tensor = torch.zeros(new_shape).to(device)

        new_tensor[:, :, 4] = 1 
        new_tensor[:, :, 0:4], new_tensor[:, :, 5:] = preds[0][:, :, 0:4], preds[0][:, :, 4:] 
        return (new_tensor, )
