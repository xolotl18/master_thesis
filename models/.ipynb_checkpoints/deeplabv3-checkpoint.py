""" DeepLabv3 Model download and change the head for your prediction"""
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models


def createDeepLabv3(outputchannels=1, version="resnet"):
    """DeepLabv3 class with custom head

    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.

    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """
    if version == "resnet":
        model = models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True)
        model.aux_classifier=None
        for par in model.parameters():
            par.requires_grad = False
        
        model.classifier = DeepLabHead(2048, outputchannels)
    else:
        model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True, progress=True)
        model.aux_classifier=None
        for par in model.parameters():
            par.requires_grad = False
        
        model.classifier = DeepLabHead(960, outputchannels)
    
    
    return model
