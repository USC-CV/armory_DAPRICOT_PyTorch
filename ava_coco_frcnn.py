"""
Pytorch Faster-RCNN for coco object detection
"""
import logging
from typing import Optional
import torch
import torchvision
from art.estimators.object_detection import PyTorchFasterRCNN


logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def make_fastrcnn_model(weights_path: Optional[str] = None) -> torch.nn.Module:
    """
    This is an MSCOCO pre-trained model.
    This model only performs inference and is not trainable. To perform other
    custom fine-tuning, please follow instructions at
    https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
    """

    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    if weights_path:
        checkpoint = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(checkpoint)

    return model


def get_art_model(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
) -> PyTorchFasterRCNN:
    model = make_fastrcnn_model(weights_path=weights_path, **model_kwargs)
    model.to(DEVICE)

    # This model receives inputs in the canonical form of [0,1], so no further
    # preprocessing is needed

    art_model = PyTorchFasterRCNN(
        model=model,
        clip_values=(0, 1.0),
        channels_first=False,
        preprocessing_defences=None,
        postprocessing_defences=None,
        preprocessing=None,
        attack_losses=(
            "loss_classifier",
            "loss_box_reg",
            "loss_objectness",
            "loss_rpn_box_reg",
        ),
        device_type=DEVICE,
        **wrapper_kwargs
    )

    return art_model