from model.DnCNN import DnCNN
from model.VDSR import VDSR
from model.EDSR import EDSR
from model.GGRL import GGRL
from model.PRL import PRL
from model.MSPRL import MSPRL


def build_net(model_name, image_channel):
    if model_name == "DnCNN":
        return DnCNN(image_channel)
    if model_name == "VDSR":
        return VDSR(image_channel)
    if model_name == "EDSR":
        return EDSR(image_channel)
    if model_name == "GGRL":
        return GGRL(image_channel)
    if model_name == "PRL":
        return PRL(image_channel)
    if model_name == "MSPRL":
        return MSPRL(image_channel)
