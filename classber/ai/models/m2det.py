# Import Library
from os.path import isfile
import gdown

def m2det_download():
    if isfile("../models/m2det512_vgg.pth")