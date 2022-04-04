# Import Library
from os.path import isfile
import gdown

def m2det_download():
    if isfile("../models/m2det512_vgg.pth"):
        return
    else:
        url = 'https://drive.google.com/uc?id=1K6X26iAXaZDQzxQFsDFTygkQ2ZLKmjbZ'
        out_path = '../models/m2det512_vgg.pth'
        gdown.download(url, out_path, quiet=False)