# Import Library
from os.path import isfile
import gdown


def m2det_download():
    if isfile("../weights/m2det512_vgg.pth"):
        return
    else:
        url = 'https://drive.google.com/uc?id=1K6X26iAXaZDQzxQFsDFTygkQ2ZLKmjbZ'
        out_path = '../weights/m2det512_vgg.pth'
        gdown.download(url, out_path, quiet=False)


def vgg_download():
    if isfile("../weights/vgg16_reducedfc.pth"):
        return
    else:
        url = 'https://drive.google.com/uc?id=1f0C0RH6DScgdiWGg3MHmNDQxbtLpb-J5'
        out_path = '../weights/vgg16_reducedfc.pth'
        gdown.download(url, out_path, quiet=False)


if __name__ == '__main__':
    m2det_download()
    vgg_download()