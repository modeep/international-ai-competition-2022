# Import Library
from os.path import isfile, isdir
from os import pardir, mkdir
import gdown


weights_path = f'{pardir}/weights'


def m2det_download():
    if isdir(weights_path):
        if isfile(f"{weights_path}/m2det512_vgg.pth"):
            print("이미 m2det model이 있습니다.")
            return
        else:
            print("m2det model를 다운로드 받습니다.")
            url = 'https://drive.google.com/uc?id=1K6X26iAXaZDQzxQFsDFTygkQ2ZLKmjbZ'
            out_path = f'{weights_path}/m2det512_vgg.pth'
            gdown.download(url, out_path, quiet=False)
    else:
        print("weights 디렉토리가 없습니다.")
        mkdir(weights_path)
        print("디렉토리를 만들었습니다.")
        m2det_download()


def vgg_download():
    if isdir(weights_path):
        if isfile(f"{weights_path}/vgg16_reducedfc.pth"):
            print("이미 vgg model이 있습니다.")
            return
        else:
            print("vgg model를 다운로드 받습니다.")
            url = 'https://drive.google.com/uc?id=1f0C0RH6DScgdiWGg3MHmNDQxbtLpb-J5'
            out_path = '../weights/vgg16_reducedfc.pth'
            gdown.download(url, out_path, quiet=False)
    else:
        print("weights 디렉토리가 없습니다.")
        mkdir(weights_path)
        print("디렉토리를 만들었습니다.")
        vgg_download()


if __name__ == '__main__':
    m2det_download()
    vgg_download()