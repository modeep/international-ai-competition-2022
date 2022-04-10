# CLASSBER
"간단히 물건을 close up 할 때 굳이 인력이 필요할까?" 라는 의문을 홈쇼핑에서 느꼈다.  
만약 AI가 홈쇼핑 물건을 학습하고 물건이 카메라 가운데에 오도록 자동으로 따라다니면 인력소모도 적고,
좀 더 카메라 감독들도 홈쇼핑에 집중할 수 있지 않을까 생각하게 되었다.  
Project CLASSBER는 실시간으로 물건을 따라다니는 AI 카메라를 만드는 것이 목표이다.

## Model
Detect Model은 [M2Det](https://github.com/VDIGPKU/M2Det) 이라는 Model을 사용하였다.
Hand Landmark Model은 [빵형의 MediaPipe 응용](https://www.youtube.com/watch?v=udeQhZHx-00) 을 참고하였다.

## Installation
```
$ cd classber
$ pip3 install -r requirements.txt

$ cd detect_ai
$ sh make.sh

$ cd models
$ python3 m2det_model.py
```

## Detect Model How To RUN
### 1. Detect Demo
```
$ cd classber/detect_ai

# Img
$ python3 demo.py -c=configs/m2det512_vgg.py -m=weights/m2det512_vgg.pth --show
# Cam
$ python3 demo.py -c=configs/m2det512_vgg.py -m=weights/m2det512_vgg.pth --show --cam=0

$ python3 classber_bottle.py
```

### 2. Evaluation
```
$ python3 test.py -c=configs/m2det512_vgg.py -m=weights/m2det512_vgg.pth
$ python test.py -c=configs/m2det512_vgg.py -m=weights/m2det512_vgg.pth --test
```

### 3. Training
All training configs and model configs are written well in configs/*.py.
```
$ CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -c=configs/m2det512_vgg.py --ngpu 4 -t True
```

## Hand Detect How To RUN


## Reference
[https://github.com/VDIGPKU/M2Det#Training](https://github.com/VDIGPKU/M2Det#Training)