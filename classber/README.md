# FIBER
"간단히 물건을 close up 할 때 굳이 인력이 필요할까?" 라는 의문을 홈쇼핑에서 느꼈다.  
만약 AI가 홈쇼핑 물건을 학습하고 물건이 카메라 가운데에 오도록 자동으로 따라다니면 인력소모도 적고,
좀 더 카메라 감독들도 홈쇼핑에 집중할 수 있지 않을까 생각하게 되었다.  
Project CLASSBER는 실시간으로 물건을 따라다니는 AI 카메라를 만드는 것이 목표이다.

## Model
Model은 [M2Det](https://github.com/VDIGPKU/M2Det) 이라는 Model을 사용하였다.

## Installation
```
$ git clone https://github.com/modeep/international-ai-competition-2022
$ cd international-ai-competition-2022

$ docker build . -t international-ai
$ docker run -it -d -p 8080:8080 --name "aic" international-ai
```