# XNOR_Net_PyTorcy_CIFAR_10_Analysis

## 개요

XNOR_NET_PyTorch CIFAR10 코드 분석  
원본 링크 <https://github.com/jiecaoyu/XNOR-Net-PyTorch>  

## 구성

< your_dir >  
ㄴ--> main.py  
ㄴ--> util.py  
ㄴ--> data.py

< your_dir/models >  
ㄴ--> nin.py  

< data_dir >  
ㄴ--> train_data  
ㄴ--> train_labels  
ㄴ--> test_data  
ㄴ--> test_labels  

## 동작 과정

### main.py  
< train 기준 >

1. command line에서 인자(args)들을 받을 수 있도록 설정  

2. cpu 또는 gpu 환경에서 무작위성 배제를 위해 seed 설정  

3. 인자로 받은 data 경로에 train_data가 존재하는지 확인. 없으면 err  

4. data를 (train_data, target), (test_data, target)으로 load  

5. 10개의 class를 정의  

6. nin.py의 Net 클래스 객체인 model 생성

7. 인자를 통해서 pretrained model 사용 여부 확인  
사용하는 경우 : pretrained model load 하고, best_acc를 pretrained model이 가지고 있는 값으로 설정  
사용하지 않는 경우 : best_acc는 0,  
model의 모든 Conv2d 모듈의 weight는 평균 0, 표준편차 0.05를 따르는 정규분포로,  
bias는 0으로 설정  

8. base_lr, Adam optimizer, CrossEntropy loss function 설정  

9. util.py의 BinOp 클래스 객체인 bin_op(model) 생성

10. train 시작

a. bin_op.binarization() 실행  
ㄴ--> Conv2d의 weight들의 평균값을 구하고, 평균값을 weight 각각에 빼준다  
ㄴ--> Conv2d의 weight를 -1.0 ~ 1.0 사이의 값으로 만든다. -1.0보다 작으면 -1.0, 1.0보다 크면 1.0  
ㄴ--> Conv2d의 weight를 따로 저장해둔다  
ㄴ--> Conv2d의 weight에 sign함수를 취해서 binarization   
-1 <= weight < 0 --> weight = -1   
weight == 0 --> weight = 0   
0 < weight <= 1 --> weight = 1   

b. 순전파  
model에 train_data를 넣고 예상되는 output을 계산한다.

c. 역전파 
output과 실제 label에 대한 loss를 구한다.

d. loss가 작아지는 방향으로 parameter와 grad 업데이트  

e. 반복  


