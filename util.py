import torch.nn as nn
import numpy

# model 은 nin.py의 Net클래스객체

class BinOp():
    def __init__(self, model):
        count_Conv2d = 0
        # Net에 있는 conv2d의 개수 카운트
        for m in model.modules():
            # 모듈 m이 conv2d라면
            if isinstance(m, nn.Conv2d):
                count_Conv2d = count_Conv2d + 1

        start_range = 1
        end_range = count_Conv2d-2

        # <bin_range> 
        # start_range부터 end_range까지 end_range-start_range+1 개의 균일한 간격의 정수들로 이루어진 array
        self.bin_range = numpy.linspace(start_range,
                end_range, end_range-start_range+1)\
                        .astype('int').tolist()
    
        # num_of_params 는 bin_range의 원소의 개수,  end_range-start_range+1 과 같음
        self.num_of_params = len(self.bin_range)
        self.saved_params = []
        self.target_params = []
        self.target_modules = []
        index = -1

        # Net에 있는 모듈에 대해서 수행
        for m in model.modules():
            # 모듈 m이 conv2d라면
            if isinstance(m, nn.Conv2d):
                index = index + 1
                # index가 bin_range에 포함된다면
                if index in self.bin_range:
                    # Conv2d.weight.data 복사 후 tmp에 저장
                    # Conv2d.weight.data 와 Conv2d.weight 의 차이 알아보기!
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)

    def binarization(self):
        self.meancenterConvParams()  # 모듈이 Conv2d일때, Conv2d.weight의 평균값을 구함
        self.clampConvParams() # weight가 -1.0 ~ 1.0 사이의 값이 되도록 조절 
        self.save_params() # Conv2d.weight저장
        self.binarizeConvParams() # Conv2d.weight를 -1 또는 1로 binarize



    def meancenterConvParams(self):
        for index in range(self.num_of_params):
            # target_modules는 nn.Conv2d.weight로 구성된 리스트
            s = self.target_modules[index].data.size()
            negMean = self.target_modules[index].data.mean(1, keepdim=True).\
                    mul(-1).expand_as(self.target_modules[index].data)
            self.target_modules[index].data = self.target_modules[index].data.add(negMean)

    def clampConvParams(self):
        for index in range(self.num_of_params):
            # target_modules는 nn.Conv2d.weight로 구성된 리스트
            self.target_modules[index].data = \
                    self.target_modules[index].data.clamp(-1.0, 1.0)

    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def binarizeConvParams(self):
        for index in range(self.num_of_params):
            n = self.target_modules[index].data[0].nelement()
            s = self.target_modules[index].data.size()
            m = self.target_modules[index].data.norm(1, 3, keepdim=True)\
                    .sum(2, keepdim=True).sum(1, keepdim=True).div(n)
            self.target_modules[index].data = \
                    self.target_modules[index].data.sign().mul(m.expand(s))

    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def updateBinaryGradWeight(self):
        for index in range(self.num_of_params):
            weight = self.target_modules[index].data
            n = weight[0].nelement()
            s = weight.size()
            m = weight.norm(1, 3, keepdim=True)\
                    .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            m[weight.lt(-1.0)] = 0 
            m[weight.gt(1.0)] = 0
            # m = m.add(1.0/n).mul(1.0-1.0/s[1]).mul(n)
            # self.target_modules[index].grad.data = \
            #         self.target_modules[index].grad.data.mul(m)
            m = m.mul(self.target_modules[index].grad.data)
            m_add = weight.sign().mul(self.target_modules[index].grad.data)
            m_add = m_add.sum(3, keepdim=True)\
                    .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            m_add = m_add.mul(weight.sign())
            self.target_modules[index].grad.data = m.add(m_add).mul(1.0-1.0/s[1]).mul(n)
