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

        # bin_range --> start_range부터 end_range까지 end_range - start_range + 1개의 균일한 간격으로 이루어진 array
        self.bin_range = numpy.linspace(start_range, end_range, end_range-start_range+1).astype('int').tolist()
    
        # num_of_params 는 bin_range의 원소의 개수,  end_range - start_range + 1 과 같음
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
                    tmp = m.weight.data.clone() # Conv2d의 weight값을 복사 후 tmp에 저장
                    self.saved_params.append(tmp) # saved_params는 nn.Conv2d.weight.data로 구성된 중첩 리스트
                    self.target_modules.append(m.weight) # target_modules는 nn.Conv2d.weight로 구성된 리스트
                    
        # Tensor --> torch.Tensor
        # nn.Conv2d.weight -->  Tensor 형태의 학습가능한 weight 
        # nn.Conv2d.bias -->  Tensor 형태의 학습가능한 bias 
        # nn.Conv2d.weight.data --> array 형태로 weight를 copy해서 반환

    def binarization(self):
        self.meancenterConvParams()  # Conv2d.weight 각각에 Conv2d.weight의 평균값을 빼준다. 
        self.clampConvParams() # weight가 -1.0 ~ 1.0 사이의 값이 되도록 조절 
        self.save_params() # Conv2d.weight저장
        self.binarizeConvParams() # Conv2d.weight를 -1 또는 1로 binarize



    def meancenterConvParams(self):
        '''
        target_modules --> nn.Conv2d.weight 전체를 모은 리스트
        target_modules[idx] --> 특정한 Conv2d.weight
        target_modules[idx].data -->  torch.Tensor
        target_modules[idx].data.size() --> tensor의 크기
        torch.Tensor클래스의 mean(dim = None, keepdim = False) --> input tensor 속 모든 원소들의 평균값을 반환
        torch.Tensor클래스의 mul(value) --> input tensor에 value를 곱한다
        torch.Tensor클래스의 expand_as(other) --> tensor의 크기를 other의 크기와 동일하게 확장시킨다
        torch.Tensor클래스의 add(other) --> tensor 속 모든 원소들에 other를 더한다
        '''
        for index in range(self.num_of_params): # num_of_params --> bin_range의 길이
            s = self.target_modules[index].data.size()
            
            # negative mean --> weight의 평균을 구한 후 -1곱해서 음수로 만든다.
            negMean = self.target_modules[index].data.mean(1, keepdim=True).mul(-1).expand_as(self.target_modules[index].data)
            # weight에 negative mean을 더해준다
            self.target_modules[index].data = self.target_modules[index].data.add(negMean)

    def clampConvParams(self):
        '''
        torch.Tensor클래스의 clamp(min,max) --> 모든 원소를 min, max 안의 범위로 고정시킨다 
        '''
        for index in range(self.num_of_params):
            self.target_modules[index].data = self.target_modules[index].data.clamp(-1.0, 1.0)

    def save_params(self):
        '''
        saved_param 리스트에 target_modules[idx].data를 copy
        '''
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def binarizeConvParams(self):
        '''
        < torch.Tensor클래스 >
        nelement() --> num element : input tensor 속 원소의 개수를 반환
        size() --> tensor의 크기 반환
        norm() --> tensor의 norm 반환, 벡터의 크기과 동일
        torch.norm(input, p='fro', dim=None, keepdim=False, out=None, dtype=None)
        cf) 벡터 v = (1, 2, 3) 이라면 벡터 v의 크기는 root(1+4+9)
        sum(dim, keepdim = False) --> 지정한 dim의 각 행을 모두 더한 값을 반환. 
        sign() --> 원소 각각에 sign()함수 적용, 양수는 1, 음수는 -1, 0은 0
        mul(other) --> 원소 각각에 other를 곱해준다. 
        div(other) --> 원소 각각에 other를 나눠준다.
        m.expand(s) --> 
        
        n --> torch.Tensor[0] 속의 원소의 개수
        s --> torch.Tensor의 크기
        m --> torch.Tensor의 norm을 구한다. --> output_0 : tensor형태 
            output_0의 2 dim의 각 행의 총 합을 구한다. --> output_1 : tensor형태
            output_1의 1 dim의 각 행의 총 합을 구한다. --> output_2 : tensor형태
            output_2를 n으로 나눈다. --> m
        
        target_modules[idx].data에 sign()함수를 적용한 후, m을 s의 크기로 확장한 값을 곱해준다. 
        '''
        for index in range(self.num_of_params):
            n = self.target_modules[index].data[0].nelement()
            s = self.target_modules[index].data.size()
            m = self.target_modules[index].data.norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(n)
            self.target_modules[index].data = self.target_modules[index].data.sign().mul(m.expand(s))

    def restore(self):
        '''
        target_modules[idx].data에 saved_param[idx]의 값을 복사해서 저장
        '''
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def updateBinaryGradWeight(self):
        '''
        n --> weight[0]의 원소의 개수
        s --> weight의 텐서의 크기
        m --> weight 텐서의 norm을 구한다. --> output_0 , tensor형태
            output_0의 2 dim의 각 행의 총합을 구한다 --> output_1, tensor형태
            output_1의 1 dim의 각 행의 총합을 구한다 --> output_2, tensor형태
            output_2를 n으로 나누고, s의 크기로 확장한다 --> m
        '''
        for index in range(self.num_of_params):
            weight = self.target_modules[index].data
            n = weight[0].nelement()
            s = weight.size()
            m = weight.norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            m[weight.lt(-1.0)] = 0 # weight가 -1보다 작으면 0
            m[weight.gt(1.0)] = 0 # weight가 1보다 크면 0
            # m = m.add(1.0/n).mul(1.0-1.0/s[1]).mul(n)
            # self.target_modules[index].grad.data = self.target_modules[index].grad.data.mul(m)
            m = m.mul(self.target_modules[index].grad.data)
            m_add = weight.sign().mul(self.target_modules[index].grad.data)
            m_add = m_add.sum(3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            m_add = m_add.mul(weight.sign())
            self.target_modules[index].grad.data = m.add(m_add).mul(1.0-1.0/s[1]).mul(n)
