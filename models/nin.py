import torch.nn as nn
import torch
import torch.nn.functional as F

class BinActive(torch.autograd.Function):
    '''
    입력으로 들어온 activations를 이진화하고, 채널의 차원은 평균값으로 바꾼다.
    torch.nn은 미니배치 형태의 input만 지원
    '''
    
    # input은 torch
    def forward(self, input):
        # save_for_backward() --> 역전파 단계에서 사용할 input 저장.
        self.save_for_backward(input)
        
        # input의 크기 반환. torch.Size는 튜플형태
        # print(size) 해보면 torch.Size([행 개수, 열 개수])로 나옴
        size = input.size()
        
        # torch.mean() --> input텐서에 있는 모든 요소의 평균값을 반환.
        mean = torch.mean(input.abs(), 1, keepdim=True)
        
        # input에 sign()함수 적용 : 0은 0, 양수는 1 음수는 -1로 바뀜.
        input = input.sign()
        
        return input, mean

    def backward(self, grad_output, grad_output_mean):
        '''
        torch.ge(a, b) --> a >= b 
        torch.le(a, b) --> a <= b
        self.saved_tensors --> 반환값은 리스트 또는 튜플인데 여기선 튜플 
        '''
        
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        
        # grad_input : -1 <= input <= 1 이면 0으로 설정
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        
        return grad_input

class BinConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=-1, stride=-1, padding=-1, dropout=0):
        # super()는 클래스 상속과 연관된 함수로 부모 클래스를 참조
        # 자식 클래스에서 메소드 이름을 가지고 부모 클래스의 메소드를 찾을 수 있음
        super(BinConv2d, self).__init__()
        
        self.layer_type = 'BinConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout

        # nn.BatchNorm2d : 4D input에 대해 배치 정규화를 적용
        self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
        #  self.bn의 weight를 0으로 만든 후 1을 더해서 1로 만들어줌
        self.bn.weight.data = self.bn.weight.data.zero_().add(1.0)
        
        # nn.Dropout : train 과정에서 확률에 따라 입력 텐서의 일부 요소를 임의로 0으로 만든다.
        if dropout!=0:
            self.dropout = nn.Dropout(dropout)
            
        # nn.Conv2d : input(텐서)에 대해 2D 컨벌루션을 적용.
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        # nn.ReLU :  ReLU 함수를 element 단위로 적용.
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.bn(x)
        x, mean = BinActive()(x)
        if self.dropout_ratio!=0:
            x = self.dropout(x)
        x = self.conv(x)
        x = self.relu(x)
        return x

class Net(nn.Module):
    def __init__(self):
        # super()는 클래스 상속과 연관된 함수로 부모 클래스를 참조
        # 자식 클래스에서 메소드 이름을 가지고 부모 클래스의 메소드를 찾을 수 있음
        super(Net, self).__init__()
        
        self.xnor = nn.Sequential(
            '''
            < nn.Sequential() >
           순차적 컨테이너. 생성자에 전달 된 순서대로 모듈이 추가된다.  순서가 지정된 모듈을 전달할 수도 있다.
  
            < nn.Conv2d() > 
            input(텐서)에 대해 2D 컨벌루션을 적용.
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
            
            < nn.BatchNorm2d() >
            4D input에 대해 배치 정규화를 적용.
            torch.nn.BatchNorm2d( num_features , eps = 1e-05 , momentum = 0.1 , affine = True , track_running_stats = True )
            
            < nn.ReLU >
            torch.nn.ReLU( inplace = False )
            ReLU 함수를 element 단위로 적용.
            
            < BinConv2d() >
            
            
            < nn.MaxPool2d() >
            input(텐서)에 2D max pooling을 적용.
            torch.nn.MaxPool2d( kernel_size , stride = None , padding = 0 , dilation = 1 , return_indices = False , ceil_mode = False )
            
            < nn.AvgPool2d() >
            input(텐서)에 2D 평균 풀링을 적용.
            torch.nn.AvgPool2d( kernel_size , stride = None , padding = 0 , ceil_mode = False , count_include_pad = True , divisor_override = None )
            
            '''
                nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False),
                nn.ReLU(inplace=True),
                BinConv2d(192, 160, kernel_size=1, stride=1, padding=0),
                BinConv2d(160,  96, kernel_size=1, stride=1, padding=0),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

                BinConv2d( 96, 192, kernel_size=5, stride=1, padding=2, dropout=0.5),
                BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),
                BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

                BinConv2d(192, 192, kernel_size=3, stride=1, padding=1, dropout=0.5),
                BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False),
                nn.Conv2d(192,  10, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
                )

    def forward(self, x):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                # hasattr(object, name) --> 'object'에 해당하는 'name'의 attribute가 있으면 True, 없으면 False
                if hasattr(m.weight, 'data'):
                    #  m.weight.data가 최소한 0.01.이 되도록 만든다.
                    m.weight.data.clamp_(min=0.01)
        x = self.xnor(x)
        x = x.view(x.size(0), 10)
        return x
