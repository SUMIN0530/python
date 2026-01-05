# Pytorch vs 텐서 기초
# pytorch = Facebook(Meta)에서 개발한 딥러닝 프레임 워크
# 특징
# - 동적 계산 그래프
# - Pythonic한 문법
# - 강력한 GPU 지원
# - 활발한 커뮤니티
# 
# TensorFlow
# - PyTorch : 연구 / 프로토타입에 강함
# - TensorFlow : 배포 / 프로덕션에 강함
# 최근에는 두 프레임워크 모두 비슷한 기능 제공

# 텐서(Tensor)
# 텐서 = 다차원 배열 = 숫자들을 여러 차원(축)으로 배열해 좋은 데이터
# 
# 0D : 스칼라 (단일 값)
# 1D : 벡터 (1차원 배열)
# 2D : 행렬 (2차원 배열)
# 3D : 3차원 텐서 (단일 값)
# Numpy와 유사하지만
# - GPU 연산 지원
# - 자동 미분 지원

import torch
# 텐서 생성
# 1. 직접 생성
scalar = torch.tensor(3.14)
vector = torch.tensor([1, 2, 3, 4])
matrix = torch.tensor([
    [1, 2],
    [3, 4],
    [5, 6]
])

print(f'스칼라 : {scalar}, 차원 : {scalar.dim()}')
print(f'벡터 : {vector}, 차원 : {vector.dim()}')
print(f'행렬 :\n {matrix}, 차원 : {matrix.dim()}')

# 특정 값으로 초기화
zeros = torch.zeros(2, 3)
ones = torch.ones(2,3)
full = torch.full((2, 3), 7)

print(f'\n영행렬 : \n {zeros}')
print(f'\n일행렬 : \n {ones}')
print(f'\n7로 채운 행렬 : \n {full}')

rand = torch.rand(2, 3) # 0 ~ 1 균등 분포
randn = torch.randn(2, 3) # 표준 정규 분포
randint = torch.randint(0, 10, (2, 3)) # 정수 무작위

print(f'균등분포 : \n {rand}')
print(f'표준정규분포 : \n {randn}')
print(f'정수 무작위 : \n {randint}')

# 연속 값
arange = torch.arange(0, 10, 2)
linspace = torch.linspace(0, 1, 5)

print(f'arange:\n {arange}')
print(f'linspace:\n {linspace}')

# 텐서 속성 
x = torch.randn(3, 4, 5)

print(f'크기 (shape): {x.shape}')
print(f'차원 수 (dim): {x.dim()}')
print(f'원소 개수 (numel): {x.numel}')
print(f'데이터 타입 (dtype): {x.dtype}')
print(f'장치 (device): {x.device}')

# 데이터 타입
float_tensor = torch.tensor([1.0, 2.0])
int_tensor = torch.tensor([1, 2])

print(f'float 타입 : {float_tensor.dtype}')
print(f'int 타입 : {int_tensor.dtype}')

# 명시적 타입 지정
x = torch.tensor([1, 2, 3], dtype=torch.float32)
y = torch.tensor([1, 2, 3], dtype=torch.int32)
z = torch.tensor([True, False], dtype=torch.bool)

print(f'x 타입: {x.dtype}')
print(f'y 타입: {y.dtype}')
print(f'z 타입: {z.dtype}')

# 타입 변환
a = x.int()
b = y.float()
print(f'x => a타입 : {a.dtype}')
print(f'y => b타입 : {b.dtype}')

# 장치 (Device)
cpu_tensor = torch.tensor([1, 2, 3])
print(f'기본 장치 : {cpu_tensor.device}')

# GPU로 이동 (CUDA 사용 가능 시)
if torch.cuda.is_available():
    gpu_tensor = cpu_tensor.to('cuda')
    print(f'GPU 텐서 : {gpu_tensor.device}')

    # 또는 
    gpu_tensor2 = cpu_tensor.cuda()
    print(f'GPU 텐서 2 : {gpu_tensor2.device}')

# Device(장치)는 이 텐서(데이터)를 CPU메모리에 둘지 GPU(CUDA) 메모리에 둘지 정하는 것
# 연산은 데이터가 있는 곳에서만 빠르게/정상적으로 일어나는데
# 서로 다른 장치에 있으면 연산이 아예 CPU <-> GPU 복사 떄문에 엄청 느려질 수 있음
# GPU로 보내서 연산을 빠르게 하려고한다. 

import numpy as np
np_array = np.array([[1, 2, 3], [4, 5, 6]])
print(f'Numpy 배열 : \n {np_array}')

# 방법 1. torch.tensor() - 복사본 생성
tensor1 = torch.tensor(np_array)
print(f'tensor1 : {tensor1}')

# 방법 2. torch.from_numpy() - 메모리 공유
tensor2 = torch.from_numpy(np_array)
print(f'tensor2 : {tensor2}')

# 메모리 공유 확인
np_array[0, 0] = 100
print(f'변경 후 tensor1 : \n{tensor1}')
print(f'변경 후 tensor2 : \n{tensor2}')

tensor = torch.tensor([[1, 2], [3, 4]])

# Numpy로 변환 
np_array = tensor.numpy()
print(f'타입 : {type(np_array)}')
print(f'배열 : \n{np_array}')

# 텐서 인덱싱과 슬라이싱
x = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# 단일 원소
print(f'x[0, 0] : {x[0, 0]}')
print(f'x[1, 2] : {x[1, 2]}')

# 행/열 접근
print(f'첫 번째 행: {x[0]}')
print(f'마지막0 행: {x[-1]}')

print(f'첫 번째 열: {x[:, 0]}')
print(f'마지막0 행: {x[:, -1]}')

x = torch.arange(12).reshape(3, 4)
print(f'원본 : \n{x}')

print(f'x[0:2, 1:3]\n {x[0:2, 1:3]}')
print(f'x[::2, ::2]\n {x[::2, ::2]}')

# 조건 인덱싱
x = torch.randn(3, 3)
print(f'원본 : \n{x}')

# 조건에 맞는 원소
mask = x > 0
print(f'양수 마스크: \n {mask}')
print(f'양수 값들: \n {x[mask]}')

# 텐서 형태 변환
x = torch.arange(12)
print(f'원본 : {x}')
print(f'크기 : {x.shape}')

# reshape
reshaped = x.reshape(3, 4)
print(f'reshaped : {reshaped}')

auto = x.reshape(4, -1) # 4행 3열
auto = x.reshape(2, -1) # 2행 6열
print(f'auto : {auto}')


# squeeze : 크기 1인 차원 제거
x = torch.randn(1, 3, 1, 4)
print(f'원본\n {x}')
print(f'크기\n {x.shape}')
print(f'squeeze\n {x.squeeze().shape}')
print(f'squeeze\n {x.squeeze(0).shape}')

# unsqueeze : 차원 추가 
y = torch.randn(3, 4)
print(f'원본\n {y}')
print(f'크기\n {y.shape}')
print(f'unsqueeze(1)\n {y.unsqueeze(1).shape}')
print(f'unsqueeze()\n {y.unsqueeze(0).shape}')
print(f'unsqueeze(-1)\n {y.unsqueeze(-1).shape}')

# transpose와 permute
# transpose : 두 차원 교환
x = torch.randn(2, 3)
print(f'원본\n {x}')
print(f'transpose\n {x.transpose(0, 1).shape}')

# permute : 차원 순서 변경
y = torch.randn(2, 3, 4)
print(f'원본\n {y}')
print(f'permute\n {y.permute(2, 0, 1).shape}')


