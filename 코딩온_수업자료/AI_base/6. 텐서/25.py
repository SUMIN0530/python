# 자동 미분
# 수치적 미분 없이 정확한 기울기 계산

# 딥러닝엣 필요한 이유
# 1. 역전파 알고리즘 핵심
# 2. 수종 미분은 복잡, 오류발생 가능
# 3. 수치 미분은 느리고 부정확

# PyTotch의 Auttograd
# - 연산 기록 (계산 그래프 생성)
# - 연쇄 법칙으로 기울기 자동 계산

import torch
# 함수 : y = x² + 2x + 1
# 미분 : dy/dx = 2x + 2

x = 3.0
# 수동 미분
manual_grad = 2 * x + 2

# PyTorch 자동 미분
x_tensor = torch. tensor(x, requires_grad=True)
y = x_tensor ** 2 + 2 * x_tensor + 1
y.backward()
auto_grad = x_tensor.grad
print(f'자동 미분(x=3) :  {auto_grad}')

# requires_grad - 기본값 False
# 기울기 추적 활성화 (True)

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(f'x : {x}')
print(f'requires_grad : {x.requires_grad}')

y = torch.tensor([1.0, 2.0, 3.0])
print(f'y : {y}')
print(f'requires_grad : {y.requires_grad}')

y.requires_grad_(True) # 이후에 활성화 가능
print(f'requires_grad : {y.requires_grad}')

# 연산결과의 requires_grad
a = torch.tensor([1.0], requires_grad=True)
b = torch.tensor([2.0], requires_grad=True)
c = torch.tensor([3.0])

d = a + b
e = a + c
f = b + c

print(f'a + b requires_grad : {d.requires_grad}')
print(f'a + c requires_grad : {e.requires_grad}')
print(f'b + c requires_grad : {f.requires_grad}')

# 왜 자동 미분이 필요한가?
# 신경망 학습의 핵심
# 1. 예측값 계산 (순전파)
# 2. 손실 계산 (얼마나 틀렸나)
# 3. 가중치 조정 (어ㅓ느 방향으로 얼마나) <- 여기서 미분 필요!
# 
# 손실 함수 : L = (wx - y)²
# 여기서 w의 최적값을 찾으려면
# -> dL/dw를 계산해서 w 업데이트!


import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 512), # 401,920 파라미터
    nn.ReLU(),

)

# 계산 그래프
# 연산의 흐름을 기록하는 그래프
# 예시 : z = (x + y) * (x - y)

# 계산 그래프
#     x(2)   y(3)
#      |  \  /  |
#      |   \/   |
#      |   /\   |
#      |  /  \  |
#     +         -
#     (5)      (-1)
#      \       /
#       \     /
#          *
#         (-5)
#          |
#          z
# 
# 순전파 : x, y -> z 계산 
# 역전파 : z의 기울기 -> x, y의 기울기 계산

x = torch.tensor(2.0, requires_grad=True) 
y = torch.tensor(3.0, requires_grad=True) 

# 순전파 Forward
# 계산 그래프 생성
a = x + y
b = x - y
z = a * b

print('=== 계산 그래프 ===')
print(f'a = x + y = {a} ')
print(f'b = x - y = {b} ')
print(f'z = a * b = {z} ')

# 역전파
z.backward()

print(f"\n∂z/∂x = {x.grad}")
print(f"∂z/∂y = {y.grad}")

# z = (x + y) * (x - y)
#
#

# 연쇄 법칙
# z = f(g(x))
# dz/dx = dz/dg * dg/dx
#
# 예시
# z = (x²)³        
#   = h(g(x))

# g(x) = x² , dg/dx = 2x
# h(g) = g³ , dh/dg = 3g²

# dz/dx 
#   = dh/dg * dg/dx 
#   = 3g² * 2x 
#   = 3(x²)² * 2x
#   = 6x⁵


x = torch.tensor(2.0, requires_grad=True)

# z = (x²)³ 
g = x ** 2 # 중간 값
z = g ** 3 # 최종 값

z.backward()

print(f'x = {x.item()}')
print(f'g = x² = {g.item()}')
print(f'z = g³ = {z.item()}')
print(f'dz/dx = 6x⁵ = {x.grad.item()}')
print(f'검증 6 * 2^5 = {6 * (2**5)}')
print()

# 동적 계산 그래프
# PyTorch는 실행 시 그래프 생성

x = torch. tensor(2.0, requires_grad=True)

# 조건에 따라 다른 연산
if x > 0:
    y = x ** 2
else:
    y = x ** 3

y.backward()
print(f'x > 0이므로 y = x², dy/dx = 2x = {x.grad}')
print()

# =================================================
# 실습 
x = torch.tensor(3.0, requires_grad=True)

# y = (2x + 1)³
# 단계별로 분해하기

# y = f(g(x))
# 1. f'(g(x)) <---------------- 취지가 이렇게 구하는게 아닌같애 g = 2*x로 잡더라
#  => 3(2x + 1)²
#
# 2. g'(x)
#  => 2
# 
# 3. dy/dx = f'(g(x)) * g'(x)
#  => (12x² + 12x + 3) * 2
#  => 24x² + 24x + 6  

x = torch.tensor(3.0, requires_grad=True)

g = (2*x + 1) ** 2 # 중간 값
f = g ** 3 # 최종 값

f.backward()

print(f'x = {x.item()}')
print(f'g = (2x + 1)² = {g.item()}')
print(f'f = g³ = {f.item()}')
print(f'dz/dx = 24x² + 24x + 6 = {x.grad.item()}')