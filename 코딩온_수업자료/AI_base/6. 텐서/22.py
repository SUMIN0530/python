# 산술 평균
import torch

a = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
b = torch.tensor([5, 6, 7, 8], dtype=torch.float32)

print(f'a + b = {a + b}')
print(f'a / b = {torch.mul (a, b)}')
print(f'a - b = {a - b}')
print(f'a / b = {torch.mul (a, b)}')

print(f'a * b = {a * b}')
print(f'a / b = {torch.mul (a, b)}')

print(f'a / b = {a / b}')
print(f'a / b = {torch.div (a, b)}')

# 스칼라 연산
x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)

print(f'x = {x}')
print(f'x + 10 = {x + 10}')
print(f'x - 10 = {x - 10}')
print(f'x * 10 = {x * 10}')
print(f'x ** 2 = {x ** 2}')
print(f'x / 10 = {x / 10}')

# 제자리 연산
x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
print(f'x = {x}')

# 제자리 연산 (언더스코어 접미사)
x.add_(10) # x = x + 10
print(f'x.add_(10) = {x}')

x.mul_(10) # x = x + 10
print(f'x.mul_(10) = {x}')

# 주의 : 자동 미분 중에는 제자리 연산 피하기

# 수학 함수
x = torch.tensor([-2, -1, 0, 1, 2], dtype=torch.float32)

# 절대값
print(f'abs : {torch.abs(x)}')

# 제곱근 (양수만)
x = torch.tensor([4, 3, 16, 25], dtype=torch.float32)
print(f'sqrt : {torch.sqrt(x)}')

# 지수와 로그
x = torch.tensor([1, 2, 3], dtype=torch.float32)
print(f'exp : {torch.exp(x)}')
print(f'log : {torch.log(x)}')

# 삼각함수
angles = torch.tensor([0, 3.141592/2, 3.141592], dtype=torch.float32)
print(f'sin : {torch.sin(angles)}')
print(f'cos : {torch.cos(angles)}')

# 반올림
w = torch.tensor([1.4, 1.5, 1.6])
print(f'원본 : {w}')
print(f'뭐가 있음 : {w}')

