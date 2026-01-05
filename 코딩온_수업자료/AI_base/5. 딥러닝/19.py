# 손실 함수
# 예측과 실제의 차이를 측정

# 목적
# - 모델 성능을 수치화
# - 최적화 목표 제공
# - 손실이 작을수록 좋은 모델

# 다른 이름:
# - 비용 함수 (Cost Funtion)
# - 목적 함수 (Objective Funtion)

# 손실 vs 비용
# 손실 (Loss)
# - 단일 샘플에 대한 오차
# - L(ŷ, y)

# 비용 (Cost)
# - 전체 샘플의 평균 손실 
# - J = (1/n) × Σ L(ŷᵢ, yᵢ)



# 회귀 손실 함수
# MSE(Mean Squared Error)

# 수식: MSE = (1/n) × Σ(yᵢ - ŷᵢ)²

# 특징
# - 오차의 제곱 평균
# - 큰 오차에 큰 패널티
# - 미분이 간단

# 사용: 일반적인 회귀 문제

import torch
import torch.nn as nn

# 예측과 실제
y_pred = torch.tensor([2.5, 0.0, 2.1, 7.8])
y_test = torch.tensor([3.0, -0.5, 2.0, 7.5])

# MSE 계산 (수동)
mse_manual = ((y_test-y_pred) ** 2).mean()
print(f'MSE (수동): {mse_manual:.4f}')

# PyTorch MSE
mse_loss = nn.MSELoss()
mse = mse_loss(y_pred, y_test)
print(f'MSE (PyTorch) {mse:.4f}')

# MAE(Mean Absolute Error)
# 수식: MAE = (1/n) × Σ|yᵢ - ŷᵢ|

# 특징
# - 오차의 절대값 평균
# - 이상치에 덜 민감
# - 0에서 미분 불가

# 사용: 이상치가 있는 회귀

mae_loss = nn.L1Loss()
mae = mae_loss(y_pred, y_test)
print(f'MAE (Pytorch) {mae:.4f}')


# Huber Loss

# 수식:
# δ 이하: 0.5 × (y - ŷ)²
# δ 초과: δ × (|y - ŷ| - 0.5δ)

# 특징
# MSE와 MAE의 조합
# - 작은 오차: MSE처럼 동작
# - 큰 오차 : MAE처럼 동작

# 사용: 이상치가 일부 있는 경우
huber_loss= nn.HuberLoss(delta=1.0)
huber = huber_loss(y_pred, y_test)
print(f'Huber (Pytorch) {huber:.4f}')

# 로짓 (Logits) : Softmax를 거치기 전의 원시 출력값
# 예 : [2.1, 0.5, -0.3]-> 아직 확률이 아님

# Normalize: 데이터를 평균 0, 표준편차 1로 변환
# (0.1307, 0.3081) : MNIST 데이터셋의 평균과 표준 편차
# - 학습 안정성 향상

# Dropout: 학습 시 일부 뉴련을 무작위로 끄는 기법
# - nn.Dropout(0.2): 20%의 뉴련을 랜덤하게 비활성화
# - 과적합 방지 효과

# model.train() vs model.eval()
# - train(): 학습 모드 (Dropout 작동)
# - eval(): 평가 모드 (Dropout 비활성화)

# torch.no_grad() : 기울기 계산 비활성화
# - 추론/ 평가 시 사용
# - 메모리 절약 + 속도 향상

# DataLoader: 데이터를 배치로 나눠서 제공
# batch_size=64: 한번에 64개씩 처리
# shuffle=True: 매 애폭마다 데이터 순서 섞기

# 애폭 (Epoch): 전체 데이터를 1회 학습
# - 10 epochs = 60,000개 데이터를 10번 학습

# 배치 (Batch): 한 번에 처리하는 데이터 묶음
# - 전체를 한번에 처리하면 메모리 부족
# - 작은 배치로 나눠서 처리