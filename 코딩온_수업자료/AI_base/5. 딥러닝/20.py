# 딥러닝 종합 실습

# Pytorch로 완전한 신경망을 구현
# MINIST 손글씨 분류 문제를 해결
# 시각화

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# GPU 사용 가능 여부
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'사용 장치: {device}')

# 재현성을 위한 시드 설정
torch.manual_seed(42)

if device.type == 'cuda':
    torch.cuda.manual_seed(42)


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

# 데이터 준비 
# 데이터 변환 정의
transform = transforms.Compose([
    transforms.ToTensor(), # 텐서로 변환 (0~1 정규화)
    transforms.Normalize((0.1307,),(0.3081,)) # MNIST 평균, 표준 편차
])

# 데이터셋 다운로드
train_dataset = datasets.MNIST(
    root='./data', 
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data', 
    train=False,
    download=True,
    transform=transform
)

print(f'훈련 데이터 : {len(train_dataset)}')
print(f'테스트 데이터 : {len(test_dataset)}')


# 데이터 로더 생성
batch_size = 64

# DataLoader 생성
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
)

print(f'훈련 배치 수 : {len(train_loader)}')
print(f'테스트 배치 수 : {len(test_loader)}')

# 데이터 시각화
fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for i, ax in enumerate(axes.flatten()):
    image, label = train_dataset[i]
    ax.imshow(image.squeeze(), cmap='gray')
    ax.set_title(f'Label: {label}')
    ax.axis('off')

plt.suptitle('MNIST 샘플 이미지')
plt.tight_layout()
plt.show()


# 신경망 모델 정의
# MLP (Multi-Layer-perceptron)
class MLP(nn.Module):
    '''
    MLP의 Docstring
    다층 퍼셉트론 

    구조 : 
    입력 (784)  -> FC1 (512) -> ReLu -> Dropout
                -> FC2 (256) -> ReLu -> Dropout
                -> FC3 (128) -> ReLu -> Dropout
                -> FC4 (10) -> ReLu -> Dropout
    '''

    def __init__(self,
        input_size=784, 
        hidden_sizes=[512, 256, 128],
        num_classes=10):
        '''
        __init__의 Docstring
        
        :param input_size: 입력 크기 (기본: 784 = 28x28)
        :param hidden_sizes: 은닉층 크기들 (기본:[512, 256, 128])
        :param num_classes: 출력 클래스 수 (기본 : 10 = 0~9)
        '''
        # 1. 부모 클래스 (nn.Module) 초기화 필수!
        super(MLP, self).__init__()

        # Flatten 층 정의
        self.flatten = nn.Flatten()
        # 역할 : [batch, 1, 28, 28] => [batch, 784]
        #  2D 이미지를 1D 벡터로 펼침

        # 3. 신경망 층들을 리스트로 구성
        layers = []
        # 층들을 순서대로 담을 리스트


        prev_size = input_size # 이전 층의 출력 크기 (처음엔 입력 크기)

        for hidden_size in hidden_sizes:
            # 완전연결층 (Fully Connected Layer)
            layers.append(nn.Linear(prev_size, hidden_size))

            # 활성화 함수
            layers.append(nn.ReLU())
            # Dropout (과적합 방지)
            layers.append(nn.Dropout(0.2)) # 과적합 방지

            prev_size = hidden_size

        # 출력층 추가
        layers.append(nn.Linear(prev_size, num_classes))

        # 4. Sequential로 층들을 하나의 모듈로 묶기
        self.network = nn.Sequential(*layers)


        def forward(self, x):
            '''
            순전파 forward의 Docstring
            
            :param self: 설명
            :param x: 입력 텐서 [batch, 1, 28,28]

            # Return :
                출력 텐서 [batch_size, 10]
            '''

            x = self.flatten(x)
            # 2. 신경망 통과
            x = self.network(x)

            # 결과 반환
            return x
        
# 모델 생성 및 GPU로 이동
model = MLP().to(device)
print(model)


# 파라미터 수 확인
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'총 파라미터: {total_params}')
print(f'학습 가능 파라미터: {trainable_params}')


# 학습 설정
# 손실 함수 (다중 분류)
criterion = nn.CrossEntropyLoss()

# 옵티마이저 (Adam)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습률 스케쥴러 (선택적)
scheduler = optim.lr_scheduler.StepLR(optimizer,
                                        step_size=5,
                                        gamma=0.5)

# 학습 및 평가 함수
def train_epoch(model, loader, criterion, optimizer, device):
    """한 에폭 학습"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # 순전파
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 통계
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """평가"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

