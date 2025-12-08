# 덱(Deque)
'''
덱(Deque, Dpuble_Ended Queue)
    양쪽 끝에서 삽입과 삭제 모두 가능
    스택과 큐의 특성을 모두 가지고 있어 매우 유연한 자료구조

특징
 - 양방향 연산(Double_ended)
    앞쪽(front) 뒤쪽(rear) 모두에서 요소 추가, 제거 가능
 - O(1) 시간 복잡도
    양쪽 끝에서의 모든 연산이 상수 시간에 수행된다.
 - 동적 크기
    필요에 따라 크기가 자동으로 조절
 - 스택과 큐 동시 구현
    하나의 자료 구조로 스택과 큐를 모두 구현
 - 회전 연산 지원
    요소들을 좌우로 회전시킬 수 있다.

주요 연산 
    append(x)         오른쪽 끝에 요소 추가
    appendleft(x)     왼쪽 끝에 요소 추가

    pop()             오른쪽 끝 요소 제거 및 반환
    popleft()         왼쪽 끝 요소 제거 및 반환

    extend(iterable)        오른쪽에 여러 요소 추가
    extendleft(iterable)    왼쪽에 여러 요소 추가

    rotate(n)       n만큼 회전
    clear()         모든 요소 제거

회문(palindrone) 검사
    level -> 처럼 앞으로, 뒤로 읽어도 똑같은 단어
'''
from collections import deque


def is_palindrome(s):
    '''덱을 이용한 회문 검사'''
    dp = deque(s)
    while len(dp) > 1:
        left_ch = dp.popleft()
        right_ch = dp.pop()
        if left_ch != right_ch:
            return False

    return True


is_palindrome('level')  # True
is_palindrome('tomato')  # False
