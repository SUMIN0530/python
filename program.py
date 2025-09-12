# 분수의 덧셈 / 기약분수로 나타내기
import math


def solution(num1, den1, num2, den2):
    # 분자
    num = (num1*den2) + (num2*den1)
    # 분모
    den = den1*den2

    # 최대공약수
    gcd = math.gcd(num, den)   # 맨 위에 impert math(라이브러리) 안하면 오류

    while num != 0:  # 이 값이 나올때 까지 실행??
        num, den = den, num % den

    # 최소공배수
    num = num // gcd
    den = den // gcd

    answer = [num, den]
    return answer
