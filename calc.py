def add(a, b):
    return a + b


def subtract(a, b):
    return a - b


def mulriply(a, b):
    return a * b


def divide(a, b):
    if b == 0:
        return '0으로 나눌 수 없습니다.', None
    return a / b
