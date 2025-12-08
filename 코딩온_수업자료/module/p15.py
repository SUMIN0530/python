# 예외
'''
프로그램 실행 중 발생하는 예상치 못한 상황
예외가 발생하면 프로그램이 즉시 정지, 그러나 예외처리 => 프로그램 계속 실행

오류 VS 예외
 - 구문 오류 (Syntax Error) : 코드를 잘 못 작성한 경우
    프로그램 시작 불가
     코드 수정 -> 
ex) print('Hello'
     
 - 예외 : 문법은 맞으나 실생 중 발생
            프로그램 실행 중 발생
            try - except로 처리
ex) result = 10 / 0
'''
# 예외 처리가 필요한 이유
age = int(input('나이: '))  # ase 입력시 오류

# 예외 처리
while True:
    try:
        age = int(input('나이: '))
        # 윗 줄에서 예외 발생시 밑줄 코드 실행 안됨.
        break
    except:
        print('숫자로 입력하세요')

# try 블록은 최소한
name = input('이름: ')
try:
    # name = input('이름: ') 불필요
    age = input('나이: ')
    # print(f'안녕하세요 {name}님') 불필요
except:
    print('오류!')
print(f'안녕하세요 {name}님')

try:
    # result = int('abc') # value
    # new_list = [1, 2, 3, 4]
    # print(new_list[5]) # index
    # answer = 10 / 0
    # print(answer)       #zerodivision

    # try문 밖에서 작성 할 경우 터미널 오류문에 같이 출력
    num = int(input('숫자를 입력하세요: '))
    if num == 0:                                           # try문 안 => 오류났다고 출력문으로 출력
        raise ZeroDivisionError('0 에러가 발생했습니다.')
    result = 10 / num

# except: 모든 에러 잡음 (위험)
except ValueError:  # 특정 예외만
    print('값 오류 발생')
except IndexError:
    print('인덱스 범위를 초과했습니다.')
except ZeroDivisionError:
    print('0으로 나눌 수 없습니다.')
except Exception as e:  # 다른 예외는 로깅
    print(f'예상치 못한 오류: {e}')
else:
    # 정상적으로 끝나면 실행
    print('정상 작동 했습니다.')

finally:
    # 예외에 상관없이 무조건 실행
    print('끝났습니다.')

# ==========실습 문제=============
# 문제 1. 나이 입력 프로그램 ------- 근데 이제 반복이 빠진...


def get_age_group():
    try:
        age = int(input('나이를 입력하세요: '))

        if 0 <= age < 20:
            print('미성년자')
        elif 20 <= age < 36:
            print('청년')
        elif 36 <= age < 60:
            print('중년')
        elif 60 <= age < 151:
            print('노년')
        else:
            print('정상 범주를 초과했습니다.')

    except ValueError:
        print('숫자로 입력해주세요.')


print(get_age_group())

# 문제 2. 리스트 안전 접근
