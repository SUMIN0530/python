# 파일 입출력 (File I/O)
'''
프로그램이 파일을 읽고(input) 쓰는(output) 작업
프로그램이 종료되어도 데이터를 보관할 수 있는 유일한 방법
프로그램의 데이터는 메모리에 저장되는데 프로그램이 종료되면 메모리의 데이터는 사라짐
파일로 저장하면 하드디스크에 영구 보관

파일 입출력 필요 상황
 - 설정 팡리 저장 : 게임 설정, 프로그램 옵션
 - 데이터 백업 : 중요한 정보 보관
 - 로그 기록 : 프로그램 실행 기록, 에러 추적
 - 데이터 교화 : 엑셀, csv 파일로 다른 프로그램과 데이터 공유
 - 대용량 처리 : 메모리에 다 못담는 빅데이터 처리
'''

# 위험한 방법 - 파일을 안 닫을 수 있음
print('=번거러운 방법=')
# 1단계. 파일 열기(open) - 파일과 연결 통로 생성
# data.txt 파일이 큰 폴더(python)에 있어야 되는건가? 그런거 같은디 꼭 같은 파일에 있을 필요 X
file = open('data.txt', 'r', encoding='utf-8')

# 2단계. 파일 작업(Read/Write) - 데이터 읽기/쓰기
content = file.read()
print(content)

# 3단계. 파일 닫기(close) - 연결 종료(중요!!!)
file.close()
print()

# 안전한 방법 - with문 (권장!)
print('=안전한 방법=')
with open('data.txt', 'r', encoding='utf-8') as f:
    content = f.read()
    print(content)
# 자동으로 close() 됨.
print()

# 새 파일 생성 또는 덮어쓰기
with open('output.txt', 'w', encoding='utf-8') as f:  # 새 파일 생성
    f.write('Hello, World! \n')
    f.write('파이썬 파일\n')  # \n없으면 줄바꿈 X

with open('output.txt', 'w', encoding='utf-8') as f:  # 파일 덮어쓰기
    f.write('추가된 내용(덮어쓰기)\n')

with open('output.txt', 'a', encoding='utf-8') as f:  # w를 a로 변경
    f.write('진짜 추가된 내용 \n')

# 1. read() -파일 전체를 하나의 문자열로
# 메모리 비효율적
with open('data.txt', 'r', encoding='utf-8') as f:
    content = f.read()  # 전체 내용
    print(content)      # 10GB면 전부 사용
print()

# read(크기) - 저장한 크기만큼만
print('read(크기) - 저장한 크기만큼만')
with open('data.txt', 'r', encoding='utf-8') as f:
    print(f'처음 위치: {f.tell()}')  # 일반적으로 0
    content = f.read(3)  # 3(크기) 만큼만
    print(content)
    print(f'3바이트 읽은 후 위치: , {f.tell()}')
print()

# 2. readlline() - 한 줄 씩 읽기
# 메모리 효율적
print('readline() - 한 줄 씩 읽기')
with open('data.txt', 'r', encoding='utf-8') as f:
    print(f'처음 위치: {f.tell()}')  # 일반적으로 0
    line1 = f.readline()
    print(line1.strip())  # 공백, 탭(\t), 줄바꿈(\n) 양쪽?
    print(f'첫 줄 읽은 후 위치: , {f.tell()}')
    line2 = f.readline()
    print(line2.strip())
    print(f'둘째 줄 읽은 후 위치: , {f.tell()}')
    f.seek(0)  # 커서 처음으로
    line3 = f.readline()
    print(line3.strip())
    print(f'섯째 줄 읽은 후 위치: , {f.tell()}')
    f.seek(18)               # 해당 바이트위치로 이동 => 두 번째 줄과 동일 (이해 불가)
    line3 = f.readline()
    print(line3.strip())
    print(f'섯째 줄 읽은 후 위치: , {f.tell()}')
    line4 = f.readline()     # 세 번째 줄 출력
    print(line4.strip())
print()

# readline() - for문
print('2. readline() - for문')
with open('data.txt', 'r', encoding='utf-8') as f:
    for line in f:          # 한 줄 씩만 메모리에 사용 (권장)
        print(line.strip())
print()

# 3. readlines()
print('readlines()')
with open('data.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()  # ['첫줄\n', '둘째줄\n', '셋째줄\n']

    for line in lines:
        print(line.strip())  # ()없으면 해당파일 주소 출력
print()

# 이미지 어쩌고
with open('이미지 파일명', 'rb') as f1:
    img = f1.read()

with open('./output(폴더가 없어서 안만들어질것-지우면됨)/이미지_copy 파일명', 'wb') as f2:
    f2.write(img)
