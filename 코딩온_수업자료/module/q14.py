# 실습 01. 회원 명부 작성하기
with open('member.txt', 'w', encoding='utf-8') as f:
    for i in range(3):
        ID = input('사용자의 이름을 입력하세요: ')
        PW = input('사용자의 비밀번호를 입력하세요: ')
        f.write(f'{ID} {PW}\n')

with open('member.txt', 'r', encoding='utf-8') as f:
    # for line in f:
    content = f.read()
    print('회원 명부:')  # print(line.split()[0]) -> 아이디만 출력
    print(f'{content}')  # print(line.split()) -> 전체 출력


# 실습 02. 회원 명부를 이용한 로그인 기능
with open('member.txt', 'r', encoding='utf-8') as f:
    in_ID = input('사용자의 이름을 입력하세요: ')
    in_PW = input('사용자의 비밀번호를 입력하세요: ')
    for line in f:
        ID, PW = line.strip().split()
        if ID == in_ID and PW == in_PW:
            print('로그인 성공')
            # 실습 03. 로그인 성공 시 전화번호 저장 -------------------------수정사항 github
            input_phone = input('전화번호를 입력하세요: ')

            members = {}  # dict 생성
            # try 시도하다. 에러가 발생하면 except로 처리하겠다.
            try:
                with open('member_tel.txt', 'r+', encoding='utf-8') as f2:
                    # 현재 f 함수? 범위 내이므로 f 사용 불가 (외부면 사용가능)
                    for line in f2:
                        saved_name, saved_phone = line.strip().split()
                    # 딕셔너리에 추가 (빈 dict 기준) key 값에 할당
                        members[saved_name] = saved_phone
            except:
                pass
            # 딕셔너리에 추가, 있으면 수정
            members[in_ID] = input_phone

            with open('member_tel.txt', 'w', encoding='utf-8') as f2:
                for name, phone in members.items():
                    f2.write(f'{name} {phone}\n')
            break
        else:
            print('로그인 실패')
