# 실습 01. class 기본 문법
# 문제 1. 책 클래스 만들기
from abc import ABC, abstractmethod


class book:
    def __init__(self, title, author, total_pages):
        self.title = title
        self.author = author
        self.total_pages = total_pages
        self.current_page = 0

    def read_page(self, pages):
        # 현재 페이지 읽음, 총 페이지 수를 넘지 않도록 처리
        if pages < 0:
            return
        self.current_page = min(self.total_pages, self.current_page + pages)
        return self.current_page

    def progress(self):
        # 전체에서 얼마나 읽었는지 %로 소수점 1자리까지 출력
        pct = (self.current_page / self.total_pages) * 100
        return round(pct, 1)  # round : 소수점 한 자리까지 출력하는 내장 함수

    def __repr__(self):
        return f"<book {self.title} by {self.author}>"


# 객체 생성
print('책정보')
b = book('파이썬 클린코드', '홍길동', total_pages=320)
print(b)
print('현재까지 읽은 페이지:', b.read_page(100), '쪽')
print('진척도:', b.progress(), '%')
print()
b.read_page(124)
print(b.progress(), '%')

# 실습 01
# 문제 2. Rectangle 클래스 구현


class Rectangle:
    # 생성자
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        # 사각형 넓이 반환 함수
        return self.width * self.height


w = int(input('너비 입력:'))
h = int(input('높이 입력:'))

rectangle = Rectangle(w, h)
print('사각형의 넓이: ', rectangle.area())

# 실습 02. 클래스 변수, 메서드 연습
# 문제 1. User 클래스 구현


class User:
    total_users = 0

    def __init__(self, username):
        self.username = username
        self.points = 0
        '''
        self.usernums = User.total_users + 1
        '''
        # 클래스 변수 업데이트
        User.total_users += 1

    def add_points(self, amount):
        # 포인트 추가 메서드
        if amount > 0:
            self.points += amount

    def get_level(self):
        # 포인트 기준 레벨 반환
        if 0 <= self.points <= 99:
            return 'Bronze'
        elif 100 <= self.points <= 499:
            return 'Silver'
        elif 500 <= self.points:
            return 'Gold'
        else:
            return

    @classmethod
    def get_total_users(cls):
        print(f'총 유저 수 : {cls.total_users}')


user1 = User('김철수')
user2 = User('홍길동')
user3 = User('이영희')
user1.add_points(23)
user2.add_points(132)
user3.add_points(576)
print(f'{user1.username}님의 등급: ', user1.get_level())
print(f'{user2.username}님의 등급: ', user2.get_level())
print(f'{user3}님의 등급: ', user3.get_level())
# 이러면 주소로 뜨는데 왜 뜨더라?
# 객체 자체는 클래스와 동등한 위치 + 주소를 가짐
# 때문에 출력시 주소 도출
# 클래스 내부 username이 필요하므로 .username 사용

User.get_total_users()  # 3
'''
del user2

User.get_total_users()  # 3 : 삭제한다고 총 인원수가 줄어들지 않음.

줄어들게 하기 위해서 
def __del__(self):
User.total_users -= 1
'''

# ========================
# 실습 03. 접근 제어와 정보 은닉 연습 (getter / setter 사용)
# 문제 1. UserAccount 클래스 : 비밀번호 보호


class UserAccount:
    def __init__(self, username, password):
        self.username = username  # public 변수
        self.__password = password  # private 변수

    def change_password(self, old_pw, new_pw):
        # 현재 비밀번호가 old와 같을 때 변경 허용
        # 틀리면 '비밀번호 불일치'
        if self.__password == old_pw:
            self.__password = new_pw
            print('비밀번호 변경 성공')
        else:
            print('비밀번호 불일치')

    def check_password(self, password):
        # 비밀번호 일치 여부 반환(True/False)
        return self.__password == password


user1 = UserAccount('sumin', '1234')
print(user1.check_password('1234'))
print(user1.check_password('12345'))
print(user1.change_password('1234', '13579'))
print()

# 문제 2.


class Student:
    def __init__(self, score=0):
        self.__score = score
    '''
        def get_score(self):
        return self.__score
    '''
    @property
    def score(self):
        return self.__score
    '''
    def set_score(self, score):
        if 0 <= score <= 100:
            self.__score = score
        else:
            raise ValueError('점수는 0 이상 100 이하만 허용됩니다.')
    '''
    @score.setter
    def score(self, value):
        if 0 <= value <= 100:
            self.__score = value
        else:
            raise ValueError('점수는 0 이상 100이하만 허용됩니다.')
        self.__score = value


s1 = Student(90)
# print(s1.get_score())
print(s1.score)
# s1.set_score(80)
s1.score = 80
# print(s1.get_score())
print(s1.score)
# s1.set_score(120)
# s1.score = 120
# print(s1.score)


# 실습 04. 상송과 오버라이딩 연습
# 문제 1. Shape 클래스 오버라이딩


class Shape:
    def __init__(self, sides, base):
        self.sides = sides
        self.base = base

    def printinfo(self):
        print(f'변의 개수{self.sides}')
        print(f'밑변의 길이{self.base}')

    def area(self):
        print('넓이 계산이 정의되지 않았습니다.')


a = Shape(4, 5)
a.printinfo()
a.area()


class Rectangle(Shape):
    def __init__(self, sides, base, height):
        super().__init__(sides, base)
        self.height = height

    def area(self):
        print('사각형 넓이')
        print(self.base * self.height)


b = Rectangle(4, 5, 6)
b.area()


class Triangle(Shape):
    def __init__(self, sides, base, height):
        super().__init__(sides, base)
        self.height = height

    def area(self):
        print('삼각형 넓이')
        print(self.base * self.height / 2)


c = Triangle(2, 3, 4)
c.area()

# 실습 05. 추상 클래스 연습문제


class Payment(ABC):
    @abstractmethod  # 자식 클래스에 미포함
    def pay(self, amount):
        pass


class CardPayment(Payment):
    def pay(self, amount):
        print(f'카드로 {amount}원을 결제합니다.')


class CashPayment(Payment):
    def pay(self, amount):
        print(f'현금으로 {amount}원을 결제합니다.')


card = CardPayment()
cash = CashPayment()
card.pay(32000)
cash.pay(12000)
