# 상속
'''
기존 클래스의 속성과 메서드를 물려받아 새로운 것을 만듦

동물 : 포유류 -> 개, 고양이,(공통 특징 : 자기, 먹기)
자동차 : 챠량 -> 승용차, 트럭
가족 : 부모 -> 자식
'''
# 상속 없이 - 코드 중복이 심각!

'''
class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def eat(self):
        print(f'{self.name}이가 먹습니다.')

    def sleep(self):
        print(f'{self.name}이가 잠을 잡니다.')

    def bark(self):
        print(f'{self.name}이가 멍멍 짖습니다.')

class Dog: (수정)
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def eat(self):
        print(f'{self.name}이가 먹습니다.')

    def sleep(self):
        print(f'{self.name}이가 잠을 잡니다.')

    def bark(self):
        print(f'{self.name}이가 멍멍 짖습니다.')
'''
# 추상 클래스

from abc import ABC, abstractmethod
class Animal:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def eat(self):
        print(f'{self.name}이가 먹습니다.')

    def sleep(self):
        print(f'{self.name}이가 잠을 잡니다.')

    def bark(self):
        print(f'{self.name}이가 멍멍 짖습니다.')


class Dog(Animal):
    def bark(self):
        print(f'{self.name}이가 멍멍 짖습니다.')


class Cat(Animal):
    def meow(self):
        print(f'{self.name}이가 야옹 웁니다.')


class Bird(Animal):
    def fly(self):
        print(f'{self.name}이가 날아다닙니다.')


dog1 = Dog('바둑이', 3)
dog1.eat()
dog1.sleep()
dog1.bark()

# 기본 문법과 용어


class 부모클래스:
    # 부모클래스 내용
    pass


class 자식클래스(부모클래스):  # 괄호 안에 부모클래스
    # 자식클래스 내용
    pass


'''
자식은 부모의 모든것을 물려받음
부모의 모든 속성과 메서드를 자동으로 사용가능
추가된 자신만의 속성과 메서드 정의 가능
'''


class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def greet(self):
        print(f'안녕하세요, {self.name}입니다.')


class Student(Person):  # 자식 클래스
    def study(self):
        print(f'{self.name}이가 공부합니다.')


class Teacher(Person):  # 자식 클래스
    def teach(self):
        print(f'{self.name}이가 수업합니다.')


student = Student('김학생', 20)
teacher = Teacher('박선생', 35)

# 부모 클래스 메서드 호출
student.greet()
teacher.greet()

# 자식 클래스만의 메서드 호출
student.study()
teacher.teach()

# super()와 생성자 상속
# 자식 클래스에서 부모 클래스에 접근 할 때

# super()없이 - 문제 발생!


class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
        print(f'Person 생성: {name} {age}살')

    def greet(self):
        print(f'안녕하세요. {self.name}입니다.')


class Student(Person):
    ''' 없어서 # print(student.name)오류 발생 
    def __init__(self, name, age):
        self.name = name
        self.age = age
        print(f'Person 생성: {name} {age}살')
    '''
    '''
    def __init__(self, name, age, student_id):
        # 부모클래스의 __init__을 호출하지 않음
        self.student_id = student_id
        print(f'Student 생성: 학번{student_id}')
    '''
    # 해결방안

    def __init__(self, name, age, student_id):
        # 부모 생성자 호출
        super().__init__(name, age)
        self.student_id = student_id
        print(f'Student 생성: 학번{student_id}')

    def greet(self):
        # super().greet() # 부모 greet() 먼저 호출
        print(f'학생입니다.')  # 위가 없으면 덮어쓰기


student = Student('김철수', 20, '20250001')
student.greet()
print(student.name)

# 메서드 오버라이딩
'''
오버라이딩
부모 클래스의 메서드를 자식 클래스에서 다시 정의
'''


class Animal:
    def make_sound(self):
        print(f'동물이 소리를 냅니다.')


class Dog(Animal):
    def make_sound(self):
        print(f'멍멍!')


class Cat(Animal):
    def make_sound(self):
        print(f'야옹~')


animals = [Dog(), Cat()]
for animal in animals:
    animal.make_sound()  # 각자 다른 소리!


class Shape:
    def __init__(self, name):
        self.name

    def area(self):
        return 0  # 기본값

    def info(self):
        print(f'{self.name}의 넓이: {self.area()}')


class Rectangle(Shape):
    def __init__(self, width, height):
        super().__init__('직사각형')
        self.width = width
        self.height = height


class Circle(Shape):
    def __init__(self, radius):
        super().__init__('원')
        self.radius = radius

# 내용 어디갔노


# 추상 클래스
'''
직접 객체를 만들 수 없고,
반드시 상속받아 완성 후 사용 가능한 미완성 설계도

동물 : 실제로 '동물'만 있는건 없고 개, 고양이, 새 등 구체적인 동물이 있음
악기 : 추상적 개념. 피아노, 기타, 드럼 등 구체적 개념이 있어야 연주 가능
'''

# 추상 클래스 없이


class Animal:
    def make_sound(self):
        pass  # 비어있음 - 구현 깜박


class Dog(Animal):
    def eat(self):
        print('강아지가 밥을 먹습니다.')


# 문제 발생
dog = Dog()
dog.make_sound()  # 아무것도 안 일어남 - 버그!

# 추상 클래스 사용
# from abc import ABC, abstractmethod 제일 위로 올라감.


class Animal(ABC):
    @abstractmethod
    def make_sound(self):
        pass  # 비어있음 - 구현 깜박


class Dog(Animal):   # <-----------------------이거이거
    def make_sound(self):
        pass  # 새로운 작성 필요

    def eat(self):
        print('강아지가 밥을 먹습니다.')


dog = Dog()  # 에러 발생 왜냐 make_sound 비정의 -> 자식 클래스에 오버라이딩? 필수

# 기본 사용법
# from abc import ABC, abstractmethod


class 추상클래스이름(ABC):  # ABC
    @abstractmethod
    def 추상메서드이름(self):
        print  # 내가 뭘 어케하는데 그지같음


        # animal = Animal() # 에러! 추상클래스는 직접 개체 생성 불가
dog = Dog()  # 추상 베서드를 모두 구현했으므로 가능


class Shape(ABC):
    # 추상클래스
    @abstractmethod
    def area(self):
        pass


class Circlle(Shape):
    def __init__(self, radius):
        # super().__init__()
        self.radius = radius

    def area(self):
        return 3.14 * self.radius * self.radius


# shape =Shape()
circle = Circle(5)
print(circle.area())


class Animal(ABC):  # 추상 클래스
    def __init__(self, name, age):
        self.name = name
        self.age = age

    # 일반 메서드 - 모든 동물이 공통으로 사용
    def sleep(self):
        print(f'{self.name}이가 잠을 잡니다.')

    def eat(self):
        print(f'{self.name}이가 먹이를 먹습니다.')

    # 추상 메서드 - 각 동물마다 다르게 구현해야 함
    @abstractmethod
    def make_sound(self):
        pass

    @abstractmethod
    def move(self):
        pass


class Dog(Animal):
    def make_sound(self):
        print(f'{self.name}: 멍멍!')

    def move(self):
        print(f'{self.name}이가 뚜어다닙니다.')
