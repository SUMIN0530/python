# 스택(Stack)
'''
스택은 후입선출(LIFO) 원칙을 따르는 선형 자료 구조
가장 나중에 들어간 데이터가 가장 먼저 나오는 구조 (책 쌓아둔 형태)
스택은 한 쪽 끝(top)에서만 데이터의 삽입과 삭제가 일어난다.

핵심 특징
 - LIFO : Last In First Out
 - 제한된 접근 : 스택의 요소들은 오직 Top을 통해서만 가능 (중간 접근 X)
 - 주요 연산의 O(1) 시간 복잡도 : push(삽입) pop(삭제) 연산 모두 O(1)의 시간 복잡도를 가짐
 - 메모리의 효울성 : 동적 배열이나 연결 이스트로 구현 가능, 크기 조절 가능

 push(data) : 스택의 맨 위 요소 추가                        # O(1)
 pop() : 스택의 맨 위 요소 제거 및 반환                     # O(1)
 peek() / top() : 맨 위 요소 확인 (제거 X)                  # O(1)
 is_empty : 스택이 비어있는지 확인 (블리언으로 출력)         # O(1)
 size() : 스택의 요소 개수 반환                             # O(1)
 '''

# 리스트로 스택 만들기


class Stack:
    def __init__(self):
        # 스택 초기화
        self.items = []    # deque() 컬랙션 데큐를 이용해서 다시 만들기

    def push(self, item):
        # 요소 추가
        self.items.append(item)

    def pop(self):
        # 요소 제거 및 반환
        if not self.is_empty():
            return self.items.pop()
        else:
            raise IndexError('stack is empty')

    def peek(self):
        # 맨 위 요소 확인
        if not self.is_empty():
            return self.items[-1]
        else:
            raise IndexError('stack is empty')

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)

    def __str__(self):
        # 스택 출력
        return str(self.items)


stack = Stack()

stack.push(1)
stack.push(2)
stack.push(3)

print(f'스택 : {stack}')
print(f'pop : {stack.pop()}')
print(f'스택 : {stack}')
print(f'peek : {stack.peek()}')
print(f'스택 : {stack}')
print(f'스택 : {stack}')
print(f'스택 : {stack}')

# ==================================================================

# 큐 (Queue)
'''
큐(Queue)는 선입선출(FIFO) 원칙을 따르는 선형 자료 구조
가장 먼저 들어간 데이터가 가장 먼저 나오는 구조 (줄서기 형태)
한 쪽 끝(Rear)에서 삽입이 일어나고 디ㅏ른 쪽 끝(Front)에서 삭제가 일어남

핵심 특징
 - FIFO : First In First Out
 - 양 끝 접근 : 큐는 뒤(Rear)에서 삽입(enqueue), 앞(Front)에서 삭제(Dequeue)가 일어남
 - 순차적 처리 : 작업들은 순서대로 처리해야 할 때 유용
 - 공평한 자원 분배 : 먼저 요청한 작업이 먼저 처리되는 공정성을 보장
'''
'''
연산 | 설명 | 시간 복잡도  # github에서 받아오자... 저걸 어케 치고 앉아 있음
** enqueue(item) ** |
'''

# 비효율적 -> 데큐로 만들것


class ListQueue:
    def __init__(self):
        # 리스트 기반 큐
        self.items = []

    def enqueue(self, item):
        # 요소 추가
        self.items.append(item)

    def dequeue(self):
        # 요소 제거 - O(n) 시간복잡도 비효율적
        if not self.is_empty():
            return self.items.pop(0)
        raise IndexError('Queue is empty')

    def fornt(self):
        # 맨 앞 요소 확인
        if not self.is_empty():
            return self.items[0]
        raise IndexError('Queue is empty')

    def is_empty(self):
        # 비어있는지 확인
        return len(self.items) == 0

    def size(self):
        return len(self.items)

    def __str__(self):
        return str(self.items)


queue = ListQueue()  # 수정 필요
queue.enqueue(1)
queue.enqueue(2)
queue.enqueue(3)
print(f'Queue: {queue}')
queue.dequeue()
print(f'Queue: {queue}')
queue.enqueue(3)
print(f'Queue: {queue}')
