# 모듈
'''
파이썬 코드가 저장된 파일
함수, 변수, 클래스 등을 모아놓은 파일로 다른 프로그램에서 가져다 쓸 수 있다.

도구 상자 : 여러 도구(함수)를 모아둔 상자(모듈)
레고 블럭 : 필요한 블록(모듈)을 가져와 조립
요리 레시피 : 필요한 레시피(모듈)을 참고

특징
코드 재사용성 : 한 번 작성한 코드 여러 곳에서 사용
유지보수 : 기능별로 분리하여 관리가 쉬움
협업 : 팀원들과 코드 공유 편리
네임스페이스 : 이름 충돌 방지
'''

# 전체 모듈 가져오기
from mypackage import module_2
from mypackage import module_1
import calculate  # ctrl 클릭 => 해당 파일로 들어감


# 작성되어 있는 모듈
import math as m  # 별칭
import math
import random
import datetime


# 패키지(Package)
'''
모듈들을 모아놓은 디렉토리
관련된 모듈들을 체계적으로 관리 가능
'''
# from mypackage import module_2
# from mypackage import module_1 # 제일 위로 올라감

module_1.greet()  # 해당 모듈 값
module_2.hello()


result = calculate.add(10, 5)
print(result)

print(math.pi)
print(random.randint(1, 11))
print(datetime.datetime.now())

# 가상환경(새로운 공간)
# 프로젝트별로 독립적인 패키지 환경을 만들 수 있다.
# python -m venv myenv : 가상환경 생성 'myenv' = 이름
# source myenv/Scripts/activate : 가상환경 python화 / 활성화 (bash 사용)
# myenv/Scripts/activate :가상환경 python화 / 활성화
# deactivate : 비활성화

# pip install numpy 설치
# pip install pandas 설치

# pip
# 파이썬 패키지 관리자
# pip list : 설치된 pip 확인
