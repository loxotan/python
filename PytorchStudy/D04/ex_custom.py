## 클래스 이해하기

## 새로운 클래스 정의
#### 클래스 이름 : Person
#### 부모 클래스 : X
#### 속  성  들  : gender, height, weight
#### 기  능  들  : eat, sleep

class Person:
    ## 공통 속성 

    ## 개인별 속성을 저장하는 메서드
    ## 사람마다 속성은 다르니까 init으로 받아야 함
    def __init__ (self, name, gender, height, weight):
        # self = 번지수
        # self번지에 가서 gender에다가 입력받은 gender를 넣기
        self.name = name
        self.gender = gender
        self.height = height
        self.weight = weight
        print('__init__()')
    
    ## 사람이 하는 기능/행동 메서드
    ## self: 메서드를 호출한 인스턴스 정보
    # 음식은 사람의 속성이 아님 - 필요할 때 입력받으면 된다
    def eat(self, food): # self번지에서만 food를 호출하면 됨
        name = self.name
        print(f'{name}이/가 {food}을/를 먹는다.')

    def sleep(self, where): 
        name = self.name
        print(f'{name}이/가 {where}에서 잔다.')


## 새로운 클래스 정의
#### 클래스 이름 : Fireman
#### 부모 클래스 : Person
#### 속  성  들  : (gender, height, weight), job
####                  -> 부모 클래스에 있음
#### 기  능  들  : (eat, sleep), bulggugi
####                  -> 부모 클래스에 있음

class Fireman(Person): #부모 클래스에 없는 속성/기능만 추가
    def __init__(self, name, gender, height, weight, job):
        super().__init__(name, gender, height, weight)# 부모클래스에서 상속
        self.job = job
    
    def bulggugi(self):
        name = self.name
        print(f"{name}이/가 불을 껐다!")

## -----------------------------
## Person 인스턴스 즉, 객체 생성
## -----------------------------
Choi = Person('수영', 'M', 177, 95)
Hwang = Person('정수', 'M', 175, 70)
Shin = Fireman('예진', 'M', 200, 120, 'fire fighter')


## -----------------------------
## Person 인스턴스의 속성과 메서드 사용하기
## -----------------------------
## 객체변수명.속성명 => 현재 저장되어 있는 속성 읽어오기
print(Choi.gender)

## 속성 값 변경하기 - 객체변수명.속성명 = 새로운 값 
print(Hwang.height)
Hwang.height += 1.5
print(Hwang.height)

## 메서드 사용하기 - 객체변수명.메서드명()
Choi.eat('동근이네 찜닭')
Hwang.sleep('길바닥')
Shin.eat('소고기 덮밥')
Shin.bulggugi()

## 