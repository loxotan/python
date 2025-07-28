import sys
input = sys.stdin.readline

x1, y1, x2, y2 = map(int, input().split())
x3, y3, x4, y4 = map(int, input().split())

# CCW 함수를 이용하여 선분이 교차하는지 확인
def ccw(p1, p2, p3):
    return (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])

# 두 선분의 교차 여부를 확인하는 함수
def is_cross(x1, y1, x2, y2, x3, y3, x4, y4):
    p1, p2, p3, p4 = (x1, y1), (x2, y2), (x3, y3), (x4, y4)

    # ccw 계산
    ccw1 = ccw(p1, p2, p3) * ccw(p1, p2, p4)
    ccw2 = ccw(p3, p4, p1) * ccw(p3, p4, p2)

    if ccw1 <= 0 and ccw2 <= 0:
        # 선분이 겹치는지 확인 (평행한 경우)
        if max(x1, x2) >= min(x3, x4) and max(x3, x4) >= min(x1, x2) and max(y1, y2) >= min(y3, y4) and max(y3, y4) >= min(y1, y2):
            return True
    return False

if is_cross(x1, y1, x2, y2, x3, y3, x4, y4):
    print(1)
else:
    print(0)
