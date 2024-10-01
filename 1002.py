import sys
input = sys.stdin.readline

def dist(x1, y1, x2, y2):
    return ((x1-x2)**2+(y1-y2)**2)**(1/2)

def cases(x1, y1, r1, x2, y2, r2):
    # 두 원의 중심이 같은 경우
    if x1 == x2 and y1 == y2:
        if r1 == r2:
            return -1  # 원이 완전히 겹침
        else:
            return 0   # 동심원 (교점 없음)
    else:
        # 중심 간의 거리의 제곱
        d_sq = (x1 - x2) ** 2 + (y1 - y2) ** 2
        # 반지름 합과 차의 제곱
        sum_radii_sq = (r1 + r2) ** 2
        diff_radii_sq = (r1 - r2) ** 2

        if d_sq > sum_radii_sq:
            return 0  # 두 원이 떨어져 있음
        elif d_sq < diff_radii_sq:
            return 0  # 한 원이 다른 원 안에 있음 (접하지 않음)
        elif d_sq == sum_radii_sq:
            return 1  # 두 원이 외접함
        elif d_sq == diff_radii_sq:
            return 1  # 한 원이 다른 원에 내접함
        else:
            return 2  # 두 원이 두 점에서 만남

T = int(input())
for _ in range(T):
    x1, y1, r1, x2, y2, r2 = map(int, input().split())
    print(cases(x1, y1, r1, x2, y2, r2))
