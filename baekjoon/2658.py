import sys
input = sys.stdin.readline

graph = [[0]*11]
for _ in range(10):
    graph.append([0]+list(map(int, input().strip())))

# 직사각형 찾기
def find_rectangle(graph):
    L,R,U,D = 11, -1, 11, -1
    for i in range(1, 11):
        for j in range(1, 11):
            if graph[i][j] == 1:
                L = min(L , j)
                R = max(R , j)
                U = min(U , i)
                D = max(D , i)
    return L,R,U,D

def LU(a, b, k): # lower left: 좌상단에서 시작해서 a부터 a+k-i까지
    for i in range(0, k+1):
        for j in range(a, a+k-i+1):
            if graph[b+i][j] == 0:
                return False
    return True

def RU(a, b, k): # upper right, 좌상단에서 시작해서 a+i부터 a+k까지 
    for i in range(0, k+1):
        for j in range(a+i, a+k+1):
            if graph[b+i][j] == 0:
                return False
    return True

def LD(a, b, k): # lower left, 좌상단에서 시작해서 i부터 a+k까지
    for i in range(0, k+1):
        for j in range(a, a+i+1):
            if graph[b+i][j] == 0:
                return False
    return True

def RD(a, b, k): # lower right, 좌상단에서 시작해서 k-i부터 a+k까지
    for i in range(0, k+1):
        for j in range(a+k-i, a+k+1):
            if graph[b+i][j] == 0:
                return False
    return True

def main():
    pts = None
    L,R,U,D = find_rectangle(graph)
    if R == -1: # 직사각형이 없는 경우
        print(0)
        return
    
    if (R-L) == (D-U):
        # 정사각형일 때 가능한 이등변삼각형 경우 4가지
        # a: 좌변(L), b: 상단(U), k: 이등변삼각형의 한 변 길이(R-L+1)
        k = R-L
        if LU(L, U, k):
            pts = [(U, L), (U, R), (D, L)]
        elif RU(L, U, k):
            pts = [(U, L), (U, R), (D, R)]
        elif LD(L, U, k):
            pts = [(U, L), (D, L), (D, R)]
        elif RD(L, U, k):
            pts = [(U, R), (D, L), (D, R)]
        else:
            pts = None
    
    elif (R-L) == 2*(D-U):
        # 가로로 긴 직사각형일 때 가능한 이등변삼각형 경우 2가지
        # 1. UP = RD + LD, DOWN = RU + LU
        k = D-U
        if RD(L, U, k) and LD(L+k, U, k):
            pts = [(U, L+k), (D, L), (D, R)]
        elif RU(L, U, k) and LU(L+k, U, k):
            pts = [(U, L), (U, R), (D, L+k)]
        else:
            pts = None
    
    elif (D-U) == 2*(R-L):
        # 세로로 긴 직사각형일 때 가능한 이등변삼각형 경우 2가지
        # 1. LEFT = RD + RU, RIGHT = LD + LU
        k = R-L
        if RD(L, U, k) and RU(L, U+k, k):
            pts = [(U, R), (U+k, L), (D, R)]
        elif LD(L, U, k) and LU(L, U+k, k):
            pts = [(U, L), (U+k, R), (D, L)]
        else:
            pts = None
    
    else: # 직사각형이 정사각형도 아니고 2배 차이가 나는 것도 아니면 직각이등변삼각형 형성이 불가능
        pts = None
    
    if pts:
        pts.sort()
        for r, c in pts: print(r, c)
    else:
        print(0)

main()