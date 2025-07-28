from collections import deque

a, b = map(int, input().split())

def calc(a, b):
    q = deque([a])
    visited = {a: 1}  # 시작점 a를 1로 설정 (연산 횟수)

    while q:
        now = q.popleft()
        
        # 목표값 b에 도달하면 그때의 연산 횟수를 반환
        if now == b:
            return visited[now]
        
        # 가능한 두 연산
        for next in (now * 10 + 1, now * 2):
            if next <= 1000000000 and next not in visited:  # 범위 내에서 아직 방문하지 않은 경우
                visited[next] = visited[now] + 1  # 연산 횟수 증가
                q.append(next)

    return -1  # 목표값 b에 도달하지 못한 경우

print(calc(a, b))
