import sys
input = sys.stdin.readline

N, M = map(int, input().split())
plan = [list(map(int, input().split())) for _ in range(N)]
door = {}

for p in range(N):
    start,end = plan[p][1], plan[p][-1]
    door.update({start: door.get(start, 0)+1, end: door.get(end, 0)+1})
    
doors = sorted(door.items(), key = lambda x:(x[1], -x[0]), reverse = True)
print(doors[0][0])
    