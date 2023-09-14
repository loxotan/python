import sys
input = sys.stdin.readline

n = int(input())
line = list(map(int, input().split()))

bunho = 1
daegi = []

while len(line) != 0:
    #다음 번호표가 나올 때 까지 모두 대기줄로
    if len(line) > 0:
        if  line[0] == bunho:
            line.pop(0)
            bunho += 1
        elif len(daegi) >0 and daegi[-1] == bunho:
            daegi.pop()
            bunho += 1
        else:
            daegi.append(line.pop(0))
    
#대기줄에 제일 안에 있는 사람이 제일 마지막에 나갈 수 있으면 성공
if daegi == sorted(daegi, reverse=True):
    print('Nice')
else:
    print('Sad')
