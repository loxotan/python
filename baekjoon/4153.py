import sys
input = sys.stdin.readline

while True:
    num_list = list(map(int, input().split()))
    if num_list[0] == 0:
        break
    else:
        d = max(num_list)
        namuji = list(set(num_list)-{d})
        if d**2 == namuji[0]**2 + namuji[1]**2:
            print('right')
        else:
            print('wrong')
            
