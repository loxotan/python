def distance(x1, y1, x2, y2):
    return(abs(x1-x2)+abs(y1-y2))

def check_dist(x1, y1, x2, y2, arr):
    result = False
    if distance(x1, y1, x2, y2) > 2: #2이상 떨어진경우 항상거짓
        return result
    elif distance(x1, y1, x2, y2) == 1: #붙어있는경우 항상참
        return True
    else: #2만큼 떨어져 있는 경우
        #같은 가로줄인 경우
        if y1 == y2:
            if arr[(x1+x2)//2][y1] == 'O':
                result = True
        #같은 세로줄인 경우
        elif x1 == x2:
            if arr[x1][(y1+y2)//2] == 'O':
                result = True
        #대각선인 경우
        else:
            if arr[x1][y2] == 'O' or arr[x2][y1] == 'O':
                result = True
    return result

def solution(places):
    answer = []
    for place in places:
        arr = [list(place[i]) for i in range(5)]
        people = []
        result = False
        for i in range(5):
            for j in range(5):
                if arr[i][j] == 'P':
                    people.append([i,j])
        
        for i in range(len(people)):
            for j in range(i+1, len(people)):
                x1, y1 = people[i]
                x2, y2 = people[j]
                if check_dist(x1, y1, x2, y2, arr):
                    answer.append(0)
                    result = True
                    break
            else:
                continue
            break
        
        if not result:
            answer.append(1)
                
    return answer
        
places = [["POOOP", "OXXOX", "OPXPX", "OOXOX", "POXXP"], ["POOPX", "OXPXP", "PXXXO", "OXXXO", "OOOPP"], ["PXOPX", "OXOXP", "OXPOX", "OXXOP", "PXPOX"], ["OOOXX", "XOOOX", "OOOXX", "OXOOX", "OOOOO"], ["PXPXP", "XPXPX", "PXPXP", "XPXPX", "PXPXP"]]
print(solution(places))