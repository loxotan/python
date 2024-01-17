n = int(input())
nums = list(map(int, input().split()))
buho = list(map(int, input().split()))

mini = int(1e9)
maxi = int(-1e9)

answer = nums[0]

def dfs(idx):
    global answer, mini, maxi
    
    if idx == n:
        if answer > maxi:
            maxi = answer
        if answer < mini:
            mini = answer
        return
    
    for i in range(4):
        tmp = answer
        if buho[i] > 0:
            if i == 0:
                answer += nums[idx]
            elif i == 1:
                answer -= nums[idx]
            elif i == 2:
                answer *= nums[idx]
            else:
                if answer >= 0:
                    answer //= nums[idx]
                else:
                    answer = (-answer//nums[idx]) * -1
                    
            buho[i] -= 1
            dfs(idx + 1)
            answer = tmp
            buho[i] += 1
        
dfs(1)
print(maxi)
print(mini)