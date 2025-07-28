import sys
input = sys.stdin.readline

T = int(input())
num = [int(input()) for _ in range(T)]
num.sort()

#산술평균
sansul = round(sum(num)/len(num))
print(sansul)

#중앙값
mean = num[len(num)//2]
print(mean)

#최빈값
dic = {}
for i in range(T):
    if num[i] in dic.keys():
        dic[num[i]] += 1
    else:
        dic[num[i]] = 1
find_max = max(list(dic.values()))
max_value = [k for k, v in dic.items() if v == find_max]
if len(max_value) > 1:
    print(sorted(max_value)[1])
else:
    print(sorted(max_value)[0])
    
#범위
rangee = max(num) - min(num)
print(rangee)
