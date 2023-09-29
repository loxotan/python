n, s = map(int, input().split())
lst = list(map(int, input().split()))

def jge(out, idx):
    global cnt
    if out and sum(out) == s:
        cnt += 1
        
    for i in range(idx, n):
        out.append(lst[i])
        jge(out, i+1)
        out.pop()
    
cnt = 0
out = []
jge(out,0)
print(cnt)