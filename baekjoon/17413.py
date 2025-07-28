import sys
input = sys.stdin.readline

S = input().rstrip()
stack = []
ans = []
for i in range(len(S)):
    if S[i] == ' ' and '<' not in stack:
        ans.append(''.join(stack[::-1]))
        ans.append(S[i])
        stack.clear()
    elif S[i] == '>':
        stack.append(S[i])
        ans.append(''.join(stack))
        stack.clear()
    elif S[i] =='<' and stack:
        ans.append(''.join(stack[::-1]))
        stack.clear()
        stack.append(S[i])
    else:
        stack.append(S[i])
        
if len(stack) != 0:
    ans.append(''.join(stack[::-1])) 
    
print(*ans, sep = '')
