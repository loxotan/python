S = input().rstrip()
stack = []
ans = 0

for G in range(len(S)):
    if S[G] == '(':
        stack.append(G)
    elif S[G] == ')':
        if G - stack[-1] == 1:
            stack.pop()
            ans += len(stack)
        else:
            stack.pop()
            ans += 1
            
print(ans)