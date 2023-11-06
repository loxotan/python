def recursion(s, l, r, d):
    if l >= r: return (1, d)
    elif s[l] != s[r]: return (0, d)
    else: return recursion(s, l+1, r-1, d+1)

def isPalindrome(s):
    return recursion(s, 0, len(s)-1, 1)

import sys
input = sys.stdin.readline

T = int(input())
for _ in range(T):
    X = input().rstrip()
    print(*isPalindrome(X))