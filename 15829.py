import sys
input = sys.stdin.readline

T = int(input())
word = list(input().rstrip())

hash = 0
for i in range(T):
    hash += ((ord(word[i])-96)*(31**i))

print(hash)