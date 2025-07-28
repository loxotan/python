# 1
# 11 2
# 111 12 21 3
# 1111 112 121 211 22 13 31
# 11111 1112 1121 1211 2111 221 212 122 113 131 311 23 32
# 111111 11112 11121 11211 12111 21111 1113 1131 1311 3111 123 132 213 231 312 321 33

# 1 2 4 7 13 24 44

#1111 1 112 1 121 1 211 1 22 1 13 1 31 1
# 111 2

import sys
input = sys.stdin.readline

T = int(input())
n_list = []
for i in range(T):
    n_list.append(int(input()))
    
triple = {1:1, 2:2, 3:4}
for j in range(4, max(n_list)+1):
    triple[j] = triple[j-1]+triple[j-2]+triple[j-3]
    
for k in range(T):
    print(triple[n_list[k]])


