import sys
input = sys.stdin.readline

def bkt(cnt, idx):
    if cnt == l:
        vo, co = 0, 0
        
        for i in range(l):
            if ans[i] in vowel:
                vo += 1
            else:
                co += 1
                
        if vo >= 1 and co >= 2:
            print(''.join(ans))
            
        return

    for i in range(idx, c):
        ans.append(words[i])
        bkt(cnt+1, i+1)
        ans.pop()
        
l, c = map(int, input().split())
words = sorted(list(map(str, input().split())))
vowel = ['a', 'e', 'i', 'o', 'u']
ans = []
bkt(0, 0)