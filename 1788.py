f_pos = [0] * 1_000_001
f_neg = [0] * 1_000_001

f_pos[1] = 1
f_neg[1] = 1
for i in range(2, 1_000_001):
    f_pos[i] = f_pos[i-1]%1_000_000_000 + f_pos[i-2]%1_000_000_000
    if i%2 == 0:
        f_neg[i] = -1*f_pos[i]
    else:
        f_neg[i] = f_pos[i]
    
N = int(input())
if N >= 0:
    ans = f_pos[N]
else:
    ans = f_neg[-N]

if ans >0:
    print(1, abs(ans)%1_000_000_000, sep = '\n')
elif ans <0:
    print(-1, abs(ans)%1_000_000_000, sep = '\n')
else:
    print(0, 0, sep ='\n')