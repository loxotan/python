N = int(input())

# [sin^2, cos^2, sincos]
esc = [[0,0,0] for _ in range(31)]
esc[0] = [0,0,1]
esc[1] = [-1, 1, 1]

for n in range(2, 31):
    esc[n][0] = esc[n-1][0] - esc[n-1][2]
    esc[n][1] = esc[n-1][1] + esc[n-1][2]
    esc[n][2] = 2*esc[n-1][0] -2*esc[n-1][1] + esc[n-1][2]

print(sum(esc[N]))
