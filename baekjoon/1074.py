import sys
input = sys.stdin.readline

def z_search(n, x, y, r, c):
    if n == 1:
        return 0
    
    else:
        mid = n//2
        if r<x+mid and c<y+mid:
            return z_search(mid, x, y, r, c)
        elif r<x+mid and c>=y+mid:
            return mid*mid + z_search(mid, x, y+mid, r, c)
        elif r>=x+mid and c<y+mid:
            return 2*mid*mid + z_search(mid, x+mid, y, r, c)
        else:
            return 3*mid*mid + z_search(mid, x+mid, y+mid, r, c)
        
n, r, c = map(int, input().split())
print(z_search(2**n, 0, 0, r, c))