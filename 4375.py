import sys
input = sys.stdin.readline

def ones(n):
    nums = '1'
    while True:
        if int(nums)%n == 0:
            print(len(nums))
            break
        else:
            nums = nums + '1'

while True:
    try:
        n = int(input())
        ones(n)
    except:
        break
    
 