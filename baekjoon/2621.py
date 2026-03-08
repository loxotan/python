import sys 
input = sys.stdin.readline

colors = []
numbers = []
for _ in range(5):
    c, n = input().split()
    colors.append(c)
    numbers.append(int(n))

score = 0

def is_straight(nums):
    nums.sort()
    return nums == list(range(nums[0], nums[0]+5))
# scoring method
if len(set(colors)) == 1: # flush
    if is_straight(numbers): # 1. straight flush
        score = 900 + max(numbers)
    else: # 4. flush
        score = 600 + max(numbers)
elif len(set(numbers)) == 2:
    for n in set(numbers): 
        if numbers.count(n) == 4: # 2. four of a kind
            score = 800 + n
            break
        elif numbers.count(n) == 3: # 3. full house
            score = 700 + 10*n + [x for x in set(numbers) if x != n][0]
            break
elif is_straight(numbers): # 5. straight
    score = 500 + max(numbers)
elif len(set(numbers)) == 3:
    for n in set(numbers):
        if numbers.count(n) == 3: # 6. three of a kind
            score = 400 + n
            break
        elif numbers.count(n) == 2: # 7. two pair
            pair1 = n
            pair2 = [x for x in set(numbers) if numbers.count(x) == 2 and x != n][0]
            score = 300 + 10*max(pair1, pair2) + min(pair1, pair2)
            break
elif len(set(numbers)) == 4: # 8. one pair
    for n in set(numbers):
        if numbers.count(n) == 2:
            score = 200 + n
            break
else: # 9. high card
    score = 100 + max(numbers)

print(score)