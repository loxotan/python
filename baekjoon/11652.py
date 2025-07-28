import sys
input = sys.stdin.readline

n = int(input())
card_count = {}

for i in range(n):
    card = int(input())
    if card in card_count:
        card_count[card] += 1
    else:
        card_count[card] = 1

card_count = sorted(card_count.items(), key= lambda x:(-x[1], x[0]))

print(card_count[0][0])