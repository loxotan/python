n, k = map(int, input().split())
card = list(map(int, input().split()))

card.sort()

print(card[k-1])