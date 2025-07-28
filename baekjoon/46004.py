month, day = map(int, input().split())

days = 0
for i in range(1, month+1):
    if i in [2, 4, 6, 8, 9, 11]:
        days += 31
    elif i in [5, 7, 10, 12]:
        days += 30
    elif i == 3:
        days += 28

days += day

days_list = ['SUN', 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT']

for j in range(7):
    if (days%7 == j):
        print(days_list[j])