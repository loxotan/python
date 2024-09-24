import sys
input = sys.stdin.readline

m = int(input())
s = set()
for _ in range(m):
    command = input().rstrip()
    if command =='all':
        s = set(range(1, 21))
    elif command == 'empty':
        s = set()
    else:
        cmd, x = command.split()
        x = int(x)
        
        if cmd == 'add':
            s.add(x)
        elif cmd == 'remove':
            try:
                s.remove(x)
            except:
                continue
        elif cmd == 'check':
            if x in s:
                print(1)
            else:
                print(0)
        elif cmd == 'toggle':
            if x in s:
                s.remove(x)
            else:
                s.add(x)
