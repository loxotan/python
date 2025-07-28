word = input()
jupmi = []
for i in range(len(word)):
    jupmi.append(word[i:])
    
jupmi.sort()

print(*jupmi, sep='\n')