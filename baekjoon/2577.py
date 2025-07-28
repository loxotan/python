a= int(input())
b= int(input())
c= int(input())

gop = str(a*b*c)
sutja = {i:0 for i in range(10)}
gop_list = list(map(int, list(gop)))
for n in gop_list:
    sutja[n] +=1
    
for value in sutja.values():
    print(value)