laser = list(input())
bar = 0
st = []

for i in range(len(laser)):
    if laser[i] == '(':
        st.append('(')
    else:
        if laser[i-1] == '(':
            st.pop()
            bar += len(st)
        else:
            st.pop()
            bar += 1
            
print(bar)