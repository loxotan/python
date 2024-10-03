MOD = int(1e9+7)
n, k = map(int, input().split())

def fact_mod(n, mod):
    result = 1
    for i in range(2, n+1):
        result = (result * i) % mod
    return result

def mod_inverse(a, mod):
    return pow(a, mod-2, mod)

def calc(n, k, mod):
    bunmo = fact_mod(n, mod)
    bunja = (fact_mod(k, mod) * fact_mod(n-k, mod))%mod
    return(bunmo * mod_inverse(bunja, mod))%mod

print(calc(n, k, MOD))