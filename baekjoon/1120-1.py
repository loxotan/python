a, b = map(int, input().split())

if a>b: #패티가 치즈보다 많으면 치즈 갯수에 맞춰서 패티 넣기
    print(2*b+1)
else: #치즈가 패티보다 많으면 패티 갯수에 맞춰서 패티 넣기
    print(2*a-1)