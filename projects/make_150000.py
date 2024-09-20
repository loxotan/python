from itertools import combinations

# 입력된 데이터 예시
data = [80900, 100800, 106900, 96600, 105000, 93100, 35900, 101000, 80700, 105900, 96100, 41000, 97800, 129300, 95500, 42400, 140000, 78400, 10400]
answer = []

# 세 날짜 조합 만들기
all_combinations = combinations(data, 3)

# 조건에 맞는 경우 찾기
for case in all_combinations:
    if 150000 < sum(case) < 160000:
        answer.append((case, sum(case)))  # 튜플 형태로 조합과 합을 저장

# 합계를 기준으로 정렬
answer = sorted(answer, key=lambda x: x[1])

# 결과 출력
for combo, total in answer:
    print(f"조합: {combo}, 합계: {total}")
