def solution(s):
    sent = list(s.split())
    answer = []
    for word in sent:
        word = list(word)
        for i in range(len(word)):
            if i%2 == 0:
                word[i] = word[i].upper()
            else:
                word[i] = word[i].lower()
        answer.append(''.join(word))
        
    return " ".join(answer)

solution("try hello world")