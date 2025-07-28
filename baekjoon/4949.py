import sys
import re

input = sys.stdin.readline

while True:
    # 글자, 공백 제거
    sentence = input().rstrip()
    if sentence == '.':
        break
    sentence = sentence.lower()
    sentence = re.sub('[a-z]','',sentence)
    sentence = re.sub(' ', '', sentence)
    
    #짝이 맞는 괄호 제거
    while True:
        if '[]' in sentence:
            sentence = sentence.replace('[]', '')
        elif '()' in sentence:
            sentence = sentence.replace('()', '')
        else:
            break
    
    if sentence == '.':
        print('yes')
    else:
        print('no')
    