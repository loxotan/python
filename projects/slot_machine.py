import random

MAX_LINES = 3
ROWS = 3
COLS = 3

symbol_count = {"A":1, "B":2, "C":4, "D":6}
symbol_value = {"A":10, "B":5, "C":4, "D":3}

def deposit():
    while True:
        balance = input("얼마를 가져오셨나요?: ")
        if balance.isdigit():
            balance = int(balance)
            if balance > 0:
                break
            else:
                print("잔고가.. 마이너스에요.")
        else:
            print("숫자를 입력하세요.")
    return balance

def choose_line():
    while True:
        lines = input(f"몇 줄 거시겠어요(1 - {MAX_LINES})?: ")
        if lines.isdigit():
            lines = int(lines)
            if not 1<= lines <= MAX_LINES:
                print(f"{MAX_LINES}줄 이상은 없어요!")
            else:
                break
        else:
            print("숫자를 입력하세요.")
    return lines

def betting(balance, lines):
    while True:
        bet = input(f"줄 당 얼마를 거시겠어요?: ")
        if bet.isdigit():
            bet = int(bet)
            if not 1<= bet * lines <= balance:
                print(f"돈이 모자라시네요. 현재 잔고는 ${balance}입니다.")
            else:
                break
        else:
            print("숫자를 입력하세요.")
    return bet

def slot_machine_roll(symbol_count, rows, cols, lines, bet):
    slots = []
    all_symbols = []
    for symbol, count in symbol_count.items():
        for _ in range(count):
            all_symbols.append(symbol)
        
    for _ in range(cols):
        col = []
        current_symbols = all_symbols[:]
        for _ in range(rows):
            c = random.choice(current_symbols)
            col.append(c)
            current_symbols.remove(c)
        slots.append(col)
        
    show_slots(slots, rows, cols)
    total_win = check_winning(slots, cols, lines, bet)
    return total_win

def show_slots(slots, rows, cols):
    for row in range(rows):
        for col in range(cols):
            if col == cols -1:
                print(f"{slots[col][row]}")
            else:
                print(f"{slots[col][row]}", end = " | ")

def check_winning(slots, cols, lines, bet):
    total_win = 0
    for i in range(lines):
        base = slots[0][i]
        for j in range(cols):
            if slots [j][i] != base:
                break
        else:
            total_win += bet * symbol_value[base]
    return total_win - bet * lines

def spin(balance):
    lines = choose_line()
    bet = betting(balance, lines)
    print(f"{lines}줄에 ${bet}씩 거셨군요...")
    total_win = slot_machine_roll(symbol_count, ROWS, COLS, lines, bet)
    if total_win >0:
        print(f"${total_win}을 땄습니다! 총 잔고는 {balance + total_win} 입니다.")
    else:
        print(f"${total_win}을 잃었습니다! 총 잔고는 {balance + total_win} 입니다.")
    return balance + total_win
    

def main():
    print("슬롯 머신에 오신 것을 환영합니다.")
    balance = deposit()
    while True:
        answer = input("게임을 시작하려면 아무 키나 누르세요 (q to quit): ")
        if answer == 'q':
            print(f"게임을 종료합니다. 당신은 ${balance}로 마쳤습니다.")
            break
        else:
            balance = spin(balance)

main()