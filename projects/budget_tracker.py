import json

def add_expense(expenses, description, amount):
    expenses.append({"description":description, "amount":amount})
    print(f"Added expense: {description}, Amount: {amount}")

def get_total_expenses(expenses):
    return sum([expense["amount"] for expense in expenses])

def get_balance(budget, expenses):
    return budget - get_total_expenses(expenses)

def show_budget_detail(budget, expenses):
    print(f"Total budget: {budget}")
    print("Expenses: ")
    for expense in expenses:
        print(f"- {expense['description']} : {expense['amount']}")
    print(f"Total Spent: {get_total_expenses(expenses)}")
    print(f"Remaining budget: {get_balance(budget, expenses)}")

def load_budget_data(filepath):
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
            return data.get("initial_budget", 0), data.get("expenses", [])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading data: {e}")
        return 0, []

def save_budget_details(filepath, initial_budget, expenses):
    data = {
        'initial_budget': initial_budget,
        'expenses' : expenses
    }
    with open(filepath, 'w') as file:
        json.dump(data, file, indent = 4)


def main():
    print("Welcome to the budget tracker!")
    filepath = "budget_data.json"
    initial_budget, expenses = load_budget_data(filepath)
    if initial_budget == 0:
        initial_budget = float(input("Please enter your initial budget: "))
    budget = initial_budget
    
    while True:
        print("\nWhat would you like to do?")
        print("1. Add an expense")
        print("2. Show budget details")
        print("3. Exit")
        choice = input("Enter your choice (1/2/3): ")
        print("\n")
        
        if choice == '1':
            description = input("Enter expense description: ")
            amount = float(input("Enter expense amount: "))
            add_expense(expenses, description, amount)
        elif choice == '2':
            show_budget_detail(budget, expenses)
        elif choice == '3':
            save_budget_details(filepath, initial_budget, expenses)
            print("Exiting budget tracker. Goodbye.")
            break
        else:
            print("Invalid choice, please choose again.")
            
if __name__ == "__main__":
    main()