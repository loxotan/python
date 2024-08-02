import os

def is_one_char_diff(s1, s2):
    """Returns True if s1 and s2 have exactly one character different."""
    if len(s1) != len(s2):
        return False
    diff_count = sum(1 for a, b in zip(s1, s2) if a != b)
    return diff_count == 1

def check_duplicates(folder_path):
    files = os.listdir(folder_path)
    name_to_numbers = {}
    number_to_names = {}

    for file in files:
        if " " in file:
            name, number = file.rsplit(" ", 1)
            name_to_numbers.setdefault(name, []).append(number)
            number_to_names.setdefault(number, []).append(name)

    name_conflicts = []
    number_conflicts = []

    # Check for name conflicts
    for name, numbers in name_to_numbers.items():
        for i in range(len(numbers)):
            for j in range(i + 1, len(numbers)):
                if is_one_char_diff(numbers[i], numbers[j]):
                    name_conflicts.append((f"{name} {numbers[i]}", f"{name} {numbers[j]}"))

    # Check for number conflicts
    for number, names in number_to_names.items():
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if is_one_char_diff(names[i], names[j]):
                    number_conflicts.append((f"{names[i]} {number}", f"{names[j]} {number}"))

    return name_conflicts, number_conflicts

# Example usage:
folder_path = r'Y:\환자별'
name_conflicts, number_conflicts = check_duplicates(folder_path)

print("Name conflicts (same name, one digit different number):")
for conflict in name_conflicts:
    print(conflict)

print("\nNumber conflicts (same number, one character different name):")
for conflict in number_conflicts:
    print(conflict)
