import random
from collections import deque


def iter_neighbors(size, row, col):
    for near_row in range(max(0, row - 1), min(size, row + 2)):
        for near_col in range(max(0, col - 1), min(size, col + 2)):
            if near_row == row and near_col == col:
                continue
            yield near_row, near_col


class Env:
    def __init__(self, size=5, num_bombs=5, seed=None):
        self.size = size
        self.num_bombs = num_bombs

        self.board = Generator(
            size=size,
            num_bombs=num_bombs,
            seed=seed
        ).generate_board()

        self.revealed = [
            [False] * size
            for _ in range(size)
        ]

        self.game_over = False
        self.won = False

    def display_board(self):
        print()

        for row in range(self.size):
            display_row = []

            for col in range(self.size):
                if not self.game_over and not self.revealed[row][col]:
                    display_row.append("X")
                elif self.board[row][col] == -1:
                    display_row.append("B")
                else:
                    display_row.append(str(self.board[row][col]))

            print(" ".join(display_row))

        print()

    def _neighbors(self, row, col):
        yield from iter_neighbors(self.size, row, col)

    def _flood_reveal_zeros(self, start_row, start_col):
        queue = deque([(start_row, start_col)])

        while queue:
            row, col = queue.popleft()

            for near_row, near_col in self._neighbors(row, col):
                if self.revealed[near_row][near_col]:
                    continue

                if self.board[near_row][near_col] == -1:
                    continue

                self.revealed[near_row][near_col] = True

                if self.board[near_row][near_col] == 0:
                    queue.append((near_row, near_col))

    def reveal_cell(self, row, col, announce=True):
        if self.game_over:
            if announce:
                print("Game over!")
            return

        if self.revealed[row][col]:
            if announce:
                print("That cell is already revealed.")
            return

        self.revealed[row][col] = True

        if self.board[row][col] == -1:
            self.game_over = True
            self.won = False
            if announce:
                print("You hit a bomb! Game over!")
            return

        if self.board[row][col] == 0:
            self._flood_reveal_zeros(row, col)

        if announce:
            print(
                f"Revealed cell ({row}, {col}) with "
                f"{self.board[row][col]} adjacent bombs."
            )

        if self.check_victory():
            self.game_over = True
            self.won = True
            if announce:
                print("Congratulations! You've revealed all safe cells!")

    def check_victory(self):
        for row in range(self.size):
            for col in range(self.size):
                is_safe = self.board[row][col] != -1
                is_hidden = not self.revealed[row][col]

                if is_safe and is_hidden:
                    return False

        return True


class Generator:
    def __init__(self, size=5, num_bombs=5, seed=None):
        if size <= 0:
            raise ValueError("size must be positive.")

        if not 0 <= num_bombs < size * size:
            raise ValueError(
                "num_bombs must be between 0 and size^2 - 1."
            )

        self.size = size
        self.num_bombs = num_bombs
        self.random = random.Random(seed)

    def generate_board(self):
        board = [
            [0] * self.size
            for _ in range(self.size)
        ]

        all_cells = [
            (row, col)
            for row in range(self.size)
            for col in range(self.size)
        ]

        bomb_positions = set(
            self.random.sample(all_cells, self.num_bombs)
        )

        for row, col in bomb_positions:
            board[row][col] = -1

        for row, col in bomb_positions:
            for near_row, near_col in iter_neighbors(self.size, row, col):
                if board[near_row][near_col] != -1:
                    board[near_row][near_col] += 1

        return board


class Player:
    def __init__(self, env):
        self.env = env

    def get_move(self):
        while True:
            try:
                row = int(
                    input(f"Enter row (0-{self.env.size - 1}): ")
                )
                col = int(
                    input(f"Enter column (0-{self.env.size - 1}): ")
                )

                if (
                    0 <= row < self.env.size
                    and 0 <= col < self.env.size
                ):
                    return row, col

                print("Invalid coordinates. Please try again.")

            except ValueError:
                print("Invalid input. Please enter integers only.")


class Solver:
    def __init__(self, env):
        self.env = env

    def _neighbors(self, row, col):
        yield from iter_neighbors(self.env.size, row, col)

    def _clone_revealed(self, revealed=None):
        source = self.env.revealed if revealed is None else revealed
        return [row[:] for row in source]

    def _mark_revealed(self, revealed, row, col):
        if self.env.board[row][col] == -1:
            return False

        queue = deque([(row, col)])
        while queue:
            cur_row, cur_col = queue.popleft()

            if revealed[cur_row][cur_col]:
                continue

            revealed[cur_row][cur_col] = True

            if self.env.board[cur_row][cur_col] != 0:
                continue

            for near_row, near_col in self._neighbors(cur_row, cur_col):
                if self.env.board[near_row][near_col] == -1:
                    continue
                if not revealed[near_row][near_col]:
                    queue.append((near_row, near_col))

        return True

    def _all_safe_revealed(self, revealed):
        for row in range(self.env.size):
            for col in range(self.env.size):
                if self.env.board[row][col] != -1 and not revealed[row][col]:
                    return False
        return True

    def _hidden_cells(self, revealed, flagged):
        hidden = set()

        for row in range(self.env.size):
            for col in range(self.env.size):
                if not revealed[row][col] and (row, col) not in flagged:
                    hidden.add((row, col))

        return hidden

    def _collect_constraints(self, revealed, flagged):
        constraints = []

        for row in range(self.env.size):
            for col in range(self.env.size):
                if not revealed[row][col]:
                    continue

                clue = self.env.board[row][col]
                if clue < 0:
                    continue

                unknown_neighbors = set()
                flagged_count = 0

                for near_row, near_col in self._neighbors(row, col):
                    if (near_row, near_col) in flagged:
                        flagged_count += 1
                    elif not revealed[near_row][near_col]:
                        unknown_neighbors.add((near_row, near_col))

                bombs_needed = clue - flagged_count

                if bombs_needed < 0:
                    return None

                if bombs_needed > len(unknown_neighbors):
                    return None

                if unknown_neighbors:
                    constraints.append((unknown_neighbors, bombs_needed))

        return constraints

    def _infer_from_constraints(self, constraints):
        safe_to_reveal = set()
        bombs_to_flag = set()

        for unknown_cells, bombs_needed in constraints:
            if bombs_needed == 0:
                safe_to_reveal.update(unknown_cells)
            elif bombs_needed == len(unknown_cells):
                bombs_to_flag.update(unknown_cells)

        for idx, (cells_a, bombs_a) in enumerate(constraints):
            for jdx, (cells_b, bombs_b) in enumerate(constraints):
                if idx == jdx:
                    continue

                if not cells_a.issubset(cells_b):
                    continue

                diff_cells = cells_b - cells_a
                diff_bombs = bombs_b - bombs_a

                if diff_bombs < 0 or diff_bombs > len(diff_cells):
                    return None

                if diff_bombs == 0:
                    safe_to_reveal.update(diff_cells)
                elif diff_bombs == len(diff_cells):
                    bombs_to_flag.update(diff_cells)

        return safe_to_reveal, bombs_to_flag

    def _infer_from_total_bomb_count(self, hidden, flagged):
        remaining_bombs = self.env.num_bombs - len(flagged)

        if remaining_bombs < 0:
            return None

        if remaining_bombs > len(hidden):
            return None

        if remaining_bombs == 0:
            return hidden, set()

        if remaining_bombs == len(hidden):
            return set(), hidden

        return set(), set()

    def _build_backtracking_model(self, initial_revealed):
        revealed = self._clone_revealed(initial_revealed)

        hidden_cells = [
            (row, col)
            for row in range(self.env.size)
            for col in range(self.env.size)
            if not revealed[row][col]
        ]

        cell_to_index = {
            cell: idx
            for idx, cell in enumerate(hidden_cells)
        }

        constraints = []

        for row in range(self.env.size):
            for col in range(self.env.size):
                if not revealed[row][col]:
                    continue

                clue = self.env.board[row][col]
                if clue < 0:
                    continue

                hidden_neighbors = [
                    cell_to_index[(near_row, near_col)]
                    for near_row, near_col in self._neighbors(row, col)
                    if not revealed[near_row][near_col]
                ]

                if clue < 0 or clue > len(hidden_neighbors):
                    return None

                if hidden_neighbors:
                    constraints.append(
                        {
                            "target": clue,
                            "cells": hidden_neighbors,
                        }
                    )
                elif clue != 0:
                    return None

        if self.env.num_bombs < 0 or self.env.num_bombs > len(hidden_cells):
            return None

        return hidden_cells, constraints

    def count_solutions(self, initial_revealed=None, max_solutions=2):
        model = self._build_backtracking_model(initial_revealed)
        if model is None:
            return 0

        hidden_cells, constraints = model
        variable_count = len(hidden_cells)

        if variable_count == 0:
            return 1 if self.env.num_bombs == 0 else 0

        constraint_data = [
            {
                "target": item["target"],
                "remaining": len(item["cells"]),
                "bombs": 0,
            }
            for item in constraints
        ]

        constraints_by_var = [[] for _ in range(variable_count)]
        for cidx, item in enumerate(constraints):
            for vidx in item["cells"]:
                constraints_by_var[vidx].append(cidx)

        assigned_bombs = 0
        solution_count = 0

        def backtrack(depth):
            nonlocal assigned_bombs, solution_count

            if solution_count >= max_solutions:
                return

            unassigned = variable_count - depth
            if assigned_bombs > self.env.num_bombs:
                return

            if assigned_bombs + unassigned < self.env.num_bombs:
                return

            if depth == variable_count:
                if assigned_bombs != self.env.num_bombs:
                    return

                for data in constraint_data:
                    if data["bombs"] != data["target"]:
                        return

                solution_count += 1
                return

            var_index = depth

            for value in (0, 1):
                if value == 1 and assigned_bombs + 1 > self.env.num_bombs:
                    continue

                changed = []
                valid = True

                for cidx in constraints_by_var[var_index]:
                    data = constraint_data[cidx]
                    data["remaining"] -= 1
                    data["bombs"] += value
                    changed.append(cidx)

                    bombs_now = data["bombs"]
                    bombs_max = data["bombs"] + data["remaining"]
                    target = data["target"]

                    if bombs_now > target or bombs_max < target:
                        valid = False
                        break

                if valid:
                    assigned_bombs += value
                    backtrack(depth + 1)
                    assigned_bombs -= value

                for cidx in changed:
                    data = constraint_data[cidx]
                    data["bombs"] -= value
                    data["remaining"] += 1

                if solution_count >= max_solutions:
                    return

        backtrack(0)
        return solution_count

    def has_unique_solution(self, initial_revealed=None):
        return self.count_solutions(
            initial_revealed=initial_revealed,
            max_solutions=2
        ) == 1

    def can_solve_without_guess(self, initial_revealed=None):
        revealed = self._clone_revealed(initial_revealed)
        flagged = set()

        while True:
            if self._all_safe_revealed(revealed):
                return True

            hidden = self._hidden_cells(revealed, flagged)

            constraints = self._collect_constraints(revealed, flagged)
            if constraints is None:
                return False

            local_inference = self._infer_from_constraints(
                constraints
            )
            if local_inference is None:
                return False

            safe_to_reveal, bombs_to_flag = local_inference

            global_inference = self._infer_from_total_bomb_count(
                hidden,
                flagged
            )
            if global_inference is None:
                return False

            global_safe, global_bombs = global_inference
            safe_to_reveal.update(global_safe)
            bombs_to_flag.update(global_bombs)

            progress = False

            new_bombs = bombs_to_flag - flagged
            if new_bombs:
                flagged.update(new_bombs)
                progress = True

            for row, col in safe_to_reveal:
                if revealed[row][col]:
                    continue

                if not self._mark_revealed(revealed, row, col):
                    return False

                progress = True

            if not progress:
                return False

    def minimize_clues(self, initial_clues=None, seed=None):
        if initial_clues is None:
            clues = [
                [self.env.board[row][col] != -1 for col in range(self.env.size)]
                for row in range(self.env.size)
            ]
        else:
            clues = [row[:] for row in initial_clues]

        clue_cells = [
            (row, col)
            for row in range(self.env.size)
            for col in range(self.env.size)
            if clues[row][col] and self.env.board[row][col] != -1
        ]

        rng = random.Random(seed)
        rng.shuffle(clue_cells)

        for row, col in clue_cells:
            clues[row][col] = False

            # Keep the removal only when puzzle remains solvable and unique.
            if (
                not self.can_solve_without_guess(initial_revealed=clues)
                or not self.has_unique_solution(initial_revealed=clues)
            ):
                clues[row][col] = True

        return clues


def make_logical_puzzle_env(size=5, num_bombs=5, max_attempts=500, seed=None):
    rng = random.Random(seed)

    for _ in range(max_attempts):
        board_seed = rng.randrange(10**9)
        clue_seed = rng.randrange(10**9)

        env = Env(size=size, num_bombs=num_bombs, seed=board_seed)

        solver = Solver(env)

        clues = solver.minimize_clues(seed=clue_seed)

        has_hidden_safe_cell = any(
            env.board[row][col] != -1 and not clues[row][col]
            for row in range(size)
            for col in range(size)
        )

        if not has_hidden_safe_cell:
            continue

        if (
            solver.can_solve_without_guess(initial_revealed=clues)
            and solver.has_unique_solution(initial_revealed=clues)
        ):
            env.revealed = clues
            env.game_over = False
            env.won = False
            return env, solver

    raise RuntimeError(
        "Could not generate a minimal no-guess puzzle within max_attempts."
    )


def main():
    env, _solver = make_logical_puzzle_env(size=8, num_bombs=10)
    player = Player(env)

    env.display_board()

    while not env.game_over:
        row, col = player.get_move()
        env.reveal_cell(row, col)
        env.display_board()


if __name__ == "__main__":
    main()