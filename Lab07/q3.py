from ortools.sat.python import cp_model

model = cp_model.CpModel()

grid = {}
for row in range(6):
    for col in range(6):
        grid[(row, col)] = model.new_int_var(1, 9, f"cell_{row}_{col}")

puzzle = [
    [0, 0, 6, 2, 0, 5],
    [0, 0, 0, 4, 6, 0],
    [0, 1, 2, 0, 0, 0],
    [5, 6, 0, 0, 0, 4],
    [0, 0, 4, 3, 0, 2],
    [3, 0, 0, 5, 0, 6],
]

for row in range(6):
    for col in range(6):
        if puzzle[row][col] != 0:
            model.add(grid[(row, col)] == puzzle[row][col])
for row in range(6):
    model.add_all_different(grid[(row, col)] for col in range(6))
for col in range(6):
    model.add_all_different(grid[(row, col)] for row in range(6))
for box_row in range(0, 6, 2):
    for box_col in range(0, 6, 3):
        model.add_all_different(
            grid[(box_row + r, box_col + c)]
            for r in range(2)
            for c in range(3)
        )

solver = cp_model.CpSolver()
status = solver.solve(model)

if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
    print()
    for row in range(6):
        if row % 2 == 0 and row != 0:
            print()
        row_str = ""
        for col in range(6):
            if col % 3 == 0 and col != 0:
                row_str += "| "
            row_str += f"{solver.Value(grid[(row, col)])} "
        print(row_str)
    print()
else:
    print("No solution found.")