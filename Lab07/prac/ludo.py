from ortools.sat.python import cp_model
model = cp_model.CpModel()

COLORS = {0: "Red", 1: "Green", 2: "Blue", 3: "Yellow"}

class MySolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._vars = variables
        self._count = 0
    def on_solution_callback(self):
        self._count += 1
        for name, val in self._vars.items():
            print(f"{name}: {COLORS[self.value(val)]}", end=" ")
        print()
    @property
    def count(self):
        return self._count

#variables
p1 = model.new_int_var(0, 3, "p1")
p2 = model.new_int_var(0, 3, "p2")
p3 = model.new_int_var(0, 3, "p3")
p4 = model.new_int_var(0, 3, "p4")

# constraints
model.add_all_different([p1, p2, p3, p4])
model.add(p1 != 2)

# solver
solver = cp_model.CpSolver()
"""
status = solver.solve(model)
"""
cb = MySolutionPrinter({"p1": p1, "p2": p2, "p3": p3, "p4": p4})
solver.parameters.enumerate_all_solutions = True
solver.solve(model, cb)
print(f"Total solutions: {cb.count}")

"""
if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL:
    print(f"P1: {COLORS[solver.value(p1)]}, P2: {COLORS[solver.value(p2)]}, P3: {COLORS[solver.value(p3)]}, P4: {COLORS[solver.value(p4)]}")
else:
    print("Solution not found")
"""
