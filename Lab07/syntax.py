from ortools.sat.python import cp_model
model = cp_model.CpModel()

x = model.new_int_var(0, 5, 'x')

model.add(x != y)

model.add_all_different([x,y,z])

model.add(x == 7)

model.add(-2*x + 3*y <= 20)

model.add_allowed_assignments([day1], [[v] for v in [2,3,4]])

solver = cp_model.CpSolver()

status = solver.solve(model)

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print(solver.value(x))

class MySolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._vars = variables
        self._count = 0
    def on_solution_callback(self):
        self._count += 1
        for name, var in self._vars.items():
            print(self.value(var))

solver.parameters.enumerate_all_solutions = True
cb = MySolutionPrinter({
    "x": x
})
solver.solve(model, cb)
