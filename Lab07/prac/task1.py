from ortools.sat.python import cp_model
model = cp_model.CpModel()

COLORS = {0: 'Red', 1: 'Blue', 2: 'Green'}

class MySolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._vars = variables
        self._count = 0
    def on_solution_callback(self):
        self._count += 1
        for name, var in self._vars.items():
            print(f"{name}: {COLORS[self.value(var)]}", end=" ")
        print()
    @property
    def count(self):
        return self._count

# variables
a = model.new_int_var(0, 2, 'a')
b = model.new_int_var(0, 2, 'b')
c = model.new_int_var(0, 2, 'c')
d = model.new_int_var(0, 2, 'd')
e = model.new_int_var(0, 2, 'e')

# constraints
model.add(a != b)
model.add(a != e)
model.add(b != c)
model.add(b != d)
model.add(c != d)
model.add(d != e)

# solve
solver = cp_model.CpSolver()
cb = MySolutionPrinter({
    'a': a,
    'b': b,
    'c': c,
    'd': d,
    'e': e,
})
solver.parameters.enumerate_all_solutions = True
solver.solve(model, cb)
print(f"Total Solutions: {cb.count}")
