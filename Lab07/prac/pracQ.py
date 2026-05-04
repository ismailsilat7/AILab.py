from ortools.sat.python import cp_model
model = cp_model.CpModel()

MODULES = {
    0: 'AI',
    1: 'Web',
    2: 'Mobile',
    3: 'Cloud',
    4: 'Security',
    5: 'Testing',
}

# variables
d1 = model.new_int_var(0, 5, 'Developer 1')
d2 = model.new_int_var(0, 5, 'Developer 2')
d3 = model.new_int_var(0, 5, 'Developer 3')
d4 = model.new_int_var(0, 5, 'Developer 4')
d5 = model.new_int_var(0, 5, 'Developer 5')
d6 = model.new_int_var(0, 5, 'Developer 6')

# constraints
model.add_all_different([d1, d2, d3, d4, d5, d6])
model.add(d3 != 3)
model.add_allowed_assignments([d4], [[v] for v in [0, 3, 4]])

b = model.new_bool_var('d2_is_web')
model.add(d2 == 1).only_enforce_if(b)
model.add(d2 != 1).only_enforce_if(b.Not())
model.add(d3 == 2).only_enforce_if(b)

d = model.new_bool_var('d5_is_security')
model.add(d5 == 4).only_enforce_if(d)
model.add(d5 != 4).only_enforce_if(d.Not())
model.add(d6 != 5).only_enforce_if(d)

class MySolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._vars = variables
        self._count = 0
    def on_solution_callback(self):
        self._count += 1
        for name, val in self._vars.items():
            print(f"{name}: {MODULES[self.value(val)]}", end=" ")
        print()
    @property
    def count(self):
        return self._count

solver = cp_model.CpSolver()
cb = MySolutionPrinter({
    'Developer 1': d1,
    'Developer 2': d2,
    'Developer 3': d3,
    'Developer 4': d4,
    'Developer 5': d5,
    'Developer 6': d6,
})
solver.parameters.enumerate_all_solutions = True
solver.solve(model, cb)
print(f"Total solutions: {cb.count}")