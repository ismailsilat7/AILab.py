from ortools.sat.python import cp_model
model = cp_model.CpModel()

SHIFTS = {
    0: 'Morning',
    1: 'Afternoon',
    2: 'Evening',
    3: 'Night',
    4: 'Emergency'
}

n1 = model.new_int_var(0, 4, 'Nurse 1')
n2 = model.new_int_var(0, 4, 'Nurse 2')
n3 = model.new_int_var(0, 4, 'Nurse 3')
n4 = model.new_int_var(0, 4, 'Nurse 4')
n5 = model.new_int_var(0, 4, 'Nurse 5')

model.add_all_different([n1, n2, n3, n4, n5])
model.add_allowed_assignments([n1], [[v] for v in [0, 1, 2]])
model.add_allowed_assignments([n2], [[0], [1]])
model.add(n5 == 4)

# Conditional 1: if N4 works Afternoon(1), then N3 cannot work Evening(2)
b = model.new_bool_var('n4_is_afternoon')
model.add(n4 == 1).only_enforce_if(b)
model.add(n4 != 1).only_enforce_if(b.Not())
model.add(n3 != 2).only_enforce_if(b)

# Conditional 2: if N1 works Morning(0), then N2 must work Afternoon(1)
d = model.new_bool_var('n1_is_morning')
model.add(n1 == 0).only_enforce_if(d)
model.add(n1 != 0).only_enforce_if(d.Not())
model.add(n2 == 1).only_enforce_if(d)

class MySolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._vars = variables
        self._count = 0

    def on_solution_callback(self):
        self._count += 1
        print(f"Solution {self._count}:")
        for name, var in self._vars.items():
            print(f"  {name}: {SHIFTS[self.value(var)]}")
        print()

    @property
    def count(self):
        return self._count

solver = cp_model.CpSolver()
cb = MySolutionPrinter({
    'Nurse 1': n1,
    'Nurse 2': n2,
    'Nurse 3': n3,
    'Nurse 4': n4,
    'Nurse 5': n5,
})
solver.parameters.enumerate_all_solutions = True
solver.solve(model, cb)
print(f"Total solutions: {cb.count}")
