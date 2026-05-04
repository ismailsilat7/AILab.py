from ortools.sat.python import cp_model
model = cp_model.CpModel()

# 5 day week
# 1 outfit to each day broski

DAYS = {
    "d1": 'Monday',
    "d2": 'Tuesday',
    "d3": 'Wednesday',
    "d4": 'Thursday',
    "d5": 'Friday'
}

class MySolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._vars = variables
        self._count = 0
    def on_solution_callback(self):
        self._count += 1
        print(f"Solution {self._count}\n")
        for name, outfit in self._vars.items():
            index = self.value(outfit)
            print(f"\t{DAYS[name]}: {f"Shalwar Qameez {outfits[index][1] + 1}" if index < len(shalwar_qameez) else f"Shirt {outfits[index][0]}, Pant {outfits[index][1]}"}")
        print()
    @property
    def count(self):
        return self._count

shalwar_qameez = [(0, i) for i in range(0, 2)]
shirt_pant = [(i,j) for i in range(1,6) for j in range(1, 4)]
outfits = shalwar_qameez + shirt_pant

d1 = model.new_int_var(0, len(outfits) - 1, 'd1')
d2 = model.new_int_var(0, len(outfits) - 1, 'd2')
d3 = model.new_int_var(0, len(outfits) - 1, 'd3')
d4 = model.new_int_var(0, len(outfits) - 1, 'd4')
d5 = model.new_int_var(0, len(outfits) - 1, 'd5')

model.add_allowed_assignments([d5], [[i] for i in range(len(shalwar_qameez))])
model.add_allowed_assignments([d1], [[i + len(shalwar_qameez)] for i in range(len(shirt_pant))])
model.add_allowed_assignments([d4], [[i + len(shalwar_qameez)] for i in range(len(shirt_pant))])
model.add_all_different([d1, d2, d3, d4, d5])

solver = cp_model.CpSolver()
cb = MySolutionPrinter({
    'd1': d1,
    'd2': d2,
    'd3': d3,
    'd4': d4,
    'd5': d5
})
solver.parameters.enumerate_all_solutions = True
solver.solve(model, cb)
print(f"\nTotal Solutions: {cb.count}")


