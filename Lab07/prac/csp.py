from ortools.sat.python import cp_model
model = cp_model.CpModel()

#                (min, max, var)
x = model.new_int_var(0,100,"x") # domain is {0,1,2}
y = model.new_int_var(2,9,"y") # domain is {0,1}
z = model.new_int_var(3,8,"z") # domain is {3,4,5,6,7,8}
# day = model.new_int_var(0, 4, "day") # Mon - Fri

# constraints types
# Not equal !=
model.add(x != y)
# equal/fixed values ==
model.add(x == 3)
# Linear inequality
model.add(2*x + 7*y + 3*z <= 50)
# all different (global) shortcut for not equal constraint
# instead of 
"""
model.add(x != y)
model.add(y != z)
model.add(x != z)
"""
# write 1 line
model.add_all_different([x,y,z]) # ALL three must be different from each other

solver = cp_model.CpSolver()
status = solver.solve(model)

"""
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    # solution esists, read the values
    print(f"X: {solver.value(x)}, Y: {solver.value(y)}, Z: {solver.value(z)}")
    # To get all solutions you need a callback
else:
    print("No solution found")
"""

# Finding ALL solutions
class MySolutionPrinter(cp_model.CpSolverSolutionCallback):
    # It inherits from cp_model.CpSolverSolutionCallback — mandatory
    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._vars = variables
        self._count = 0
    
    def on_solution_callback(self):
        # on_solution_callback is where your printing goes — the solver calls it automatically
        self._count += 1
        for name, var in self._vars.items():
            print(f"{name}: {self.value(var)}", end= " ")
            # Inside the callback you use self.value(var) not solver.value(var)
        print()
    
    @property
    def count(self):
        return self._count

cb = MySolutionPrinter({"x": x, "y": y, "z": z})
solver.parameters.enumerate_all_solutions = True
solver.solve(model, cb)
print(f"Total solutions: {cb.count}")


