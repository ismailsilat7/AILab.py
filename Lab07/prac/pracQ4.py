from ortools.sat.python import cp_model
model = cp_model.CpModel()

TIME_SLOT = {
    1: '9:00 - 10:00 AM',
    2: '10:00 - 11:00 AM',
    3: '11:00 - 12:00 PM',
    4: '2:00 - 3:00 PM',
    5: '3:00 - 4:00 PM',
}

t1 = model.new_int_var(1, 5, "Machine Learning")
t2 = model.new_int_var(1, 5, "Robotics")
t3 = model.new_int_var(1, 5, "Natural Language Processing")
t4 = model.new_int_var(1, 5, "Computer Vision")
t5 = model.new_int_var(1, 5, "Reinforcement Learning")

model.add_all_different([t1, t2, t3, t4, t5])
# t1 & t2 cannot be in consecutive slots
model.add(t1 - t3 != 1)
model.add(t1 - t3 != -1)

model.add(t2 < t4)
model.add_allowed_assignments([t5], [[v] for v in [3, 4, 5]])
model.add(t3 > t1)
model.add(t4 != 5)

# 3 shld be either t1 or t2
b = model.new_bool_var("t1_is_3")
model.add(t1 == 3).only_enforce_if(b)
model.add(t1 != 3).only_enforce_if(b.Not())
d = model.new_bool_var("t2_is_3")
model.add(t2 == 3).only_enforce_if(d)
model.add(t2 != 3).only_enforce_if(d.Not())

model.add_bool_or([b, d])

class MySolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._vars = variables
        self._count = 0
    def on_solution_callback(self):
        self._count += 1
        print(f"Solution {self._count}")
        for name, val in self._vars.items():
            print(f"\t{name}: {TIME_SLOT[self.value(val)]}")
        print()
    @property
    def count(self):
        return self._count

solver = cp_model.CpSolver()
cb = MySolutionPrinter({
    "Machine Learning": t1,
    "Robotics": t2,
    "Natural Language Processing": t3,
    "Computer Vision": t4,
    "Reinforcement Learning": t5,
})
solver.parameters.enumerate_all_solutions = True
solver.solve(model, cb)
print(f"Total solutions: {cb.count}")

