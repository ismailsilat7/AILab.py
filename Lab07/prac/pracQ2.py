from ortools.sat.python import cp_model
model = cp_model.CpModel()

ROOMS = {
    1: 'R1',
    2: 'R2',
    3: 'R3',
    4: 'R4',
    5: 'R5',
}

c1 = model.new_int_var(1, 5, 'Math')
c2 = model.new_int_var(1, 5, 'Physics')
c3 = model.new_int_var(1, 5, 'CS')
c4 = model.new_int_var(1, 5, 'English')
c5 = model.new_int_var(1, 5, 'Biology')

model.add_all_different([c1, c2, c3, c4, c5])
model.add_allowed_assignments([c1], [[v] for v in [1, 2]])
model.add(c2 - c3 != 1)
model.add(c2 - c3 != -1)
model.add(c4 != 5)
model.add(c5 > c1)

b = model.new_bool_var('cs_in_r1')
model.add(c3 == 1).only_enforce_if(b)
model.add(c3 != 1).only_enforce_if(b.Not())
model.add(c4 == 4).only_enforce_if(b)

d = model.new_bool_var('physics_in_r3')
model.add(c2 == 3).only_enforce_if(d)
model.add(c2 != 3).only_enforce_if(d.Not())
model.add(c5 != 4).only_enforce_if(d)

solver = cp_model.CpSolver()
status = solver.solve(model)
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print("Solution Found")
    print(f"{c1.name}: {ROOMS[solver.value(c1)]}")
    print(f"{c2.name}: {ROOMS[solver.value(c2)]}")
    print(f"{c3.name}: {ROOMS[solver.value(c3)]}")
    print(f"{c4.name}: {ROOMS[solver.value(c4)]}")
    print(f"{c5.name}: {ROOMS[solver.value(c5)]}")
else:
    print("Solution not found")

