from ortools.sat.python import cp_model

model = cp_model.CpModel()

outfit_names = {0: "SQ1", 1: "SQ2"}
i = 2
for s in range(1, 6):
    for p in range(1, 4):
        outfit_names[i] = f"S{s}-P{p}"
        i += 1

days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
total_outfits = 17

schedule = {
    day: model.new_int_var(0, total_outfits - 1, day)
    for day in days
}

model.add_all_different(schedule.values())
for day in ["Monday", "Thursday"]:
    model.add(schedule[day] >= 2)
model.add(schedule["Friday"] <= 1)

solver = cp_model.CpSolver()
status = solver.solve(model)

if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
    for day, var in schedule.items():
        print(f"{day}: {outfit_names[solver.value(var)]}")