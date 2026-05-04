from ortools.sat.python import cp_model

model = cp_model.CpModel()

num_colors = 3
COLORS = {
    0: "Red",
    1: "Green",
    2: "Blue"
}
graph = {
    "A" : model.new_int_var(0, num_colors - 1, "A"),
    "B" : model.new_int_var(0, num_colors - 1, "B"),
    "C" : model.new_int_var(0, num_colors - 1, "C"),
    "D" : model.new_int_var(0, num_colors - 1, "D"),
    "E" : model.new_int_var(0, num_colors - 1, "E")
}
edges = [
    ("A", "B"), ("A", "E"), 
    ("B", "C"), ("B", "D"),
    ("C", "D"), ("D", "E")
]

for n1, n2 in edges:
    model.add(graph[n1] != graph[n2])

class ColorSolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(self, graph):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__graph = graph
        self.__count = 0

    def on_solution_callback(self):
        self.__count += 1
        color_names = {0: "Red", 1: "Green", 2: "Blue"}
        print(f"Solution {self.__count}:", end="  ")
        for node, var in self.__graph.items():
            print(f"{node} = {COLORS[self.value(var)]}", end="  ")
        print()

    @property
    def solution_count(self):
        return self.__count

solver = cp_model.CpSolver()
solver.parameters.enumerate_all_solutions = True  

printer = ColorSolutionPrinter(graph)
solver.solve(model, printer)

print(f"Total solutions: {printer.solution_count}")