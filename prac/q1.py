import heapq
import random

N = 4
M = 5
MAX_STEPS = 8

class Environment:
    def __init__(self, grid):
        self.grid = grid

    def getPercept(self, state):
        return state
    
class UtilityBasedAgent:
    def __init__(self):
        self.utility = {0: 0, 1: 1, 2: 2, 'X': -10, 'S': 0}
    
    def calculate_utiltiy(self, state, grid):
        i,j = state
        if grid[i][j] in self.utility:
            return self.utility[grid[i][j]]
        return 0
    
    def get_neighbors(self, state, grid):
        moves = [(0,1), (1,0), (0,-1), (-1,0)]
        neighbors = []
        i,j = state
        for dx,dy in moves:
            x = dx + i
            y = dy + j
            if 0 <= x < N and 0 <= y < M and grid[x][y] != 'X':
                neighbors.append((x,y))
        return neighbors
    
    def dls(self, state, grid, utility):
        frontier = []
        paths = []
        utilities = []
        max_depth = MAX_STEPS
        frontier.append((0, state, [state], grid, 0))
        
        while frontier:
            depth, state, path, old_grid, old_utility = frontier.pop()
            if depth == max_depth:
                paths.append(path)
                utilities.append(old_utility)
                continue
            new_grid = [list(row) for row in old_grid]
            new_grid[state[0]][state[1]] = 0
            neighbors = self.get_neighbors(state, grid)
            for neighbor in neighbors:
                new_utility = old_utility + self.calculate_utiltiy(neighbor, new_grid)
                this_path = path + [neighbor]
                frontier.append((depth + 1, neighbor, this_path, new_grid, new_utility))
        return paths, utilities

    def act(self, grid, state):
        paths, utilities = self.dls(state, grid, self.utility)
        if len(paths) == 0:
            print(f"No path found...")
            return None
        best_utility = utilities[0]
        best_path = paths[0]
        for i in range(1, len(paths)):
            print(f"Path: {paths[i]}\n Utility: {utilities[i]}\n")
            if best_utility < utilities[i]:
                best_utility = utilities[i]
                best_path = paths[i]
        return best_utility, best_path
    
# The 4x5 Grid from the manual
room_grid = [
    ['S', 2, 0, 0, 1],
    [ 0 , 'X', 1, 2, 0],
    [ 0 , 2, 0, 'X', 0],
    [ 0 , 0, 1, 0, 2]
]

# Find the Start position dynamically
start_pos = (0, 0)
for r in range(len(room_grid)):
    for c in range(len(room_grid[0])):
        if room_grid[r][c] == 'S':
            start_pos = (r, c)
            room_grid[r][c] = 0 # Clean the start square

env = Environment(room_grid)
agent = UtilityBasedAgent()

grid_percept = env.getPercept(start_pos)
best_score, best_route = agent.act(room_grid, grid_percept)

print(f"Maximum Cleanliness Score: {best_score}")
print("Optimal Path:", best_route)

