import random

N = 8 # queens

def calculate_conflicts(state):
    conflicts = 0
    n = N
    for i in range(n):
        for j in range(i + 1, n):
            if (state[i] == state[j] or abs(state[i] - state[j]) == abs(i - j)):
                conflicts += 1
    return conflicts

def get_neighbors(state):
    neighbors = []
    for i in range(N):
        for j in range(N):
            if state[i] != j:
                new_state = list(state)
                new_state[i] = j
                neighbors.append(new_state)
    return neighbors

def hill_climbing(max=20):
    for i in range(max):
        current_state = [int(random.random() * N) for _ in range(N)]
        current_conflicts = calculate_conflicts(current_state)
        if i != 0:
            print(f"Random Restart (Attempt: {i + 1}), state: \n{current_state}, conflicts: {current_conflicts}")
        else:
            print(f"Starting, state: \n{current_state}, conflicts: {current_conflicts}")

        while True:
            # Steepest-Ascent Hill Climbing:
            neighbors = get_neighbors(current_state)
            best_neighbor = neighbors[0]
            best_conflicts = calculate_conflicts(best_neighbor)
            for neighbor in neighbors[1:]:
                neighbor_conflicts = calculate_conflicts(neighbor)
                if neighbor_conflicts < best_conflicts:
                    best_neighbor = neighbor
                    best_conflicts = neighbor_conflicts
            
            if best_conflicts >= current_conflicts:
                break
            current_conflicts = best_conflicts
            current_state = best_neighbor
            print(f"State change, state: \n{current_state}, conflicts: {current_conflicts}")
        
        if current_conflicts != 0:
            print("Solution not found - stuck at local minima. Restarting...")
        else:
            print(f"Solution found; solution: {current_state} with conflicts = {current_conflicts}")
            return (True, i + 1)
    return (False, max) 

solved = hill_climbing()
if solved[0]:
    print(f"Solved in attempts {solved[1]}")
else:
    print(f"Couldn't solve even with max attempts ({solved[1]})")



