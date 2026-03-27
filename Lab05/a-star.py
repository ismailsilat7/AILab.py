import heapq

grid = [
    ['S', 1, 2, '#', 3],
    [1, '#', 2, '#', '#'],
    [1, 2, 1, 2, '#'],
    ['#', '#', 1, '#', '#'],
    [1, 1, 1, 2, 'T']
]

N = len(grid)

def get_heuristic(pos, goal=(4,4)):
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

def construct_path(came_from, currentPos):
    path = []
    while currentPos:
        path.append(currentPos)
        currentPos = came_from[currentPos]
    path.reverse()
    return path

def get_neighbors(pos):
    moves = [(1,0), (-1,0), (0,1), (0,-1)]
    i,j = pos
    neighbors = []
    for dx,dy in moves:
        x = i + dx
        y = j + dy
        if 0 <= x < N and 0 <= y < N:
            neighbors.append((x,y))
    return neighbors

def a_star(start=(0,0), grid=grid):
    frontier = []
    g_cost = {}
    came_from = {}
    visited = set()

    heapq.heappush(frontier, (get_heuristic(start) + 0, start))
    came_from[start] = None
    g_cost[start] = 0

    while frontier:
        heuristic, currentPos = heapq.heappop(frontier)
        current_cost = g_cost[currentPos]
        if currentPos in visited:
            continue
        i,j = currentPos
        visited.add(currentPos)
        print(f"Visiting ({i},{j}) with cost {g_cost[currentPos]} ")

        if grid[i][j] == 'T':
            print("Goal found!")
            path = construct_path(came_from, currentPos)
            return path, g_cost[currentPos]
        
        neighbors = get_neighbors(currentPos)
        for neighbor in neighbors:
            x,y = neighbor
            if grid[x][y] == '#':
                continue
            if isinstance(grid[x][y], int):
                new_cost = grid[x][y] + current_cost
            else:
                new_cost = 1 + current_cost 
            new_heuristic = get_heuristic(neighbor)
            if neighbor not in g_cost or new_cost < g_cost[neighbor]:
                g_cost[neighbor] = new_cost
                came_from[neighbor] = currentPos
                heapq.heappush(frontier, (new_cost + new_heuristic, neighbor))
    print("Goal not found!")
    return None, float('inf')


print("------Grid------")
for i in range(N):
    for j in range(N):
        print(f"{grid[i][j]} ",end="")
    print()
path, cost = a_star()
if path:
    print("Path: S --> ", end="")
    for i in range(1, len(path) - 1):
        print(f"{path[i]} --> ", end="")
    print("Goal")
    print(f"Total Cost: {cost}")


