
maze = [
    [0,1,0,0,0,0],
    [0,1,0,1,1,0],
    [0,0,0,0,1,0],
    [0,1,1,0,0,0],
    [0,0,1,0,1,0],
    [1,0,0,0,1,0]
]

start = (0,0)
goal = (5,5)
N = 6

def get_neighbors(pos, maze):
    moves = [(1,0), (-1,0), (0,1), (0, -1)]
    i,j = pos
    neighbors = []
    for dx,dy in moves:
        x = i + dx
        y = j + dy
        if 0 <= x < N and 0 <= y < N and maze[x][y] != 1:
            neighbors.append((x,y))
    return neighbors

def get_path(current_pos, came_from):
    path = []
    while current_pos is not None:
        path.append(current_pos)
        current_pos = came_from[current_pos]
    path.reverse()
    return path

def dfs(start=start, goal=goal, maze=maze):
    nodes = 0
    frontier = [start]
    visitied = set()
    visitied.add(start)
    came_from = {}
    came_from[start] = None
    while frontier:
        current_pos = frontier.pop() # stack LIFO
        i,j = current_pos
        print(f"Visiting ({i},{j})")
        nodes += 1
        if current_pos == goal:
            print("Goal Found via DFS!")
            path = get_path(current_pos, came_from)
            return path, nodes
        neighbors = get_neighbors(current_pos, maze)
        for neighbor in neighbors:
            if neighbor not in visitied:
                came_from[neighbor] = current_pos
                visitied.add(neighbor)
                frontier.append(neighbor)
    print("Goal not found via DFS!")
    return None, nodes

def bfs(start=start, goal=goal, maze=maze):
    nodes = 0
    frontier = [start]
    visitied = set()
    visitied.add(start)
    came_from = {}
    came_from[start] = None
    while frontier:
        current_pos = frontier.pop(0) # stack FIFO
        i,j = current_pos
        print(f"Visiting ({i},{j})")
        nodes += 1
        if current_pos == goal:
            print("Goal Found via BFS!")
            path = get_path(current_pos, came_from)
            return path, nodes
        neighbors = get_neighbors(current_pos, maze)
        for neighbor in neighbors:
            if neighbor not in visitied:
                came_from[neighbor] = current_pos
                visitied.add(neighbor)
                frontier.append(neighbor)
    print("Goal not found via BFS!")
    return None, nodes

print("---Maze---")
for i in range(N):
    for j in range(N):
        print(f"{maze[i][j]} ",end="")
    print()

print("-----DFS-----")
path, nodes_visited = dfs()
if path:
    print("Path: S --> ", end="")
    for i in range(1, len(path) - 1):
        print(f"{path[i]} --> ", end="")
    print("Goal")
print(f"Total Nodes visited: {nodes_visited}")

print("-----BFS-----")
path, nodes_visited = bfs()
if path:
    print("Path: S --> ", end="")
    for i in range(1, len(path) - 1):
        print(f"{path[i]} --> ", end="")
    print("Goal")
print(f"Total Nodes visited: {nodes_visited}")

