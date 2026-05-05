import random
import heapq
N = 8

def random_grid(n=N):
    grid = [[0 for i in range(n)] for i in range(n)]
    positions = []
    num_items = 0
    for _ in range(n):
        i,j = (random.randint(0, n - 1), random.randint(0, n - 1))
        if grid[i][j] == 0 and i != 0 and j != 0:
            num_items += 1
            grid[i][j] = 1
            positions.append((i,j))
    return num_items, grid, positions

num_items, grid, positions = random_grid()

def get_heuristic(pos, positions):
    min_heuristic = float('inf')
    i,j = pos
    for object in positions:
        x,y = object
        heuristic = abs(i - x) + abs(j - y)
        min_heuristic = min(heuristic, min_heuristic)
    return min_heuristic

def valid_pos(pos):
    if (0 <= pos[0] < N and 0 <= pos[1] < N):
        return True
    return False

def get_neighbors(pos):
    neighbors = []
    moves = [(0,1), (0,-1), (1,0), (-1,0)]
    i,j = pos
    for dx,dy in moves:
        x = i + dx
        y = j + dy
        if 0 <= x < N and 0 <= y < N and grid[i][j] != 1:
            neighbors.append((x,y))
    return neighbors


def greeedy_bfs(start=(0,0) ,grid=grid, num_items=num_items, positions=positions):
    frontier = []
    currentPos = start
    heapq.heappush(frontier, (get_heuristic(currentPos, positions), currentPos))
    visited = set()
    while positions:
        heuristic, currentPos = heapq.heappop(frontier)
        i,j = currentPos
        print(f"Visiting ({i},{j})")
        visited.add(currentPos)
        if(currentPos in positions):
            positions.remove(currentPos)
            grid[i][j] = 0
            print(f"Picked item at ({i},{j}), items_remaining: {len(positions)}")
            visited.clear()
            frontier.clear()
            heapq.heappush(frontier, (get_heuristic(currentPos, positions), currentPos))
            if len(positions) == 0:
                print("All items picked! exiting...")
                break
            continue
        neighbors = get_neighbors(currentPos)
        for neighbor in neighbors:
            if neighbor not in visited:
                heapq.heappush(frontier, (get_heuristic(neighbor, positions), neighbor))

print("------Grid------")
for i in range(N):
    for j in range(N):
        print(f"{grid[i][j]} ",end="")
    print()
print(f"Total items: {len(positions)}")
print("Searching from (0,0)")
greeedy_bfs()      

