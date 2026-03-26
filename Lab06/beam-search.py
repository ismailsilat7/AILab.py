import heapq

graph = {
    'S': [('A', 3), ('B', 6), ('C', 5)],
    'A': [('D', 9), ('E', 8)],
    'B': [('F', 12), ('G', 14)],
    'C': [('H', 7)],
    'H': [('I', 5), ('J', 6)],
    'I': [('K', 1), ('L', 10), ('M', 2)],
    'D': [],'E': [],
    'F': [],'G': [],
    'J': [],'K': [],
    'L': [],'M': []
}

def beam_search(start, goal, width=5, maxwidth=7):
    states = [ (0, [start]) ]
    level = 1
    while states:
        if level % 3 == 0 and width < maxwidth:
            width += 1
        print(f"Level: {level}, Width: {width}\nBeam Nodes: {states}")
        all_neighbors = []
        for cost, path in states:
            curr_node = path[-1]
            if curr_node == goal:
                return path, cost
            else:
                neighbors = graph[curr_node]
                for neighbor, edge_cost in neighbors:
                    new_cost = cost + edge_cost
                    new_path = path + [neighbor]
                    all_neighbors.append((new_cost, new_path))
        states = heapq.nsmallest(width, all_neighbors, key=lambda x:x[0])
        level += 1
    return None, float('inf')

start = 'S'
goal = 'G'
path, cost = beam_search(start, goal)

if path:
    print(f"Path found with cost {cost}\nPath: {path}")
else:
    print("No Path found")
    