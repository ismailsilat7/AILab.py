import math

class Node:
    def __init__(self, name, value=None):
        self.name = name
        self.value = value
        self.children = []

A = Node('A')
B = Node('B')
C = Node('C')
D = Node('D')
E = Node('E')
F = Node('F')
G = Node('G')

leaves = [3,5, 6,9, 1,2, 0,-1]
leaf_nodes = [Node(str(v), v) for v in leaves]

A.children = [B, C]
B.children = [D, E]
C.children = [F, G]
D.children = leaf_nodes[0:2]
E.children = leaf_nodes[2:4]
F.children = leaf_nodes[4:6]
G.children = leaf_nodes[6:8]

def minimax(node, depth, is_maximizing):
    if depth == 0 or not node.children:
        return node.value
    best = -math.inf
    if is_maximizing:
        for child in node.children:
            val = minimax(child, depth - 1, is_maximizing=False)
            best = max(val, best)
        node.value = best
        return best
    else:
        best = math.inf
        for child in node.children:
            val = minimax(child, depth - 1, is_maximizing=True)
            best = min(val, best)
        node.value = best
        return best

optimal_value = minimax(A, depth=3, is_maximizing=True)

print("Internal Node Values")
for node in [D, E, F, G, B, C, A]:
    print(f"  {node.name} = {node.value}")

print(f"Optimal value at root A = {optimal_value}")

best_child = max(A.children, key=lambda n: n.value)
print(f"Best move for Max: A -> {best_child.name}")
