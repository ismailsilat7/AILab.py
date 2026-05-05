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

def alpha_beta(node, depth, alpha, beta, is_maximizing, pruned, visited):
    visited.append(node.name)
    if depth == 0 or not node.children:
        return node.value
    best = -math.inf
    if is_maximizing:
        for i, child in enumerate(node.children):
            val = alpha_beta(child, depth - 1, alpha, beta, False, pruned, visited)
            best = max(val, best)
            alpha = max(best, alpha)
            if alpha >= beta:
                skipped = node.children[i+1:]          # ← everything AFTER current
                pruned.extend([c.name for c in skipped])
                break
        node.value = best
        return best
    else:
        best = math.inf
        for i, child in enumerate(node.children):
            val = alpha_beta(child, depth - 1, alpha, beta, True, pruned, visited)
            best = min(val, best)
            beta= min(best, beta)
            if alpha >= beta:
                skipped = node.children[i+1:]          # ← everything AFTER current
                pruned.extend([c.name for c in skipped])
                break
        node.value = best
        return best

pruned = []
visited = []
optimal_value = alpha_beta(A, 3, -math.inf, math.inf , True, pruned, visited)

print("Internal Node Values")
for node in [D, E, F, G, B, C, A]:
    print(f"  {node.name} = {node.value}")

print(f"Optimal value at root A = {optimal_value}")

best_child = max(A.children, key=lambda n: n.value)
print(f"Best move for Max: A -> {best_child.name}")

print(f"\nPruned branches  : {pruned}")
print(f"Nodes visited AB : {len(visited)}  → {visited}")
print(f"Nodes in Minimax : 15 (all nodes)")
print(f"Nodes saved      : {15 - len(visited)}")
