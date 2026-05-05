import math
class Node:
    def __init__(self, name, value=None):
        self.name = name
        self.value = value
        self.children = []

def minimax(node, depth, isMaximizing):
    if depth == 0 or not node.children:
        return node.value
    
    if isMaximizing:
        best = -math.inf
        for child in node.children:
            value = minimax(child, depth - 1, isMaximizing=False)
            best = max(value, best)
        node.value = best
        return best
    else:
        best = math.inf
        for child in node.children:
            value = minimax(child, depth - 1, isMaximizing=True)
            best = min(value, best)
        node.value = best
        return best

pruned = []
visited = []
def alpha_beta(node, depth, alpha, beta, isMaximizing, pruned, visited):
    visited.append(node)
    if depth == 0 or not node.children:
        return node.value
    if isMaximizing:
        best = -math.inf
        for i, child in enumerate(node.children):
            value = alpha_beta(child, depth - 1, alpha, beta, False)
            best = max(value, best)
            alpha = max(best, alpha)
            if alpha >= beta:
                skipped = node.children[i+1:]
                pruned.extend([c.name for c in skipped])
                break
        node.value = best
        return best
    else:
        best = math.inf
        for child in node.children:
            value = alpha_beta(child, depth - 1, alpha, beta, True)
            best = min(value, best)
            beta = min(best, beta)
            if alpha >= beta:
                skipped = node.children[i+1:]
                pruned.extend([c.name for c in skipped])
                break
        node.value = best
        return best

root = Node('Root')
A = Node('A')
B = Node('B')
C = Node('C')
D = Node('D')

root.children = [A, B, C, D]
A.children = [Node('A1', 7), Node('A2', 3)]
B.children = [Node('B1', 6), Node('B2', 5)]
C.children = [Node('C1', 9), Node('C2', 1)]
D.children = [Node('D1', 5), Node('D2', 11)]

optimal = minimax(root, 2, True)
for card in [A, B, C, D]:
    print(f"Card {card.name}: oponent picks {card.value}")

best_card = max(root.children, key=lambda n: n.value)
print(f"\nAI picks: Card {best_card.name}")
print(f"Guaranteed payoff: {optimal}")

