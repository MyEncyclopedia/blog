
def minimax(node: Node, depth: int, maximizingPlayer: bool) -> int:
    if depth == 0 or is_terminal(node):
        return evaluate_terminal(node)
    if maximizingPlayer:
        value:int = −∞
        for child in node:
            value = max(value, minimax(child, depth − 1, False))
        return value
    else: # minimizing player
        value := +∞
        for child in node:
            value = min(value, minimax(child, depth − 1, True))
        return value