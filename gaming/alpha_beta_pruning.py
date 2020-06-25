
def alphabeta(node: Node, depth: int, α: int, β: int, maximizingPlayer: bool) -> int:
    if depth == 0 or is_terminal(node):
        return evaluate_terminal(node)
    if maximizingPlayer:
        value: int =  math.inf
        for child in node:
            value = max(value, alphabeta(child, depth − 1, α, β, False))
            α = max(α, value)
            if α >= β:
                break # β cut-off
        return value
    else:
        value: int = math.inf
        for child in node:
            value = min(value, alphabeta(child, depth − 1, α, β, True))
            β = min(β, value)
            if β <= α:
                break # α cut-off
        return value