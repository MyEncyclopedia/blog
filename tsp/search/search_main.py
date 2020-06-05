from typing import Tuple, List

initial = [0.35, 0.25, 0.4]

transition = [
    [0.3, 0.6, 0.1],
    [0.4, 0.2, 0.4],
    [0.3, 0.4, 0.4]]

def search_brute_force(initial: List, transition: List, L: int) -> Tuple[float, Tuple]:
    from itertools import combinations_with_replacement
    v = [0, 1, 2]
    path_all = combinations_with_replacement(v, L)

    max_prop = 0.0
    max_route = None
    prob = 0.0
    for path in list(path_all):
        for idx, v in enumerate(path):
            if idx == 0:
                prob = initial[v]  # reset to initial state
            else:
                prev_v = path[idx-1]
                prob *= transition[prev_v][v]
        if prob > max_prop:
            max_prop = max(max_prop, prob)
            max_route = path
    return max_prop, max_route


def search_dp(initial: List, transition: List, L: int) -> float:
    N = len(initial)
    dp = [[0.0 for c in range(N)] for r in range(L)]
    dp[0] = initial[:]

    for l in range(1, L):
        for v in range(N):
            for prev_v in range(N):
                dp[l][v] = max(dp[l][v], dp[l - 1][prev_v] * transition[prev_v][v])

    return max(dp[L-1])


def search_greedy(initial: List, transition: List, L: int) -> Tuple[float, Tuple]:
    import numpy as np
    max_route = []
    max_prop = 0.0
    states = np.array(initial)

    prev_max_v = None
    for l in range(0, L):
        max_v = np.argmax(states)
        max_route.append(max_v)
        if l == 0:
            max_prop = initial[max_v]
        else:
            max_prop = max_prop * transition[prev_max_v][max_v]
        states = max_prop * states
        prev_max_v = max_v

    return max_prop, max_route


def search_beam(initial: List, transition: List, L: int, K: int) -> Tuple[float, Tuple]:
    N = len(initial)
    from queue import PriorityQueue
    current_q = PriorityQueue()
    next_q = PriorityQueue()

    from functools import total_ordering
    @total_ordering
    class PQItem(object):
        def __init__(self, prob, route):
            self.prob = prob
            self.route = route
            self.last_v = int(route[-1])

        def __eq__(self, other):
            return self.prob == other.prob

        def __lt__(self, other):
            return self.prob > other.prob

    for v in range(N):
        next_q.put(PQItem(initial[v], str(v)))

    for l in range(1, L):
        current_q = next_q
        next_q = PriorityQueue()
        k = K
        while not current_q.empty() and k > 0:
            item = current_q.get()
            prob, route, prev_v = item.prob, item.route, item.last_v
            k -= 1
            for v in range(N):
                nextItem = PQItem(prob = prob * transition[prev_v][v], route = route + str(v))
                next_q.put(nextItem)

    max_item = next_q.get()

    return max_item.prob, list(map(lambda x: int(x), max_item.route))


if __name__ == "__main__":
    max_prop, max_route = search_brute_force(initial, transition, 3)
    print(max_prop, max_route)
    max_prop, max_route = search_beam(initial, transition, 3, 2)
    print(max_prop, max_route)
    # print(max_path)
