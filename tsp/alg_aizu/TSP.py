from typing import List

INT_INF = -1

class Graph:
    v_num: int
    edges: List[List[int]]

    def __init__(self, v_num: int):
        self.v_num = v_num
        self.edges = [[INT_INF for c in range(v_num)] for r in range(v_num)]

    def setDist(self, src: int, dest: int, dist: int):
        self.edges[src][dest] = dist


class TSPSolver:
    g: Graph
    dp: List[List[int]]

    def __init__(self, g: Graph):
        self.g = g
        self.dp = [[None for c in range(g.v_num)] for r in range(1 << g.v_num)]

    def solve(self) -> int:
        return self._recurse(0, 0)

    def _recurse(self, v: int, state: int) -> int:
        """

        :param v:
        :param state:
        :return: -1 means INF
        """
        dp = self.dp
        edges = self.g.edges

        if dp[state][v] is not None:
            return dp[state][v]

        if (state == (1 << self.g.v_num) - 1) and (v == 0):
            dp[state][v] = 0
            return dp[state][v]

        ret: int = INT_INF
        for u in range(self.g.v_num):
            if (state & (1 << u)) == 0:
                s: int = self._recurse(u, state | 1 << u)
                if s != INT_INF and edges[v][u] != INT_INF:
                    if ret == INT_INF:
                        ret = s + edges[v][u]
                    else:
                        ret = min(ret, s + edges[v][u])
        dp[state][v] = ret
        return ret


def main():
    V, E = map(int, input().split())
    g: Graph = Graph(V)
    for _ in range(E):
        src, dest, dist = map(int, input().split())
        g.setDist(src, dest, dist)

    tsp: TSPSolver = TSPSolver(g)
    print(tsp.solve())


if __name__ == "__main__":
    main()
