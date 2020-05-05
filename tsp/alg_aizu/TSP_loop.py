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

    def __init__(self, g: Graph):
        self.g = g

    def solve(self) -> int:
        """

        :param v:
        :param state:
        :return: -1 means INF
        """
        N = self.g.v_num
        dp = [[INT_INF for c in range(N)] for r in range(1 << N)]

        dp[(1 << N) - 1][0] = 0

        for state in range((1 << N) - 2, -1, -1):
            for v in range(N):
                for u in range(N):
                    if ((state >> u) & 1) == 0:
                        if dp[state | 1 << u][u] != INT_INF and self.g.edges[v][u] != INT_INF:
                            if dp[state][v] == INT_INF:
                                dp[state][v] = dp[state | 1 << u][u] + self.g.edges[v][u]
                            else:
                                dp[state][v] = min(dp[state][v], dp[state | 1 << u][u] + self.g.edges[v][u])
        return dp[0][0]


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
