import math
from typing import List

FLOAT_INF = float("inf")

class Graph:
    v_num: int
    edges: List[List[float]]

    def __init__(self, v_num: int):
        self.v_num = v_num
        self.edges = [[FLOAT_INF for c in range(v_num)] for r in range(v_num)]

    def setDist(self, src: int, dest: int, dist: float):
        self.edges[src][dest] = dist


class TSPSolver:
    g: Graph
    dp: List[List[float]]
    parent: List[List[int]]
    tour: List[int]
    dist: float

    def __init__(self, g: Graph):
        self.g = g
        self.dp = [[-1.0 for c in range(g.v_num)] for r in range(1 << g.v_num)]
        self.parent = [[-1 for c in range(g.v_num)] for r in range(1 << g.v_num)]

    def solve(self):
        self.dist = self._recurse(0, 0)
        self._form_tour()

    def _recurse(self, v: int, state: int) -> float:
        """

        :param v:
        :param state:
        :return: -1 means INF
        """
        dp = self.dp
        edges = self.g.edges

        if dp[state][v] >= 0.0:
            return dp[state][v]

        if (state == (1 << self.g.v_num) - 1) and (v == 0):
            dp[state][v] = 0.0
            return dp[state][v]

        ret: float = FLOAT_INF
        u_min: int = -1
        for u in range(self.g.v_num):
            if (state & (1 << u)) == 0:
                s: float = self._recurse(u, state | 1 << u)
                if s + edges[v][u] < ret:
                    ret = s + edges[v][u]
                    u_min = u
        dp[state][v] = ret
        self.parent[state][v] = u_min
        return ret

    def _form_tour(self):
        self.tour = [0]
        bit = 0
        v = 0
        for _ in range(self.g.v_num - 1):
            v = self.parent[bit][v]
            self.tour.append(v)
            bit = bit | (1 << v)
        self.tour.append(0)


def main(line: str):
    input, output = line.split('output')
    input = input.strip()
    output = output.strip()

    points = list(map(float, input.split(' ')))
    coordinates = []
    for i in range(len(points) // 2):
        coordinates.append((points[2*i], points[2*i+1]))

    N = len(coordinates)
    g: Graph = Graph(N)
    for v in range(N):
        for u in range(N):
            diff_x = coordinates[v][0] - coordinates[u][0]
            diff_y = coordinates[v][1] - coordinates[u][1]
            dist: float = math.sqrt(diff_x * diff_x + diff_y * diff_y)
            g.setDist(u, v, dist)
            g.setDist(v, u, dist)
    tsp: TSPSolver = TSPSolver(g)
    tsp.solve()

    output_dist: float = 0.0
    output_tour = list(map(lambda x: int(x) - 1, output.split(' ')))
    for v in range(1, len(output_tour)):
        pre_v = output_tour[v-1]
        curr_v = output_tour[v]
        diff_x = coordinates[pre_v][0] - coordinates[curr_v][0]
        diff_y = coordinates[pre_v][1] - coordinates[curr_v][1]
        dist: float = math.sqrt(diff_x * diff_x + diff_y * diff_y)
        output_dist += dist

    assert tsp.tour == output_tour
    assert abs(tsp.dist - output_dist) < 10e-5


if __name__ == "__main__":
    line = '0.678669 0.091371 0.881432 0.856994 0.030809 0.079008 0.810585 0.128031 0.951074 0.864690 0.751941 0.333851 0.711105 0.622832 0.854909 0.376740 0.255645 0.829741 0.024269 0.003400 output 1 10 3 9 2 5 7 8 6 4 1 '

    main(line)
