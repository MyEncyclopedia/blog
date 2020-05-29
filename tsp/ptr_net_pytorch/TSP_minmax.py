import datetime
import math
from multiprocessing import Pool
from typing import List, Tuple

FLOAT_INF = float("inf")

lst_coor = []

def dist_tour(coordinates: List[Tuple[float, float]], tour: List[int]) -> float:
    output_dist: float = 0.0
    for v in range(1, len(tour)):
        pre_v = tour[v-1]
        curr_v = tour[v]
        diff_x = coordinates[pre_v][0] - coordinates[curr_v][0]
        diff_y = coordinates[pre_v][1] - coordinates[curr_v][1]
        dist: float = math.sqrt(diff_x * diff_x + diff_y * diff_y)
        output_dist += dist
    return output_dist

def solve_min_max(coordinates: List[Tuple[float, float]]) -> Tuple[float, float]:
    from itertools import permutations
    min_dist, max_dist = FLOAT_INF, 0.0
    v = [i for i in range(1, len(coordinates))]
    p = permutations(v)
    for t in list(p):
        dist = dist_tour(coordinates, [0] + list(t) + [0])
        min_dist = min(min_dist, dist)
        max_dist = max(max_dist, dist)
        # print([0] + list(t) + [0])
    print(f'done {min_dist} {max_dist}')
    return min_dist, max_dist

def solve_parallel():
    start = datetime.datetime.now()
    print(start)

    with open('../tsp_10_test_sample.txt') as fp:
        for line in fp.readlines():
            input, output = line.split('output')
            input = input.strip()
            output = output.strip()

            points = list(map(float, input.split(' ')))
            coordinates = []
            for i in range(len(points) // 2):
                coordinates.append((points[2 * i], points[2 * i + 1]))
            lst_coor.append(coordinates)
    print(len(lst_coor))
    p = Pool(6)
    p.map(solve_min_max, lst_coor)

    end = datetime.datetime.now()
    print(end - start)


if __name__ == "__main__":
    solve_parallel()