import random


def run(N: int) -> bool:
    pos = [i for i in range(N)]

    def swap(i, j):
        tmp = pos[i]
        pos[i] = pos[j]
        pos[j] = tmp

    for i in range(N):
        if i == 0:
            rnd = random.randint(0, N - 1)
            swap(rnd, N - 1)
        else:
            if pos[i] == i:
                swap(i, N - i - 1)
            else:
                # remaining
                rnd = random.randint(0, N - i - 1)
                if i == N - 1:
                    assert rnd == 0
                swap(rnd, N - i - 1)
    return pos[0] == N - 1


if __name__ == "__main__":
    N = 4
    ret = {True: 0, False: 0}
    for _ in range(500):
        result = run(N)
        print(f'{result}\n')
        # print(result)
        ret[result] += 1

    print(f'{ret}')

# ret = {0: 0, 1:0, 2:0, 3:0}
# for _ in range(1000):
# 	ret[random.randint(0, 3)] += 1
# print(ret)
