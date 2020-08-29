import random
import numpy as np

def simulate_bruteforce(n: int) -> bool:
    """
    Simulates one round. Unbounded time complexity.
    :param n: total number of seats
    :return: True if last one has last seat, otherwise False
    """

    seats = [False for _ in range(n)]

    for i in range(n-1):
        if i == 0:  # first one, always random
            seats[random.randint(0, n - 1)] = True
        else:
            if not seats[i]:  # i-th has his seat
                seats[i] = True
            else:
                while True:
                    rnd = random.randint(0, n - 1)
                    if not seats[rnd]:
                        seats[rnd] = True
                        break
    return not seats[n-1]


def simulate_online(n: int) -> bool:
    """
    Simulates one round of complexity O(N).
    :param n: total number of seats
    :return: True if last one has last seat, otherwise False
    """

    seats = [i for i in range(n)]

    def swap(i, j):
        tmp = seats[i]
        seats[i] = seats[j]
        seats[j] = tmp

    # for each person, the seats are [i, n-1]
    for i in range(n-1):
        if i == 0:  # first one, always random
            rnd = random.randint(0, n - 1)
            swap(rnd, 0)
        else:
            if seats[i] == i:  # i-th still has his seat
                pass
            else:
                rnd = random.randint(i, n - 1)  # selects idx from [i, n-1]
                swap(rnd, i)
    return seats[n-1] == n - 1


def plot(y):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    x = np.arange(len(y))

    ax = sns.pointplot(x=x[1:], y=y[1:], join=False)
    ax.set_title("1227")

    plt.show()

if __name__ == "__main__":
    stat = np.zeros(21)
    sample_size = 500
    for n in range(1, 21):
        # total = sum(1 if simulate_online(n) else 0 for _ in range(500))
        total = sum(1 if simulate_bruteforce(n) else 0 for _ in range(sample_size))
        stat[n] = total / sample_size

    print(f'{stat}')
    plot(stat)

