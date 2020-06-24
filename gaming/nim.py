import numpy as np
import matplotlib.pyplot as plt

def plot_coor(arr):
    t = list(map(lambda x: 1 if x else None, arr))
    f = list(map(lambda x: 2 if not x else None, arr))
    fig, ax = plt.subplots()
    pt = ax.plot(t, '*', markersize=10)
    pf = ax.plot(f, '.', markersize=10)
    ax.legend(['Win', 'Lose'], loc='upper left')
    plt.yticks([0, 1, 2, 3])
    plt.xticks(range(len(arr)))
    # ax.set_ylim(0, 3)
    plt.show()



def canWin(n):
    if n <= 3:
        return True
    for i in range(1, 4):
        if not canWin(n - i):
            return True
    return False

if __name__ == "__main__":
    aa = [canWin(i) for i in range(0, 20)]
    # plot_matrix(aa)
    plot_coor(aa)