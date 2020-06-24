
def draw(idp):
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    x = np.arange(0, len(idp), 1)
    y = np.arange(0, len(idp[0]), 1)
    X, Y = np.meshgrid(x, y)
    Z = np.array(idp)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    plt.show()


class Solution(object):
    def getMoneyAmount(self, n):
        dp = [[0] * (n+1) for _ in range(n+1)]
        idp = [[0] * (n+1) for _ in range(n+1)]
        for gap in range(1, n):
            for lo in range(1, n+1-gap):
                hi = lo + gap
                dp[lo][hi] = 1000000
                for x in range(lo, hi):
                    v_x = x + max(dp[lo][x-1], dp[x+1][hi])
                    if v_x < dp[lo][hi]:
                        dp[lo][hi] = v_x
                        idp[lo][hi] = x
        draw(idp)
        return idp[1][n]



if __name__ == "__main__":
    s = Solution()
    s.getMoneyAmount(20)
    # for i in range(1, 100):
    #     print(s.getMoneyAmount(i))