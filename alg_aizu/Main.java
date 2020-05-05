// http://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=DPL_2_A
import java.util.Arrays;
import java.util.Scanner;

public class Main {
    public static class Graph {
        public final int V_NUM;
        public final int[][] edges;

        public Graph(int V_NUM) {
            this.V_NUM = V_NUM;
            this.edges = new int[V_NUM][V_NUM];
            for (int i = 0; i < V_NUM; i++) {
                Arrays.fill(this.edges[i], Integer.MAX_VALUE);
            }
        }

        public void setDist(int src, int dest, int dist) {
            this.edges[src][dest] = dist;
        }

    }

    public static class TSP {
        public final Graph g;
        long[][] dp;

        public TSP(Graph g) {
            this.g = g;
        }

        public long solve() {
            int N = g.V_NUM;
            dp = new long[1 << N][N];
            for (int i = 0; i < dp.length; i++) {
                Arrays.fill(dp[i], -1);
            }

            long ret = recurse(0, 0);
            return ret == Integer.MAX_VALUE ? -1 : ret;
        }

        private long recurse(int state, int v) {
            int ALL = (1 << g.V_NUM) - 1;
            if (dp[state][v] >= 0) {
                return dp[state][v];
            }
            if (state == ALL && v == 0) {
                dp[state][v] = 0;
                return 0;
            }
            long res = Integer.MAX_VALUE;
            for (int u = 0; u < g.V_NUM; u++) {
                if ((state & (1 << u)) == 0) {
                    long s = recurse(state | 1 << u, u);
                    System.out.println("" + Integer.toBinaryString(state | 1 << u) + "," + u + " = " + s);
                    res = Math.min(res, s + g.edges[v][u]);
                }
            }
            dp[state][v] = res;
            return res;

        }

    }

    public static void main(String[] args) {

        Scanner in = new Scanner(System.in);
        int V = in.nextInt();
        int E = in.nextInt();
        Graph g = new Graph(V);
        while (E > 0) {
            int src = in.nextInt();
            int dest = in.nextInt();
            int dist = in.nextInt();
            g.setDist(src, dest, dist);
            E--;
        }
        System.out.println(new TSP(g).solve());
    }
}