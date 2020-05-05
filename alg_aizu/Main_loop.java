import java.util.Arrays;
import java.util.Scanner;

public class Main_loop {
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

        public TSP(Graph g) {
            this.g = g;
        }

        public long solve() {
            int N = g.V_NUM;
            long[][] dp = new long[1 << N][N];
            // init dp[][] with MAX
            for (int i = 0; i < dp.length; i++) {
                Arrays.fill(dp[i], Integer.MAX_VALUE);
            }
            dp[(1 << N) - 1][0] = 0;

            for (int state = (1 << N) - 2; state >= 0; state--) {
                for (int v = 0; v < N; v++) {
                    for (int u = 0; u < N; u++) {
                        if (((state >> u) & 1) == 0) {
                            dp[state][v] = Math.min(dp[state][v], dp[state | 1 << u][u] + g.edges[v][u]);
                        }
                    }
                }
            }
            return dp[0][0] == Integer.MAX_VALUE ? -1 : dp[0][0];
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
