import org.omg.CORBA.INTERNAL;

import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.*;

public class Lc19 {
    public int maxArea2(int h, int w, int[] horizontalCuts, int[] verticalCuts) {
        Arrays.sort(horizontalCuts);
        Arrays.sort(verticalCuts);
        int lenx = horizontalCuts.length;
        int leny = verticalCuts.length;
        long maxX = horizontalCuts[0];
        long maxY = verticalCuts[0];
        for (int i = 1; i < lenx; i++) {
            maxX = Math.max(horizontalCuts[i] - horizontalCuts[i - 1], maxX);
        }
        maxX = Math.max(maxX, h - horizontalCuts[lenx - 1]);
        for (int i = 1; i < leny; i++) {
            maxY = Math.max(verticalCuts[i] - verticalCuts[i - 1], maxY);
        }
        maxY = Math.max(maxY, w - verticalCuts[leny - 1]);
        return (int) ((maxX * maxY) % (1000000007));
    }
//    int[][] memo; // 记忆数组
//
//    public int minDifficulty(int[] jobDifficulty, int d) {
//        // 如果d大于数组元素个数，无法分组，返回-1
//        if (d > jobDifficulty.length) return -1;
//        // 初始化记忆数组
//        memo = new int[d + 1][jobDifficulty.length];
//        // 递归求解
//        return help(jobDifficulty, d, 0);
//    }
//
//    private int help(int[] jobDifficulty, int left, int start) {
//        if(memo[left][start]>0)return memo[left][start];
//        int maxD = 0,res=0;
//        for (int i = start; i <=jobDifficulty.length-left ; i++) {
//            maxD =Math.max(maxD,jobDifficulty[i]);
//            res = maxD;
//            if(left>1){
//                res+=help(jobDifficulty,left-1,i+1);
//            }
//        }
//        memo[left][start]=res;
//        return memo[left][start];
//    }

    public String reverseVowels(String s) {
        HashSet<Character> vowel = new HashSet<>();
        vowel.add('a');
        vowel.add('e');
        vowel.add('i');
        vowel.add('o');
        vowel.add('u');
        vowel.add('A');
        vowel.add('E');
        vowel.add('I');
        vowel.add('O');
        vowel.add('U');
        List<Character> v = new LinkedList<>();
        List<Integer> vId = new ArrayList<>();
        for (int i = 0; i < s.length(); i++) {
            if (vowel.contains(s.charAt(i))) {
                v.add(s.charAt(i));
                vId.add(i);
            }
        }
        StringBuffer sb = new StringBuffer();
        Collections.reverse(v);
        int id = 0;
        for (int i = 0; i < s.length(); i++) {
            if (id < vId.size() && i == vId.get(id)) {
                sb.append(v.get(id++));
            } else {
                sb.append(s.charAt(i));
            }
        }
        return sb.toString();

    }

//    public int minDifficulty(int[] jobDifficulty, int d) {
//        int n = jobDifficulty.length;
//        if (d > n) return -1;
//        int[][] F = new int[d + 1][n + 1];
//        for (int i = 1; i <= n; i++) F[1][i] = Math.max(F[1][i - 1], jobDifficulty[i - 1]);
//        for (int i = 2; i <= d; i++) {
//            for (int j = i; j <= n; j++) {
//                F[i][j] = Integer.MAX_VALUE;
//                int currMax = 0;
//                for (int k = j; k >= i; k--) {
//                    currMax = Math.max(currMax, jobDifficulty[k - 1]);
//                    F[i][j] = Math.min(F[i][j], F[i - 1][k - 1] + currMax);
//                }
////                for (int k = i; k <= j; k++) {
////                    currMax = Math.max(currMax, jobDifficulty[k-1]);
////                    F[i][j] = Math.min(F[i][j], F[i-1][k-1] + currMax);
////                }
//            }
//        }
//        return F[d][n];
//    }

    public int minDifficulty(int[] jobDifficulty, int d) {
        int n = jobDifficulty.length;
        if (d > n) return -1;
        int[][] dp = new int[n + 1][d + 1];
        int md = 0;
        for (int i = 1; i <= n; i++) {
            md = Math.max(md, jobDifficulty[i - 1]);
            dp[i][1] = md;
        }
        //dp i,j = ith day, 1-indexed, jth job 1-indexed
        for (int i = 1; i <= n; i++) {
            for (int j = 2; j <= d; j++) {
                md = 0;
                dp[i][j] = Integer.MAX_VALUE;
                for (int k = i; k >= j; k--) {
                    md = Math.max(md, jobDifficulty[k - 1]);
                    dp[i][j] = Math.min(dp[i][j], md + dp[k - 1][j - 1]);
                }
            }
        }
        return dp[n][d];
    }

    public String breakPalindrome(String palindrome) {
        if (palindrome.length() == 1) return "";
        StringBuilder sb = new StringBuilder();
        char[] arr = palindrome.toCharArray();
        boolean changed = false;
        for (int i = 0; i < arr.length - 1; i++) {
            if ((arr.length & 1) == 1 && i == arr.length / 2) {
                sb.append(arr[i]);
                continue;
            }
            if (arr[i] > 'a' && !changed) {
                changed = true;
                sb.append('a');
            } else {
                sb.append(arr[i]);
            }
        }
        if (!changed) {
            sb.append((char) (arr[arr.length - 1] + 1));
        } else {
            sb.append(arr[arr.length - 1]);
        }
        return sb.toString();

    }

    //    1218
    public int longestSubsequence(int[] arr, int difference) {
        Map<Integer, Integer> pre = new HashMap<>();
        Map<Integer, Integer> cur = new HashMap<>();
        int preN = arr[0];
        for (int i = 1; i < arr.length; i++) {
            int dif = arr[i] - preN;
            cur = new HashMap<>();
            boolean find = false;
            for (int each : pre.keySet()) {
                if (each == dif) {
                    cur.put(each, pre.get(each) + 1);
                    find = true;
                } else if (dif > 0 || each + dif > 0) {
                    cur.put(each + dif, pre.get(each));
                } else {
                    cur.put(each + dif, cur.getOrDefault(each + dif, 2));
                }
            }
            if (!find) {
                cur.put(dif, cur.getOrDefault(dif, 2));
            }
            pre = cur;
            preN = arr[i];
        }
        return cur.getOrDefault(difference, 1);
    }

    public int longestSubsequence2(int[] arr, int difference) {
        int len = arr.length;
        int[] dp = new int[len];
        Map<Integer, List<Integer>> map = new HashMap<>();
//        for (int i = 0; i < len; i++) {
//            map.computeIfAbsent(arr[i],k -> new ArrayList<>()).add(i);
//        }
        Arrays.fill(dp, 1);
        for (int i = 0; i < len; i++) {
            if (map.containsKey(arr[i] - difference)) {
                List<Integer> list = map.get(arr[i] - difference);
                for (int j = 0; j < list.size(); j++) {
                    int cur = list.get(j);
                    dp[i] = Math.max(dp[i], dp[cur] + 1);
                }
            }
            map.computeIfAbsent(arr[i], k -> new ArrayList<>()).add(i);
        }
        return Arrays.stream(dp).max().getAsInt();
    }

    public int lenLongestFibSubseq(int[] arr) {
        int len = arr.length, max = 0;
        int[][] dp = new int[len][len];
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < len; i++) {
            map.put(arr[i], i);
        }
        for (int i = 0; i < len; i++) {
            for (int j = i + 1; j < len; j++) {
                int k = arr[j] - arr[i];
                if (k >= arr[i]) break;
                if (map.containsKey(k)) {
                    int pos = map.get(k);
                    dp[i][j] = dp[pos][i] + 1;
                    max = Math.max(max, dp[i][j] + 2);
                }
            }
        }
        return max;
    }

    public int lenLongestFibSubseq2(int[] arr) {
        int len = arr.length, max = 0;
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < len; i++) {
            map.put(arr[i], i);
        }
        Map<Integer, Integer> dp = new HashMap<>();
        // k j i
        for (int i = 0; i < len; i++) {
            for (int j = 0; j < i; j++) {
                int k = arr[i] - arr[j];
                if (map.containsKey(k) && map.get(k) < j) {
                    int tmp = dp.getOrDefault(map.get(k) * len + j, 0);
                    dp.put(j * len + i, tmp + 1);
                    max = Math.max(max, tmp + 3);
                }
            }
        }
        return max;
    }

    public int lenLongestFibSubseq3(int[] A) {
        int N = A.length;
        Map<Integer, Integer> index = new HashMap();
        for (int i = 0; i < N; ++i)
            index.put(A[i], i);

        Map<Integer, Integer> longest = new HashMap();
        int ans = 0;
        //   i j k
        for (int k = 0; k < N; ++k)
            for (int j = 0; j < k; ++j) {
                int i = index.getOrDefault(A[k] - A[j], -1);
                if (i >= 0 && i < j) {
                    // Encoding tuple (i, j) as integer (i * N + j)
                    int cand = longest.getOrDefault(i * N + j, 2) + 1;
                    longest.put(j * N + k, cand);
                    ans = Math.max(ans, cand);
                }
            }

        return ans >= 3 ? ans : 0;
    }

    public int numRescueBoats(int[] people, int limit) {
        int sum = 0, left = 0, len = people.length, right = people.length - 1;
        while (left <= right) {
            int tmp = people[left] + people[right];
            if (tmp <= limit) {
                left++;
                right--;
            } else {
                right--;
            }
            sum++;
        }
        return sum;
    }

    // 787
    public List<List<Integer>> allPathsSourceTarget(int[][] graph) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> tmp = new ArrayList<>();
        dfs(graph, 0, tmp, res);
        return res;
    }

    private void dfs(int[][] graph, int start, List<Integer> tmp, List<List<Integer>> res) {
        tmp.add(start);
        if (start == graph.length - 1) {
            res.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = 0; i < graph[start].length; i++) {
            dfs(graph, graph[start][i], tmp, res);
        }
        tmp.remove(tmp.size() - 1);
    }

    int min = Integer.MAX_VALUE;
    int INF = 0x3f3f3f3f;

    public int findCheapestPrice(int n, int[][] flights, int src, int dst, int k) {
        Map<Integer, List<int[]>> map = new HashMap<>();
        for (int[] each : flights) {
            if (!map.containsKey(each[0])) {
                map.putIfAbsent(each[0], new ArrayList<>());
            }
            map.get(each[0]).add(each);
        }
        int[][] vis = new int[n][k + 2];

        min = dfs3(map, src, dst, k + 1, vis);
        return min == INF ? -1 : min;
    }

    private int dfs3(Map<Integer, List<int[]>> map, int src, int dst, int k, int[][] vis) {
        if (k < 0) return INF;
        if (src == dst) return 0;
        if (vis[src][k] != 0) return vis[src][k];

        List<int[]> tar = map.get(src);
        if (tar == null) return INF;
        int min = INF;
        for (int i = 0; i < tar.size(); i++) {
            int[] tmp = map.get(src).get(i);
            min = Math.min(min, dfs3(map, tmp[1], dst, k - 1, vis) + tmp[2]);
        }
        vis[src][k] = min;
        return min;
    }

//    public int findCheapestPrice2(int n, int[][] flights, int src, int dst, int k) {
//        Map<Integer, List<int[]>> map = new HashMap<>();
//        for (int[] each : flights) {
//            if (!map.containsKey(each[0])) {
//                map.putIfAbsent(each[0], new ArrayList<>());
//            }
//            map.get(each[0]).add(each);
//        }
////        int[][] vis = new int[n][k + 2];
//
//        dfs2(map, src, dst, k + 1, vis);
//        return min == INF ? -1 : min;
//    }

    private void dfs2(Map<Integer, List<int[]>> map, int src, int dst, int k, int sum, boolean[] vis) {
        if (k < 0 || vis[src]) return;
        if (src == dst) {
            min = Math.min(min, sum);
            return;
        }
        vis[src] = true;
        System.out.println(src + ":" + k);
        List<int[]> tar = map.get(src);
        if (tar == null) return;
        for (int i = 0; i < tar.size(); i++) {
            int[] tmp = map.get(src).get(i);
            dfs2(map, tmp[1], dst, k - 1, sum + tmp[2], vis);

        }
        vis[src] = false;
    }

    public int findCheapestPrice2(int n, int[][] flights, int src, int dst, int k) {
        int INF = 0x3f3f3f3f;
        int[][] dp = new int[n][k + 2];
        for (int i = 0; i < n; i++) {
            Arrays.fill(dp[i], INF);
        }
        dp[dst][0] = 0;
        for (int i = 1; i <= k + 1; i++) {
            for (int[] each : flights) {
                dp[each[0]][i] = Math.min(dp[each[0]][i], dp[each[1]][i - 1] + each[2]);
            }
        }
        return Arrays.stream(dp[src]).min().getAsInt();
    }

    //1262
    public int maxSumDivThree(int[] nums) {
        int len = nums.length;
        int[][] dp = new int[2][3];
        // 0 for old, 1 for new
        for (int each : nums) {
            for (int i = 0; i < 3; i++) {
                int sum = dp[0][i] + each;
                dp[1][sum % 3] = Math.max(dp[1][sum % 3], sum);
            }
            dp[0] = dp[1].clone();
        }
        return dp[0][0];
    }

//    public double knightProbability(int n, int k, int row, int column) {
//        int[][] olddp = new int[n][n];
//        int[][] newdp = new int[n][n];
//        int[][] directions = {{2, -1}, {2, 1}, {1, 2}, {1, -2}, {-2, 1}, {-2, -1}, {-1, 2}, {-1, -2}};
//        Queue<int[]>queue =new LinkedList<>();
//        queue.offer(new int[]{row,column});
//        int count = 0;
////        Set<Integer> set =new HashSet<>();
//        while(count<k){
//            int size = queue.size();
//            count++;
////            set = new HashSet<>();
//            for (int i = 0; i < size; i++) {
//                int[] cur = queue.poll();
//                int flag = cur[0]*n+cur[1];
////                if(set.contains(flag)){
////                    continue;
////                }
////                set.add(flag);
//                for (int j = 0; j < directions.length; j++) {
//                    int x_n = cur[0]+directions[j][0];
//                    int y_n = cur[1]+directions[j][1];
//                    if(x_n>=0&&x_n<n&&y_n>=0&&y_n<n){
//                        queue.offer(new int[]{x_n,y_n});
//                    }
//                }
//            }
//        }
//        return queue.size()/(Math.pow(8,k));
//    }

    public double knightProbability(int n, int k, int row, int column) {
        double[][] olddp = new double[n][n];
        double[][] newdp = new double[n][n];
        int[][] directions = {{2, -1}, {2, 1}, {1, 2}, {1, -2}, {-2, 1}, {-2, -1}, {-1, 2}, {-1, -2}};
        int count = 0;
        olddp[row][column] = 1;
        while (count < k) {
            count++;
            for (int i = 0; i < n; i++) {
                Arrays.fill(newdp[i], 0);
            }
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (olddp[i][j] != 0) {
                        for (int l = 0; l < directions.length; l++) {
                            int x_n = i + directions[l][0];
                            int y_n = j + directions[l][1];
                            if (x_n >= 0 && x_n < n && y_n >= 0 && y_n < n) {
                                newdp[x_n][y_n] += olddp[i][j];
                            }
                        }
                    }
                }
            }

            for (int i = 0; i < n; i++) {
                olddp[i] = newdp[i].clone();
            }
        }
        double sum = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                sum += olddp[i][j];
            }
        }
        System.out.println(sum);
//            return sum ;
        return sum / (Math.pow(8, k));
    }

    public int knightDialer(int N) {
        int MOD = 1_000_000_007;
        int[][] moves = new int[][]{
                {4, 6}, {6, 8}, {7, 9}, {4, 8}, {3, 9, 0},
                {}, {1, 7, 0}, {2, 6}, {1, 3}, {2, 4}};
        int[][] dp = new int[2][10];
        Arrays.fill(dp[0], 1);
        for (int i = 1; i < N; i++) {

            for (int j = 0; j < 10; j++) {
                for (int k = 0; k < moves[j].length; k++) {
                    dp[1][j] = (dp[1][j] + dp[0][moves[j][k]]) % MOD;
                }
            }
            dp[0] = dp[1].clone();
            Arrays.fill(dp[1], 0);
        }
        int sum = 0;
        for (int i = 0; i < 10; i++) {
            sum = (dp[0][i] + sum) % MOD;
        }
        return sum;
    }

    public int[] numsSameConsecDiff(int n, int k) {
        List<int[]> res = new ArrayList<>();
        int[] path = new int[n];
        dfs4(path, 0, res, k);
        int[] ans = new int[res.size()];
        int id = 0;
        for (int i = 0; i < res.size(); i++) {
            ans[id++] = convert(res.get(i));
        }
        return ans;
    }

    private int convert(int[] ints) {
        int res = 0;
        for (int i = 0; i < ints.length; i++) {
            res = res * 10 + ints[i];
        }
        return res;
    }

    private void dfs4(int[] path, int pos, List<int[]> res, int k) {
        if (pos == path.length) {
            res.add(path.clone());
            return;
        }
        for (int i = 0; i < 10; i++) {
            if (pos == 0 || pos > 0 && Math.abs(i - path[pos - 1]) == k) {
                if (pos == 0 && i == 0) continue;
                path[pos] = i;
                dfs4(path, pos + 1, res, k);
                path[pos] = 0;
            }
        }
    }

    //    523
    public boolean checkSubarraySum(int[] nums, int k) {
        int n = nums.length;
        int[] sum = new int[n + 1];
        int s = 0;
        for (int i = 0; i < n; i++) {
            s += nums[i];
            sum[i + 1] = s;
        }
        Set<Integer> set = new HashSet<>();
        for (int i = 2; i < n + 1; i++) {
            set.add(sum[i - 2] % k);
            if (set.contains(sum[i] % k)) {
                return true;
            }
        }
        return false;
    }

    //    576
    int m1 = 0, n1 = 0, max1 = 0, ans = 0;
    int[][][] cache;
    int[][] dirs = new int[][]{{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    int MOD = (int) 1e9 + 7;

    public int findPaths(int _m, int _n, int _max, int r, int c) {
        m1 = _m;
        n1 = _n;
        max1 = _max;
        cache = new int[m1][n1][max1 + 1];
        for (int i = 0; i < m1; i++) {
            for (int j = 0; j < n1; j++) {
                for (int k = 0; k < max1 + 1; k++) {
                    cache[i][j][k] = -1;
                }
            }
        }
        ans = dfs5(r, c, max1);
        return ans;
    }

    private int dfs5(int r, int c, int max1) {
        if (r < 0 || r >= m1 || c < 0 || c >= n1) return 1;
        if (max1 == 0) return 0;
        if (r - max1 >= 0 && r + max1 < m1 && c - max1 >= 0 && c + max1 < n1) return 0;
        if (cache[r][c][max1] != -1) return cache[r][c][max1];
        int sum = 0;
        for (int[] each : dirs) {
            int x = r + each[0], y = c + each[1];
            sum += dfs5(x, y, max1 - 1);
            sum = sum % MOD;
        }
        cache[r][c][max1] = sum;
        return sum;
    }

    //    dp
    public int findPaths2(int _m, int _n, int _max, int r, int c) {
        int MOD = (int) 1e9 + 7;
        int[][] dirs = new int[][]{{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        int[][][] dp = new int[_m][_n][_max + 1];
        for (int i = 0; i < _m; i++) {
            for (int j = 0; j < _n; j++) {
                for (int k = 1; k < _max + 1; k++) {
                    for (int[] each : dirs) {
                        int x = i + each[0], y = j + each[1];
                        if (x < 0 || x >= _m || y < 0 || y >= _n) {
                            dp[i][j][k]++;
                        }
                    }
                }
            }
        }
        for (int i = 1; i <= _max; i++) {
            for (int j = 0; j < _m; j++) {
                for (int k = 0; k < _n; k++) {
                    for (int[] each : dirs) {
                        int x = j + each[0], y = k + each[1];
                        if (x >= 0 && x < _m && y >= 0 && y < _n) {
                            dp[j][k][i] = (dp[j][k][i] + dp[x][y][i - 1]) % MOD;
                        }
                    }
                }
            }
        }
        return dp[r][c][_max];
    }

    public int findPaths3(int m, int n, int maxMove, int startRow, int startColumn) {
        int MOD = 1000000007;
        int[][][] dp = new int[m][n][maxMove + 1];
        int[][] derections = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        // dp[i][j][step] = dp[i-1][j][step-1] + dp[i+1][j][step-1] + dp[i][j-1][step-1] + dp[i][j+1][step]
        // dp矩阵初始化
        for (int step = 1; step <= maxMove; step++) {
            for (int r = 0; r < m; r++) {
                for (int c = 0; c < n; c++) {
                    if (r == 0) dp[r][c][step]++;
                    if (c == 0) dp[r][c][step]++;
                    if (r == m - 1) dp[r][c][step]++;
                    if (c == n - 1) dp[r][c][step]++;
                }
            }
        }
        // 更新矩阵
        for (int step = 1; step <= maxMove; step++) {
            for (int r = 0; r < m; r++) {
                for (int c = 0; c < n; c++) {
                    for (int[] d : derections) {
                        int nr = r + d[0], nc = c + d[1];
                        if (nr >= 0 && nr < m && nc >= 0 && nc < n) {
                            dp[r][c][step] += dp[nr][nc][step - 1];
                            dp[r][c][step] %= MOD;
                        }
                    }
                }
            }
        }
        return dp[startRow][startColumn][maxMove] % MOD;
    }

    //    1220
//    int ans2 = 0;
//    public int countVowelPermutation(int n) {
//        Map<Character,List<Character>> map = new HashMap<>();
//        map.computeIfAbsent('a',k->new ArrayList<>()).add('e');
//        map.computeIfAbsent('e',k->new ArrayList<>()).addAll(Arrays.asList('a','i'));
//        map.computeIfAbsent('i',k->new ArrayList<>()).addAll(Arrays.asList('a','e','o','u'));
//        map.computeIfAbsent('o',k->new ArrayList<>()).addAll(Arrays.asList('i','u'));
//        map.computeIfAbsent('u',k->new ArrayList<>()).addAll(Arrays.asList('a'));
////        dfs7(map,0,n,'z');
//        int res = dfs8(map,0,n,'z',0);
//        return res;
//    }
//
//    private void dfs7(Map<Character, List<Character>> map, int start, int n, char last) {
//        int MOD =(int)1e9+7;
//        if(start==n){
//            ans2 = (ans2 +1)%MOD;
//            return;
//        }
//        if(last=='z'){
//            for(char each:map.keySet()){
//                dfs7(map, start+1, n, each);
//            }
//        }else{
//            List<Character> list = map.get(last);
//            for (int i = 0; i < list.size(); i++) {
//                dfs7(map, start+1, n, list.get(i));
//            }
//        }
//    }
    public int countVowelPermutation(int n) {
        Map<Integer, List<Integer>> map = new HashMap<>();
//        map.computeIfAbsent('a', k -> new ArrayList<>()).addAll(Arrays.asList('e', 'i','u'));
//        map.computeIfAbsent('e', k -> new ArrayList<>()).addAll(Arrays.asList('a', 'i'));
//        map.computeIfAbsent('i', k -> new ArrayList<>()).addAll(Arrays.asList('e', 'o'));
//        map.computeIfAbsent('o', k -> new ArrayList<>()).addAll(Arrays.asList('i'));
//        map.computeIfAbsent('u', k -> new ArrayList<>()).addAll(Arrays.asList('i','o'));
        map.computeIfAbsent(0, k -> new ArrayList<>()).addAll(Arrays.asList(1, 2, 4));
        map.computeIfAbsent(1, k -> new ArrayList<>()).addAll(Arrays.asList(0, 2));
        map.computeIfAbsent(2, k -> new ArrayList<>()).addAll(Arrays.asList(1, 3));
        map.computeIfAbsent(3, k -> new ArrayList<>()).addAll(Arrays.asList(2));
        map.computeIfAbsent(4, k -> new ArrayList<>()).addAll(Arrays.asList(2, 3));
        int[][] dp = new int[n + 1][5];
        int MOD = (int) 1e9 + 7;
        Arrays.fill(dp[1], 1);
        for (int i = 2; i <= n; i++) {
            for (int j = 0; j < 5; j++) {
                List<Integer> list = map.get(j);
                for (int k = 0; k < list.size(); k++) {
                    int cur = list.get(k);
                    dp[i][j] = (dp[i][j] + dp[i - 1][cur]) % MOD;
                }
            }
        }
        int sum = 0;
        for (int i = 0; i < 5; i++) {
            sum = (sum + dp[n][i]) % MOD;
        }
        return sum;
    }

    //    public int numSubarrayProductLessThanK(int[] nums, int k) {
//        int res =0,left=0,right =0,prod = 1,len =nums.length;
//        while(right<len){
//            prod*=nums[right];
//            while(prod>=k){
//                prod/=nums[left];
//                left++;
//            }
//            res++;
//            right++;
//        }
//        while(right==len&&left<len){
//            left++;
//            res++;
//        }
//        return res;
//    }
    public int numSubarrayProductLessThanK(int[] nums, int k) {
        int res = 0, left = 0, right = 0, prod = 1, len = nums.length;
        while (right < len) {
            prod *= nums[right];
            while (prod >= k) {
                prod /= nums[left];
                left++;
            }
            res += (right - left + 1);
            right++;
        }
        return res;
    }

    public boolean isLongPressedName(String name, String typed) {
        int left = 0, right = 0;
        char pre = 'a';
        if (typed.length() < name.length()) return false;
        while (left < name.length()) {
            if (right == typed.length()) return false;
            if (name.charAt(left) == typed.charAt(right)) {
                pre = name.charAt(left);
                left++;
                right++;
            } else {
                if (typed.charAt(right) == pre) {
                    while (right < typed.length() && typed.charAt(right) == pre) {
                        right++;
                    }
                } else {
                    return false;
                }
            }
        }
        if (right == typed.length()) return true;
        if (typed.charAt(right) == pre) {
            while (right < typed.length()) {
                if (typed.charAt(right) == pre) {
                    right++;
                } else {
                    return false;
                }
            }
        } else {
            return false;
        }
        return true;
    }

    //    995
//    public int minKBitFlips(int[] A, int K) {
//        int left =0,len = A.length,sum=0;
//        for (int i = 0; i < len; i++) {
//            if(A[i]==0){
//                if(!flip(A,i,K)){
//                    return -1;
//                }else{
//                    sum++;
//                }
//            }
//        }
//        return sum;
//    }
//
//    private boolean flip(int[] a, int start, int k) {
//        if(start+k>a.length){
//            return false;
//        }
//        for (int i = start; i <start+k ; i++) {
//            if(a[i]==1){
//                a[i]=0;
//            }else{
//                a[i]=1;
//            }
//        }
//        return true;
//    }
    public int minKBitFlips(int[] A, int K) {
        int left = 0, sum = 0;
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < A.length; i++) {
            if (!queue.isEmpty() && queue.peek() + K <= i) {
                queue.poll();
            }

            if (A[i] == queue.size() % 2) {
                if (i + K > A.length) {
                    return -1;
                }
                queue.offer(i);
                sum++;
            }
        }
        return sum;
    }

    public int minKnightMoves(int x, int y) {
        Set<String> set = new HashSet<>();
        Queue<int[]> queue = new LinkedList<>();
        int[][] directions = {{2, -1}, {2, 1}, {1, 2}, {1, -2}, {-2, 1}, {-2, -1}, {-1, 2}, {-1, -2}};
        queue.offer(new int[]{0, 0});
        int sum = 0;
        while (!queue.isEmpty()) {
            int size = queue.size();
            sum++;
            for (int i = 0; i < size; i++) {
                int[] cur = queue.poll();
                if (cur[0] == x && cur[1] == y) {
                    return sum - 1;
                }
                String curWord = cur[0] + ":" + cur[1];
                if (set.contains(curWord)) {
                    continue;
                } else {
                    set.add(curWord);
                }
                if (x >= 0 && cur[0] + 3 < 0) {
                    continue;
                }
                if (x < 0 && cur[0] - 3 > 0)
                    continue;
                if (y >= 0 && cur[1] + 3 < 0)
                    continue;
                if (y < 0 && cur[1] - 3 > 0)
                    continue;
                for (int[] each : directions) {
                    queue.offer(new int[]{cur[0] + each[0], cur[1] + each[1]});
                }
            }
        }
        return -1;
    }

    public int minimumDifference(int[] nums, int k) {
        Arrays.sort(nums);
        int sum = 0, min = Integer.MAX_VALUE, kmin = 0, kmax = 0;
        for (int i = 0; i < nums.length; i++) {
            kmin = nums[i];
            kmax = nums[i];
            if (i + k > nums.length) break;
            for (int j = i; j < i + k; j++) {
                kmin = Math.min(kmin, nums[j]);
                kmax = Math.max(kmax, nums[j]);
            }
            min = Math.min(min, kmax - kmin);
        }
        return min;

    }

    public String kthLargestNumber(String[] nums, int k) {
        Arrays.sort(nums, (a, b) -> {
            if (a.length() != b.length()) return a.length() - b.length();
            else {
                return a.compareTo(b);
            }
        });
        return nums[nums.length - k];
    }

    public String kthLargestNumber2(String[] nums, int k) {
        quickSort(nums, 0, nums.length - 1);
        return nums[nums.length-k];
    }

    private void quickSort(String[] nums, int left, int right) {
        if (left >= right) return;
        int i = left, j = right;
        String pivot = nums[left];
        while (i < j) {
            while (i < j && compare(nums[j], pivot)) j--;
            while (i < j && compare(pivot, nums[i])) i++;
            swap(nums, i, j);
        }
        swap(nums, left, i);
        quickSort(nums, left, i - 1);
        quickSort(nums, i + 1, right);
    }

    private void swap(String[] nums, int left, int right) {
        String a = nums[left];
        nums[left] = nums[right];
        nums[right] = a;
    }

    //num>=pivot
    private boolean compare(String num, String pivot) {
        if (num.equals(pivot) ) return true;
        if(num.length()!=pivot.length())return num.length()-pivot.length()>0;
        if (num.compareTo(pivot) > 0) return true;
        else return false;
    }

    public static void main(String[] args) {
        Lc19 lc19 = new Lc19();
        int[] s4 = {2};
        int[] s5 = {2};

//        int r4 = lc19.maxArea2(1000000000, 1000000000, s4, s5);
        TrieS2 trieS2 = new TrieS2();
        String[] s6 = {"a", "banana", "app", "appl", "ap", "apply", "apple"};
//        trieS2.longestWord(s6);
        TrieS3 trieS3 = new TrieS3();
        String[] s7 = {"mobile", "mouse", "moneypot", "monitor", "mousepad"};
//        trieS3.suggestedProducts(s7, "mouse");

        TrieS5 trieS5 = new TrieS5();
//        trieS5.lexicalOrder(13);

        int[] s1 = {6, 5, 4, 3, 2, 1};
//        lc19.minDifficulty(s1, 2);

        int[] s2 = {1, 2, 3, 4};
        int[] s3 = {1, 5, 7, 8, 5, 3, 4, 2, 1};
//        int r2 = lc19.longestSubsequence(s3, 1);
//        System.out.println(r2);

        int[] s8 = {1, 2, 3, 4, 5, 6, 7, 8};
//        lc19.lenLongestFibSubseq(s8);
        int[] s9 = {2, 4, 7, 8, 9, 10, 14, 15, 18, 23, 32, 50};
//        lc19.lenLongestFibSubseq2(s9);
        int[][] s10 = {{1, 2}, {3}, {3}, {}};
//        lc19.allPathsSourceTarget(s10);
        int[][] s11 = {{0, 1, 100}, {1, 2, 100}, {0, 2, 500}};
        int[][] s12 = {{4, 1, 1}, {1, 2, 3}, {0, 3, 2}, {0, 4, 10}, {3, 1, 1}, {1, 4, 3}};
        int[][] s14 = {{0, 1, 100}, {1, 2, 100}, {0, 2, 500}};
        int[][] s13 = {{0, 12, 28}, {5, 6, 39}, {8, 6, 59}, {13, 15, 7}, {13, 12, 38}, {10, 12, 35}, {15, 3, 23}, {7, 11, 26}, {9, 4, 65}, {10, 2, 38}, {4, 7, 7}, {14, 15, 31}, {2, 12, 44}, {8, 10, 34}, {13, 6, 29}, {5, 14, 89}, {11, 16, 13}, {7, 3, 46}, {10, 15, 19}, {12, 4, 58}, {13, 16, 11}, {16, 4, 76}, {2, 0, 12}, {15, 0, 22}, {16, 12, 13}, {7, 1, 29}, {7, 14, 100}, {16, 1, 14}, {9, 6, 74}, {11, 1, 73}, {2, 11, 60}, {10, 11, 85}, {2, 5, 49}, {3, 4, 17}, {4, 9, 77}, {16, 3, 47}, {15, 6, 78}, {14, 1, 90}, {10, 5, 95}, {1, 11, 30}, {11, 0, 37}, {10, 4, 86}, {0, 8, 57}, {6, 14, 68}, {16, 8, 3}, {13, 0, 65}, {2, 13, 6}, {5, 13, 5}, {8, 11, 31}, {6, 10, 20}, {6, 2, 33}, {9, 1, 3}, {14, 9, 58}, {12, 3, 19}, {11, 2, 74}, {12, 14, 48}, {16, 11, 100}, {3, 12, 38}, {12, 13, 77}, {10, 9, 99}, {15, 13, 98}, {15, 12, 71}, {1, 4, 28}, {7, 0, 83}, {3, 5, 100}, {8, 9, 14}, {15, 11, 57}, {3, 6, 65}, {1, 3, 45}, {14, 7, 74}, {2, 10, 39}, {4, 8, 73}, {13, 5, 77}, {10, 0, 43}, {12, 9, 92}, {8, 2, 26}, {1, 7, 7}, {9, 12, 10}, {13, 11, 64}, {8, 13, 80}, {6, 12, 74}, {9, 7, 35}, {0, 15, 48}, {3, 7, 87}, {16, 9, 42}, {5, 16, 64}, {4, 5, 65}, {15, 14, 70}, {12, 0, 13}, {16, 14, 52}, {3, 10, 80}, {14, 11, 85}, {15, 2, 77}, {4, 11, 19}, {2, 7, 49}, {10, 7, 78}, {14, 6, 84}, {13, 7, 50}, {11, 6, 75}, {5, 10, 46}, {13, 8, 43}, {9, 10, 49}, {7, 12, 64}, {0, 10, 76}, {5, 9, 77}, {8, 3, 28}, {11, 9, 28}, {12, 16, 87}, {12, 6, 24}, {9, 15, 94}, {5, 7, 77}, {4, 10, 18}, {7, 2, 11}, {9, 5, 41}};
//        int r12 =lc19.findCheapestPrice(3, s11, 0, 2, 1);
//        int r12 = lc19.findCheapestPrice2(3, s11, 0, 2, 1);
//        int r12 =lc19.findCheapestPrice(5, s12, 2, 1, 1);
//        int r12 =lc19.findCheapestPrice(3, s14, 0, 2, 1);
//        int r12 =lc19.findCheapestPrice(17, s13, 13, 4, 13);
//        System.out.println(r12);


//        lc19.knightProbability(3,2,0,0);
//        double r13 = lc19.knightProbability(8, 30, 6, 4);
//        System.out.println(r13);

//        MedianFinder medianFinder = new MedianFinder();
//        medianFinder.addNum(1);
//        medianFinder.addNum(2);
//        medianFinder.findMedian();

//        int[] r14 = lc19.numsSameConsecDiff(3, 7);
//        System.out.println(Arrays.toString(r14));

//        lc19.findPaths3(2, 2, 2, 0, 0);

//        lc19.countVowelPermutation(2);

        int[] s16 = {10, 5, 2, 6};
//        int r16 = lc19.numSubarrayProductLessThanK(s16, 100);
//        System.out.println(r16);

//        boolean r17 = lc19.isLongPressedName("alex","aaleex");
//        boolean r17 = lc19.isLongPressedName("kikcxmvzi", "kiikcxxmmvvzz");
//        System.out.println(r17);

        int[] s17 = {0, 0, 0, 1, 0, 1, 1, 0};
//        lc19.minKBitFlips(s17, 3);
//        lc19.minKnightMoves(1,1);

        int[] s18 = {90};
//        int r18 = lc19.minimumDifference(s18, 1);
//        System.out.println(r18);

        String[] s19 = {"233","97"};
        lc19.kthLargestNumber2(s19,1);

    }
}

//295
class MedianFinder {

    // big small big.size > small then big.peek eauqls the median
    PriorityQueue<Integer> small = new PriorityQueue<>();
    PriorityQueue<Integer> big = new PriorityQueue<>((a, b) -> b - a);

    /**
     * initialize your data structure here.
     */
    public MedianFinder() {

    }

    public void addNum(int num) {
        if (!big.isEmpty() && num <= big.peek()) {
            big.offer(num);
        } else {
            small.offer(num);
        }
        while (big.size() > small.size() + 1) {
            small.offer(big.poll());
        }
        while (big.size() < small.size()) {
            big.offer(small.poll());
        }
    }

    public double findMedian() {
        int size = big.size() + small.size();
        if ((size & 1) == 1) {
            return big.peek();
        } else {
            return (big.peek() + small.peek()) / 2.0;
        }
    }
}


class TrieSolution {
    public String longestWord(String[] words) {
        Trie trie = new Trie();
        int index = 0;
        for (String word : words) {
            trie.insert(word, ++index); //indexed by 1
        }
        trie.words = words;
        return trie.dfs();
    }

    class Node {
        int index;
        char cur;
        Map<Character, Node> children = new HashMap<>();

        public Node(char cur) {
            this.cur = cur;
        }
    }

    class Trie {
        String[] words;
        Node root = new Node('0');

        void insert(String word, int index) {
            char[] arr = word.toCharArray();
            Node cur = root;
            for (Character each : arr) {
                cur.children.computeIfAbsent(each, k -> new Node(each));
                cur = cur.children.get(each);
            }
            cur.index = index;
        }

        String dfs() {
            Queue<Node> queue = new LinkedList<>();
            String ans = "";
            queue.offer(root);
            while (!queue.isEmpty()) {
                Node cur = queue.poll();
                if (cur == root || cur.index != 0) {
                    if (cur != root) {
                        String word = words[cur.index - 1];
                        if (word.length() > ans.length() || (word.length() == ans.length() && word.compareTo(ans) < 0)) {
                            ans = word;
                        }
                    }
                    for (Node each : cur.children.values()) {
                        queue.offer(each);
                    }
                }
            }
            return ans;
        }
    }
}

class TrieS2 {
    Trie root = new Trie();
    int maxLen = 0;
    String res = "";

    class Trie {
        Trie[] children = new Trie[26];
        boolean isEnd;
        String word;

        void insert(String word) {
            Trie cur = this;
            for (int i = 0; i < word.length(); i++) {
                char c = word.charAt(i);
                if (cur.children[c - 'a'] == null) {
                    cur.children[c - 'a'] = new Trie();
                }
                cur = cur.children[c - 'a'];
            }
            cur.isEnd = true;
            cur.word = word;
        }
    }

    public void dfs(Trie root, int depth) {
        if (depth > 0 && !root.isEnd) return;
        if (depth > maxLen && root.isEnd) {
            maxLen = depth;
            res = root.word;
        }
        for (int i = 0; i < 26; i++) {
            if (root.children[i] != null) {
                dfs(root.children[i], depth + 1);
            }
        }
    }

    public String longestWord(String[] words) {
        Trie root = new Trie();
        for (String each : words) {
            root.insert(each);
        }
        dfs(root, 0);
        return res;
    }
}

class TrieS3 {
    Trie root = new Trie();

    class Trie {
        Trie[] children = new Trie[26];
        boolean isEnd;
        String word;

        void insert(String word) {
            Trie cur = this;
            for (int i = 0; i < word.length(); i++) {
                char c = word.charAt(i);
                if (cur.children[c - 'a'] == null) {
                    cur.children[c - 'a'] = new Trie();
                }
                cur = cur.children[c - 'a'];
            }
            cur.isEnd = true;
            cur.word = word;
        }
    }

    boolean finished = false, keepSearching = true;

    void dfs(String word, int pos, List<String> res, Trie root) {
        if (!keepSearching) return;
        if (pos == word.length()) finished = true;

        if (!finished) {
            int cur = word.charAt(pos) - 'a';
            if (root.children[cur] != null) {
                if (root.children[cur].isEnd)
                    res.add(root.children[cur].word);
                dfs(word, pos + 1, res, root.children[cur]);
            } else {
                keepSearching = false;
                return;
            }
        }
        if (finished) {
            for (int i = 0; i < 26; i++) {
                Trie now = root.children[i];
                if (now != null) {
                    if (now.isEnd)
                        res.add(now.word);
                    dfs(word, pos + 1, res, now);
                }
            }
        }
    }

    public List<List<String>> suggestedProducts(String[] products, String searchWord) {
        for (String each : products) {
            root.insert(each);
        }
        List<List<String>> res = new ArrayList<>();
        for (int i = 1; i < searchWord.length(); i++) {
            List<String> tmp = new ArrayList<>();
            finished = false;
            dfs(searchWord.substring(0, i), 0, tmp, root);
            res.add(tmp);
        }
        return res;
    }
}

class TrieS4 {
    Trie root = new Trie();

    class Trie {
        Trie[] children = new Trie[26];
        PriorityQueue<String> queue = new PriorityQueue<>((a, b) -> b.compareTo(a));
        String word = "";
    }

    void insert(Trie root, String word) {
        char[] arr = word.toCharArray();
        Trie cur = root;
        for (int i = 0; i < arr.length; i++) {
            if (cur.children[arr[i] - 'a'] == null) {
                cur.children[arr[i] - 'a'] = new Trie();
            }
            cur.children[arr[i] - 'a'].queue.offer(word);
            if (cur.children[arr[i] - 'a'].queue.size() > 3) {
                cur.children[arr[i] - 'a'].queue.poll();
            }
            cur = cur.children[arr[i] - 'a'];
        }
    }

    public List<List<String>> suggestedProducts(String[] products, String searchWord) {
        List<List<String>> ans = new ArrayList<>();
        for (String s : products)
            insert(root, s);
        startWith(searchWord, ans);
        return ans;
    }

    private void startWith(String searchWord, List<List<String>> ans) {
        char[] arr = searchWord.toCharArray();
        boolean exist = true;
        Trie cur = root;
        for (int i = 0; i < arr.length; i++) {
            List<String> tmp = new ArrayList<>();
            if (!exist || cur.children[arr[i] - 'a'] == null) {
                exist = false;
                ans.add(tmp);
                continue;
            }
            tmp.addAll(cur.children[arr[i] - 'a'].queue);
            Collections.sort(tmp);
            ans.add(tmp);
            cur = cur.children[arr[i] - 'a'];
        }
    }
}

class TrieS5 {
    class Trie {
        int num;
        Trie[] children = new Trie[10];
        boolean isEnd;
    }

    Trie root = new Trie();

    void insert(int num) {
        Trie cur = root;
        List<Integer> arr = convertToArr(num);
        for (int i = 0; i < arr.size(); i++) {
            int now = arr.get(i);
            if (cur.children[now] == null) {
                cur.children[now] = new Trie();
            }
            cur = cur.children[now];
        }
        cur.isEnd = true;
        cur.num = num;
    }

    private List<Integer> convertToArr(int num) {
        List<Integer> res = new ArrayList<>();
        while (num > 0) {
            res.add(num % 10);
            num /= 10;
        }
        Collections.reverse(res);
        return res;
    }

    public List<Integer> lexicalOrder(int n) {
        List<Integer> res = new ArrayList<>();
        for (int i = 1; i <= n; i++) {
            insert(i);
        }
        dfs(res, root);
        return res;
    }

    private void dfs(List<Integer> res, Trie cur) {
        for (int i = 0; i < 10; i++) {

            if (cur.children[i] != null) {
                if (cur.children[i].isEnd) {
                    res.add(cur.children[i].num);
                }
                dfs(res, cur.children[i]);
            }
        }
    }
}

class WordsFrequency {

    Map<String, Integer> map = new HashMap<>();

    public WordsFrequency(String[] book) {
        map.clear();
        for (String each : book) {
            map.put(each, map.getOrDefault(each, 0) + 1);
        }
    }

    public int get(String word) {
        return map.getOrDefault(word, 0);
    }
}

class Solution4 {
    class TrieNode {
        public static final int num = 26;
        TrieNode[] next;
        boolean isEnd;
        PriorityQueue<String> queue;

        public TrieNode() {
            next = new TrieNode[num];
            queue = new PriorityQueue<>((o1, o2) -> o2.compareTo(o1));
        }
    }

    private static final int size = 3;
    private TrieNode root;
    private List<List<String>> ans;

    public void insert(String word) {
        TrieNode node = root;
        for (char c : word.toCharArray()) {
            if (node.next[c - 'a'] == null) {
                node.next[c - 'a'] = new TrieNode();
            }
            node = node.next[c - 'a'];
            node.queue.offer(word);
            if (node.queue.size() > size)
                node.queue.poll();
        }
        node.isEnd = true;
    }

    public void startWith(String word) {
        TrieNode node = root;
        boolean exist = true;
        for (char c : word.toCharArray()) {
            if (!exist || node.next[c - 'a'] == null) {
                exist = false;
                ans.add(new ArrayList<>());
                continue;
            }
            node = node.next[c - 'a'];
            List<String> tmp = new ArrayList<>();
            while (!node.queue.isEmpty()) {
                tmp.add(node.queue.poll());
            }
            Collections.reverse(tmp);
            ans.add(tmp);
        }
    }

    public List<List<String>> suggestedProducts(String[] products, String searchWord) {
        ans = new ArrayList<>();
        root = new TrieNode();
        for (String s : products)
            insert(s);
        startWith(searchWord);
        return ans;
    }
}