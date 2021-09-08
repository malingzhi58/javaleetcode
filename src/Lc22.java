import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.*;
import java.util.Map;

public class Lc22 {
    public double[] calcEquation(List<List<String>> equations, double[] values, List<List<String>> queries) {
        int equationsSize = equations.size();
        UnionFind unionFind = new UnionFind(2 * equationsSize);
        // 第 1 步：预处理，将变量的值与 id 进行映射，使得并查集的底层使用数组实现，方便编码
        Map<String, Integer> hashMap = new HashMap<>(2 * equationsSize);
        int id = 0;
        for (int i = 0; i < equationsSize; i++) {
            List<String> equation = equations.get(i);
            String var1 = equation.get(0);
            String var2 = equation.get(1);

            if (!hashMap.containsKey(var1)) {
                hashMap.put(var1, id);
                id++;
            }
            if (!hashMap.containsKey(var2)) {
                hashMap.put(var2, id);
                id++;
            }
            unionFind.union(hashMap.get(var1), hashMap.get(var2), values[i]);
        }

        // 第 2 步：做查询
        int queriesSize = queries.size();
        double[] res = new double[queriesSize];
        for (int i = 0; i < queriesSize; i++) {
            String var1 = queries.get(i).get(0);
            String var2 = queries.get(i).get(1);

            Integer id1 = hashMap.get(var1);
            Integer id2 = hashMap.get(var2);

            if (id1 == null || id2 == null) {
                res[i] = -1.0d;
            } else {
                res[i] = unionFind.isConnected(id1, id2);
            }
        }
        return res;
    }

    private class UnionFind {

        private int[] parent;

        /**
         * 指向的父结点的权值
         */
        private double[] weight;


        public UnionFind(int n) {
            this.parent = new int[n];
            this.weight = new double[n];
            for (int i = 0; i < n; i++) {
                parent[i] = i;
                weight[i] = 1.0d;
            }
        }

        public void union(int x, int y, double value) {
            int rootX = find(x);
            int rootY = find(y);
            if (rootX == rootY) {
                return;
            }

            parent[rootX] = rootY;
            // 关系式的推导请见「参考代码」下方的示意图
            weight[rootX] = weight[y] * value / weight[x];
        }

        /**
         * 路径压缩
         *
         * @param x
         * @return 根结点的 id
         */
        public int find(int x) {
            if (x != parent[x]) {
                int origin = parent[x];
                parent[x] = find(parent[x]);
                weight[x] *= weight[origin];
            }
            return parent[x];
        }

        public double isConnected(int x, int y) {
            int rootX = find(x);
            int rootY = find(y);
            if (rootX == rootY) {
                return weight[x] / weight[y];
            } else {
                return -1.0d;
            }
        }
    }

    static byte[] longToByteArray(long value) {
        return ByteBuffer.allocate(8).putLong(value).array();
    }

    static long byteArrayToLong(byte[] array) {
        return ByteBuffer.wrap(array).getLong();
    }

    public int expressiveWords(String s, String[] words) {
        int res = 0;
        for (String each : words) {
            int id1 = 0, id2 = 0;
            boolean no = true;
            while (id1 < each.length() && id2 < s.length()) {
                if (each.charAt(id1) == s.charAt(id2)) {
                    int count1 = 0, count2 = 0;
                    char cur = each.charAt(id1);
                    while (id1 < each.length() && each.charAt(id1) == cur) {
                        id1++;
                        count1++;
                    }
                    while (id2 < s.length() && s.charAt(id2) == cur) {
                        id2++;
                        count2++;
                    }
                    if (count2 > count1 && count2 < 3) {
                        no = false;
                        break;
                    }
                } else {
                    no = false;
                    break;
                }
            }
            if (no && id2 == s.length()) res++;

        }
        return res;
    }

//    public static void main(String[] args) {
//        long maxValue = Long.parseUnsignedLong("FFFFFFFFFFFFFFFF", 16);
//        byte[] b = longToByteArray(maxValue);
//        System.out.println("b = " + Arrays.toString(b));
//
//        long value = byteArrayToLong(b);
//        System.out.println("value = " + value);
//        System.out.println("hex value = " + Long.toUnsignedString(value, 16));
//    }

    public String originalDigits(String s) {
        StringBuilder sb = new StringBuilder();
        Map<Character, Integer> map = new HashMap<>();
        for (int i = 0; i < s.length(); i++) {
            map.put(s.charAt(i), map.getOrDefault(s.charAt(i), 0) + 1);
        }

        return sb.toString();
    }

    public int countRangeSum(int[] nums, int lower, int upper) {
        int len = nums.length;
//        int[][] dp =new int[len][len];
        int res = 0;
        for (int i = 0; i < len; i++) {
            long sum = 0;
            for (int j = i; j < len; j++) {
                sum += nums[j];
//                dp[i][j]=sum;
                if (sum <= upper && sum >= lower) res++;
            }
        }
        return res;
    }

    public int numberOfWeakCharacters(int[][] properties) {
        int res = 0;
//        Arrays.sort(properties,(a,b)->{
//            if(a[0]!=b[0])return a[0]-b[0];
//            else return a[1]-b[1];
//        });
//        int[][] p2 = properties.clone();
//        Arrays.sort(p2,(a,b)->{
//            if(a[1]!=b[1])return a[1]-b[1];
//            else return a[0]-b[0];
//        });
        int maxx = Integer.MIN_VALUE, maxy = Integer.MIN_VALUE;
        int left = 0, len = properties.length, right = len - 1;
        for (int i = 0; i < len; i++) {
            if (properties[i][0] > maxx && properties[i][1] > maxy) {
                maxx = properties[i][0];
                maxy = properties[i][1];
            }
        }
        for (int i = 0; i < len; i++) {
            if (properties[i][0] < maxx && properties[i][1] < maxy) {
                res++;
            }
        }
        return res;
    }


    public int firstDayBeenInAllRooms(int[] nextVisit) {
        int len = nextVisit.length;
        long[] dp = new long[len];
        int MOD = (int) 1e9 + 7;
        dp[0] = 0;
        for (int i = 1; i < len; i++) {
            dp[i] = (2 * dp[i - 1] + 2 - dp[nextVisit[i - 1]] + MOD) % MOD;
        }
        return (int) dp[len - 1];
    }

    public String stackCis(String input) {
        Stack<Character> stack = new Stack<>();
        int id = 0, len = input.length();
        while (id < len) {
            if (input.charAt(id) == '[') {
                int time = 0;
                while (input.charAt(id) != ']') {
                    time = time * 10 + Integer.parseInt(String.valueOf(input.charAt(id)));
                    id++;
                }
                StringBuilder sb = new StringBuilder();
                while (stack.peek() != '(') {
                    sb.append(stack.pop());
                }
                stack.pop();
                sb = sb.reverse();
                while (time-- > 0) {
                    for (int i = 0; i < sb.length(); i++) {
                        stack.push(sb.charAt(i));
                    }
                }
                id++;
            } else {
                stack.push(input.charAt(id++));
            }
        }
        StringBuilder res = new StringBuilder();
        return res.toString();
    }


    private void dfs(int[] times, boolean[] vis, int[] nextVisit, int pos) {
        boolean all = true;
        vis[pos] = true;
        for (int i = 0; i < vis.length; i++) {
            if (!vis[i]) {
                all = false;
                break;
            }
        }
        if (all) {
            return;
        }
        times[pos]++;
        if (times[pos] % 2 == 0) {
            dfs(times, vis, nextVisit, (pos + 1) % vis.length);
        } else {
            dfs(times, vis, nextVisit, nextVisit[pos]);
        }
    }

    public int kthFactor(int n, int k) {
        int id = 1, fac = 1;
        while (id <= k) {
            if (n % fac == 0) {
                if (id == k)
                    return fac;
                id++;
            }
            fac++;
        }
        return fac;
    }
//    public int jobScheduling(int[] startTime, int[] endTime, int[] profit) {
//        int n = startTime.length;
//        int[][] jobs = new int[n][3];
//        for (int i = 0; i < n; i++) {
//            jobs[i] = new int[] {startTime[i], endTime[i], profit[i]};
//        }
//        Arrays.sort(jobs, (a, b)->a[1] - b[1]);
//        TreeMap<Integer, Integer> dp = new TreeMap<>();
//        dp.put(0, 0);
//        for (int[] job : jobs) {
//            int cur = dp.floorEntry(job[0]).getValue() + job[2];
//            if (cur > dp.lastEntry().getValue())
//                dp.put(job[1], cur);
//        }
//        return dp.lastEntry().getValue();
//    }

    public int jobScheduling(int[] startTime, int[] endTime, int[] profit) {
        int n = startTime.length;
        int[][] jobs = new int[n][3];
        for (int i = 0; i < n; i++) {
            jobs[i] = new int[]{startTime[i], endTime[i], profit[i]};
        }
        Arrays.sort(jobs, (a, b) -> a[1] - b[1]);
        TreeMap<Integer, Integer> map = new TreeMap<>();
        map.put(0, 0);
        int max = 0;
        for (int[] job : jobs) {
            int cur = map.floorEntry(job[0]).getValue() + job[2];
            max = Math.max(max, cur);
            map.put(job[1], cur);
        }
        return max;
//        return map.lastEntry().getValue();
    }

    public int findLongestChain(int[][] pairs) {
        TreeMap<Integer, Integer> map = new TreeMap<>();
        map.put(-1001, 0);
        Arrays.sort(pairs, (a, b) -> {
            if (a[1] != b[1]) return a[1] - b[1];
            else return a[0] - b[0];
        });
        for (int[] pair : pairs) {
            int cur = map.lowerEntry(pair[0]).getValue() + 1;
            if (cur > map.lastEntry().getValue()) {
                map.put(pair[1], cur);
            }
        }
        return map.lastEntry().getValue();
    }

    public int findLongestChain2(int[][] pairs) {
        Arrays.sort(pairs, (a, b) -> {
            if (a[1] != b[1]) return a[1] - b[1];
            else return a[0] - b[0];
        });
        int len = pairs.length;
        int max = 1, last = pairs[len - 1][0];
        for (int i = len - 2; i >= 0; i--) {
            if (pairs[i][1] < last) {
                max++;
                last = pairs[i][0];
            } else {
                last = Math.max(last, pairs[i][0]);
            }
        }
        return max;
    }

    public int maxEvents(int[][] events) {
        Arrays.sort(events, (a, b) -> {
            if (a[1] != b[1]) return a[1] - b[1];
            else return a[0] - b[0];
        });
        int max = 0, last = 1;
        for (int i = 0; i < events.length; i++) {
            if (events[i][1] >= last) {
                last = Math.max(last, events[i][0]);
                max++;
                last++;
            } else {
                last = Math.max(last, events[i][0]);
            }
        }
        return max;
    }

    public int numDecodings(String s) {
        int[] memo = new int[s.length()];
        Arrays.fill(memo, -1);
        return dfs2(s, 0, memo);
    }

    private int dfs2(String s, int pos, int[] memo) {
        if (pos == memo.length) return 1;
        if (memo[pos] != -1) return memo[pos];
        int res = 0;
        String one = s.substring(pos, pos + 1);
        if (!one.equals("0")) {
            res += dfs2(s, pos + 1, memo);
        }
        if (pos + 2 <= s.length()) {
            String two = s.substring(pos, pos + 2);
            int twoint = Integer.parseInt(two);
            if (!two.startsWith("0") && twoint >= 10 && twoint <= 26) {
                res += dfs2(s, pos + 2, memo);
            }
        }
        memo[pos] = res;
        return res;
    }

    private boolean[][] marked;
    private int[][] direction = {{-1, 0}, {0, -1}, {0, 1}, {1, 0}};

    public List<String> findWords(char[][] board, String[] words) {
        List<String> res = new ArrayList<>();
        for (int k = 0; k < words.length; k++) {
            boolean find = false;
            for (int i = 0; i < board.length; i++) {
                if (find) break;
                for (int j = 0; j < board[0].length; j++) {
                    if (find) break;
                    marked = new boolean[board.length][board[0].length];
                    String cur = words[k];
                    if (dfs3(cur, board, marked, i, j, 0)) {
                        res.add(cur);
                        find = true;
                    }
                }
            }
        }
        return res;
    }

    private boolean dfs3(String cur, char[][] board, boolean[][] marked, int row, int col, int pos) {
        if (pos == cur.length()) return true;
        if (row < 0 || row >= board.length || col < 0 || col >= board[0].length || marked[row][col]) {
            return false;
        }
        marked[row][col] = true;
        if (cur.charAt(pos) != board[row][col]) return false;
        for (int i = 0; i < direction.length; i++) {
            if (dfs3(cur, board, marked, row + direction[i][0], col + direction[i][1], pos + 1)) {
                return true;
            }
        }
        return false;
    }

//    int max =Integer.MAX_VALUE,t;
//    public int minimizeTheDifference(int[][] mat, int target) {
//        int row = mat.length,col = mat[0].length;
//        t= target;
//        dfs4(mat,0,0);
//        return max;
//    }
//
//    private void dfs4(int[][] mat, int row, int sum) {
//        if(row==mat.length){
//            max = Math.min(max,Math.abs(sum-t));
//            return;
//        }
//        for (int i = 0; i < mat[0].length; i++) {
//            dfs4(mat, row+1, sum+mat[row][i]);
//        }
//    }


    int max = Integer.MAX_VALUE, t;

    public int minimizeTheDifference(int[][] mat, int target) {
        int row = mat.length, col = mat[0].length;
        t = target;
        dfs4(mat, 0, 0);
        return max;
    }

    boolean[][] dp = new boolean[71][5000];

    private void dfs4(int[][] mat, int row, int sum) {
        if (row == mat.length) {
            max = Math.min(max, Math.abs(sum - t));
            return;
        }
        if (sum - t > max || dp[row][sum]) return;
        dp[row][sum] = true;
        for (int i = 0; i < mat[0].length; i++) {
            dfs4(mat, row + 1, sum + mat[row][i]);
        }

    }

    public static int maximumDifference(int[] inputArr) {
        int answer = 0;
        // Write your code here
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        pq.offer(inputArr[0]);
        for (int i = 1; i < inputArr.length; i++) {
            answer = Math.max(answer, inputArr[i] - pq.peek());
            pq.offer(inputArr[i]);
        }

        return answer;
    }

    public static int numberOfDays(int[] days) {
        int answer = 0;
        // Write your code here
        Map<Integer, Integer> map = new HashMap<>();
        for (int each : days) {
            map.put(each, map.getOrDefault(each, 0) + 1);
        }
        for (int key : map.keySet()) {
            if (map.get(key) % 2 == 1) {
                answer++;
            }
        }
        return answer;
    }

    public static String checkIPValidity(String addressIP) {
        String valid = "VALID", invalid = "INVALID";
        if (addressIP == null || addressIP.length() == 0) return invalid;
        int len = addressIP.length();
        for (int i = 0; i < len; i++) {
            if (!Character.isDigit(addressIP.charAt(i)) && addressIP.charAt(i) != '.') {
                return invalid;
            }
        }
        String[] splitIp = addressIP.split("\\.");
        if (splitIp.length != 4) return invalid;
        for (int i = 0; i < splitIp.length; i++) {
            if (splitIp[i].length() > 3) return invalid;
            if (splitIp[i].length() > 1 && splitIp[i].charAt(0) == '0') return invalid;
            int cur = Integer.parseInt(splitIp[i]);
            if (cur >= 0 && cur <= 255) continue;
            else return invalid;
        }

        return valid;
    }

    public int balancedStringSplit(String s) {
        int res = 0, lnum = 0, rnum = 0, idx = 0;
        while (idx < s.length()) {
            if (s.charAt(idx) == 'L') {
                lnum++;
            } else {
                rnum++;
            }
            if (lnum == rnum) {
                res++;
                lnum = 0;
                rnum = 0;
            }
            idx++;
        }
        return res;
    }

    public int connectSticks(int[] sticks) {
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        for (int each : sticks) {
            pq.offer(each);
        }
        int cost = 0;
        while (pq.size() > 1) {
            int one = pq.poll();
            int two = pq.poll();
            int combined = one + two;
            cost += combined;
            pq.offer(combined);
        }
        return cost;
    }

    static int minimumSwaps(int[] arr) {
        int res = 0, len = arr.length;
        Map<Integer, Integer> valToIdx = new HashMap<>();
        Map<Integer, Integer> idxToVal = new HashMap<>();
        for (int i = 0; i < len; i++) {
            valToIdx.put(arr[i], i + 1);
            idxToVal.put(i + 1, arr[i]);
        }
        int idx = 0;
        while (idx < len) {
            if (idx + 1 != arr[idx]) {
                res++;
                int cur = arr[idx];
                int ex = idxToVal.get(arr[idx]);
//                int expos = valToIdx.get(ex);
                swap(arr, idx, cur-1);
                idxToVal.put(idx + 1, ex);
                idxToVal.put(cur, cur);
                valToIdx.put(ex, idx + 1);
                valToIdx.put(cur, cur);
            } else {
                idx++;
            }
        }
        return res;

    }

    private static void swap(int[] arr, int idx, int expos) {
        int tmp = arr[idx];
        arr[idx] = arr[expos];
        arr[expos] = tmp;
    }



    public int findMaximizedCapital(int k, int w, int[] profits, int[] capital) {
        int len =profits.length;
        int[][] projs = new int[len][2];
        for (int i = 0; i < len; i++) {
            projs[i]= new int[]{profits[i],capital[i]};
        }
        PriorityQueue<int[]> toBePicked = new PriorityQueue<>((a,b)->(b[0]-a[0]));
        PriorityQueue<int[]> notEnough = new PriorityQueue<>((a,b)->(a[1]-b[1]));
        for(int[] each:projs){
            notEnough.offer(each);
        }
        while(!notEnough.isEmpty()){
            while(!notEnough.isEmpty()&&notEnough.peek()[1]<=w){
                toBePicked.offer(notEnough.poll());
            }
            if(k-->0&&!toBePicked.isEmpty()){
                w+=toBePicked.poll()[0];
            }
            if(k==0)break;
        }
        while(k-->0&&!toBePicked.isEmpty()){
            w+=toBePicked.poll()[0];
        }
        return w;
    }

    public static void main(String[] args) {
        Lc22 lc22 = new Lc22();
//        lc22.calcEquation(Arrays.asList())
        String[] s1 = {"hello", "hi", "helo"};
//        int r1 = lc22.expressiveWords("heeellooo", s1);
//        System.out.println(r1);
        int[][] s2 = {{1, 5}, {10, 4}, {4, 3}};
//        lc22.numberOfWeakCharacters(s2);
        int[] s3 = {0, 0};
//        int r3 = lc22.firstDayBeenInAllRooms(s3);
//        System.out.println(r3);
        int[] s4 = {1, 2, 3, 3}, s5 = {3, 4, 5, 6}, s6 = {50, 10, 40, 70};
//        int r4 = lc22.jobScheduling(s4, s5, s6);
//        System.out.println(r4);
        char[][] s7 = {{'a', 'b', 'c', 'e'}, {'x', 'x', 'c', 'd'}, {'x', 'x', 'b', 'a'}};
        String[] s8 = {"abc", "abcd"};
//        lc22.findWords(s7, s8);
//        String r9=checkIPValidity("100.100.100.100");
//        System.out.println(r9);

        int[] s10 = {4, 3, 1, 2};
        int r10 = minimumSwaps(s10);
        System.out.println(r10);
        ArrayList<Object> ll = new ArrayList<>();

    }
}






