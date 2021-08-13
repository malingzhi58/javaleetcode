import javax.swing.*;
import java.util.*;
import java.util.stream.Collectors;

public class Lc15 {

    // 邻接表存储的图
    public boolean circularArrayLoop(int[] nums) {
        List<List<Integer>> graph = new ArrayList<>();
        int len = nums.length;
        for (int i = 0; i < len; i++) {
            graph.add(new ArrayList<>());
        }
        // forward
        int[] inDegree = new int[len];
        for (int i = 0; i < len; i++) {
            if (nums[i] <= 0) continue;
            int next = ((i + nums[i]) % len + len) % len;
            graph.get(i).add(next);
            inDegree[next]++;
        }
        if (topoSort(graph, inDegree)) {
            return true;
        }
        graph.clear();
        for (int i = 0; i < len; i++) {
            graph.add(new ArrayList<>());
        }
        Arrays.fill(inDegree, 0);
        for (int i = 0; i < len; i++) {
            if (nums[i] >= 0) continue;
            int next = ((i + nums[i]) % len + len) % len;
            graph.get(i).add(next);
            inDegree[next]++;
        }
        if (topoSort(graph, inDegree)) {
            return true;
        }
        return false;
    }

    private boolean topoSort(List<List<Integer>> graph, int[] inDegree) {
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < inDegree.length; i++) {
            if (inDegree[i] == 0) {
                queue.offer(i);
            }
        }
        while (!queue.isEmpty()) {
            int cur = queue.poll();
            List<Integer> to = graph.get(cur);
            for (int i = 0; i < to.size(); i++) {
                int toNum = to.get(i);
                inDegree[toNum]--;
                if (inDegree[toNum] == 0) {
                    queue.offer(toNum);
                }
            }
        }
        for (int i = 0; i < inDegree.length; i++) {
            if (inDegree[i] != 0) return true;
        }
        return false;
    }

    public boolean canFinish(int numCourses, int[][] prerequisites) {
        int[] inDegree = new int[numCourses];
        List<List<Integer>> graph = new ArrayList<>();
        for (int i = 0; i < numCourses; i++) {
            graph.add(new ArrayList<>());
        }
        for (int[] each : prerequisites) {
            graph.get(each[1]).add(each[0]);
            inDegree[each[0]]++;
        }
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < numCourses; i++) {
            if (inDegree[i] == 0) {
                queue.offer(i);
            }
        }
        while (!queue.isEmpty()) {
            int cur = queue.poll();
            for (int num : graph.get(cur)) {
                inDegree[num]--;
                if (inDegree[num] == 0) {
                    queue.offer(num);
                }
            }
        }
        for (int i = 0; i < numCourses; i++) {
            if (inDegree[i] != 0) return false;
        }
        return true;
    }

    public int[] findOrder(int numCourses, int[][] prerequisites) {
        int[] inDegree = new int[numCourses];
        List<List<Integer>> graph = new ArrayList<>();
        for (int i = 0; i < numCourses; i++) {
            graph.add(new ArrayList<>());
        }
        for (int[] each : prerequisites) {
            graph.get(each[1]).add(each[0]);
            inDegree[each[0]]++;
        }
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < numCourses; i++) {
            if (inDegree[i] == 0) {
                queue.offer(i);
            }
        }
        int[] res = new int[numCourses];
        int idx = 0;
        while (!queue.isEmpty()) {
            int cur = queue.poll();
            res[idx++] = cur;
            for (int num : graph.get(cur)) {
                inDegree[num]--;
                if (inDegree[num] == 0) {
                    queue.offer(num);
                }
            }
        }
        if (idx == numCourses) return res;
        return new int[]{};
    }

    int row = 0, col = 0;
    int[][] directions = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    public int longestIncreasingPath(int[][] matrix) {
        row = matrix.length;
        col = matrix[0].length;
        int[][] dp = new int[row][col];
        int max = 0;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                max = Math.max(max, dfs(matrix, dp, i, j));
            }
        }
        return max;
    }

    private int dfs(int[][] matrix, int[][] dp, int x, int y) {
        if (!isValid(x, y)) return 0;
        int cur = matrix[x][y];
        int max = 0;
        for (int[] each : directions) {
            int x1 = x + each[0], y1 = y + each[1];
            if (!isValid(x1, y1)) continue;
            if (matrix[x1][y1] <= matrix[x][y]) continue;
            if (dp[x1][y1] != 0) {
                max = Math.max(max, dp[x1][y1]);
            } else {
                max = Math.max(max, dfs(matrix, dp, x1, y1));
            }
        }
        dp[x][y] = max + 1;
        return dp[x][y];
    }

    private boolean isValid(int x, int y) {
        return (x >= 0) && (x < row) && (y >= 0) && (y < col);
    }

    public String makeFancyString(String s) {
        StringBuilder sb = new StringBuilder();
        int start = 0;
        for (int i = 0; i < s.length(); i++) {
            if (i < 2) {
                sb.append(s.charAt(i));
            }
            if (s.charAt(i) == s.charAt(i - 1) && s.charAt(i) == s.charAt(i - 2)) {
                continue;
            } else {
                sb.append(s.charAt(i));
            }
        }
        return sb.toString();
    }

    public boolean checkMove(char[][] board, int rMove, int cMove, char color) {
        char oppo = ' ';
        if (color == 'W') oppo = 'B';
        else oppo = 'W';
        int limit = 0, row = board.length, col = board[0].length;
        boolean find = false;
        // upper
        for (int i = rMove - 1; i >= 0; i--) {
            if (board[i][cMove] == color) {
                find = true;
                limit = i;
                break;
            }
        }
        if (find && rMove - limit >= 2) {
            find = true;
            for (int i = limit + 1; i < rMove; i++) {
                if (board[i][cMove] == color || board[i][cMove] == '.') {
                    find = false;
                }
            }
            if (find) return true;
        }
        // down
        find = false;
        for (int i = rMove + 1; i < row; i++) {
            if (board[i][cMove] == color) {
                find = true;
                limit = i;
                break;
            }
        }
        if (find && limit - rMove >= 2) {
            find = true;
            for (int i = rMove + 1; i < limit; i++) {
                if (board[i][cMove] == color || board[i][cMove] == '.') {
                    find = false;
                }
            }
            if (find) return true;
        }
        // left
        find = false;
        for (int i = cMove - 1; i >= 0; i--) {
            if (board[rMove][i] == color) {
                find = true;
                limit = i;
                break;
            }
        }
        if (find && cMove - limit >= 2) {
            find = true;
            for (int i = limit + 1; i < cMove; i++) {
                if (board[rMove][i] == color || board[rMove][i] == '.') {
                    find = false;
                }
            }
            if (find) return true;
        }
        //right
        find = false;
        for (int i = cMove + 1; i < col; i++) {
            if (board[rMove][i] == color) {
                find = true;
                limit = i;
                break;
            }
        }
        if (find && limit - cMove >= 2) {
            find = true;
            for (int i = cMove + 1; i < limit; i++) {
                if (board[rMove][i] == color || board[rMove][i] == '.') {
                    find = false;
                }
            }
            if (find) return true;
        }
        //diag up left
        find = false;
        int jlimit = 0;
        for (int i = rMove - 1, j = cMove - 1; i >= 0 && j >= 0; i--, j--) {
            if (board[i][j] == color) {
                find = true;
                limit = i;
                jlimit = j;
                break;
            }
        }
        if (find && rMove - limit >= 2) {
            find = true;
            for (int i = limit + 1, j = jlimit + 1; i < rMove && j < cMove; i++, j++) {
                if (board[i][j] == color || board[i][j] == '.') {
                    find = false;
                }
            }
            if (find) return true;
        }
        //diag up right
        find = false;
        jlimit = 0;
        for (int i = rMove - 1, j = cMove + 1; i >= 0 && j < col; i--, j++) {
            if (board[i][j] == color) {
                find = true;
                limit = i;
                jlimit = j;
                break;
            }
        }
        if (find && rMove - limit >= 2) {
            find = true;
            for (int i = limit + 1, j = jlimit - 1; i < rMove && j > cMove; i++, j--) {
                if (board[i][j] == color || board[i][j] == '.') {
                    find = false;
                }
            }
            if (find) return true;
        }
        //diag down left
        find = false;
        jlimit = 0;
        for (int i = rMove + 1, j = cMove - 1; i < row && j >= 0; i++, j--) {
            if (board[i][j] == color) {
                find = true;
                limit = i;
                jlimit = j;
                break;
            }
        }
        if (find && limit - rMove >= 2) {
            find = true;
            for (int i = limit - 1, j = jlimit + 1; i > rMove && j < cMove; i--, j++) {
                if (board[i][j] == color || board[i][j] == '.') {
                    find = false;
                }
            }
            if (find) return true;
        }
        //diag down right
        find = false;
        jlimit = 0;
        for (int i = rMove + 1, j = cMove + 1; i < row && j < col; i++, j++) {
            if (board[i][j] == color) {
                find = true;
                limit = i;
                jlimit = j;
                break;
            }
        }
        if (find && limit - rMove >= 2) {
            find = true;
            for (int i = limit - 1, j = jlimit - 1; i > rMove && j > cMove; i--, j--) {
                if (board[i][j] == color || board[i][j] == '.') {
                    find = false;
                }
            }
            if (find) return true;
        }
        return false;
    }

//    public int minSpaceWastedKResizing(int[] nums, int k) {
//        int max = Arrays.stream(nums).max().getAsInt();
//        int maxIdx = 0, len = nums.length;
//        // 0 for idx, 1 for num
//        PriorityQueue<int[]> queue = new PriorityQueue<>((a, b) -> {
//            if (a[1] != b[1]) return a[1] - b[1];
//            else return a[0] - b[0];
//        });
//        for (int i = 0; i < len; i++) {
//            queue.offer(new int[]{i, nums[i]});
//            if (nums[i] == max) {
//                maxIdx = i;
//                break;
//            }
//        }
//        int[] res = new int[len];
//        Arrays.fill(res, max);
//        while (k > 0) {
//            int[] cur = queue.poll();
//            int id = cur[0], num = cur[1];
//            if (id < maxIdx) {
//                for (int i = id; i < maxIdx && isCloseTo(nums[i], num, res[i]) && num >= nums[i]; i++) {
//                    res[i] = num;
//                }
//            } else {
//                for (int i = id; i < len; i++) {
//                    res[i] = num;
//                }
//            }
//            k--;
//        }
//        int sum = 0;
//        for (int i = 0; i < len; i++) {
//            sum += (res[i] - nums[i]);
//        }
//        return sum;
//    }

    boolean isCloseTo(int num, int changto, int res) {
        return changto - num < res - num;
    }

    public int tribonacci(int n) {
        int pre = 0, one = 1, two = 1, three = 0;
        if (n == 0) return 0;
        if (n == 1 || n == 2) return 1;

        while (n - 3 >= 0) {
            three = pre + one + two;
            pre = one;
            one = two;
            two = three;
            n--;
        }
        return three;
    }

    public int nthSuperUglyNumber(int n, int[] primes) {
        PriorityQueue<Long> pq = new PriorityQueue<>();
        pq.offer(1L);
        Set<Long> set = new HashSet<>();
        set.add(1L);
        int ugly = 1;
        for (int i = 0; i < n; i++) {
            long cur = pq.poll();
            ugly = (int) cur;
            for (int each : primes) {
                long next = each * cur;
                if (set.contains(next)) {
                    continue;
                } else {
                    set.add(next);
                    pq.offer(next);
                }
            }
        }
        return ugly;
    }

    int INF = 0x3f3f3f3f;
    int n, sum;

    public int minSpaceWastedKResizing(int[] nums, int k) {
        n = nums.length;
        sum = 0;
        int[][] premax = new int[n][n];
        for (int i = 0; i < n; i++) {
            int m = 0;
            for (int j = i; j < n; j++) {
                m = Math.max(m, nums[j]);
                // premax[i:j] both inclusive
                premax[i][j] = m * (j - i + 1);
            }
            sum += nums[i];
        }
        int[][] dp = new int[n][k + 2];
        for (int i = 0; i < n; i++) {
            Arrays.fill(dp[i], INF);
        }
        for (int i = 0; i < n; i++) {
            for (int j = 1; j <= k + 1; j++) {
                for (int l = 0; l <= i; l++) {
                    dp[i][j] = Math.min(dp[i][j], (l == 0 ? 0 : dp[l - 1][j - 1]) + premax[l][i]);
                }
            }
        }
        return dp[n - 1][k + 1] - sum;
    }

    public int minSpaceWastedKResizing2(int[] nums, int k) {
        n = nums.length;
        sum = 0;
        int[][] premax = new int[n][n];
        for (int i = 0; i < n; i++) {
            int m = 0;
            for (int j = i; j < n; j++) {
                m = Math.max(m, nums[j]);
                premax[i][j] = m * (j + 1 - i);
            }
            sum += nums[i];
        }

        int[][] dp = new int[n][k + 2];
        for (int i = 0; i < n; i++) Arrays.fill(dp[i], INF);

        for (int i = 0; i < n; i++)
            for (int j = 1; j <= k + 1; j++)
                for (int l = 0; l <= i; l++) {
                    if (l > 0) {
                        int dp_1 = dp[l - 1][j - 1];
                        int another = dp[l - 1][j - 1] + premax[l][i];
                    }
                    dp[i][j] = Math.min(dp[i][j], (l == 0 ? 0 : dp[l - 1][j - 1]) + premax[l][i]);
                }
        return dp[n - 1][k + 1] - sum;
    }


    private boolean valid(int x, int y, int row, int col) {
        return x >= 0 && x < row && y >= 0 && y < col;
    }

    public int numberOfArithmeticSlices2(int[] nums) {
        int len = nums.length;
        int dp = 0;
        for (int i = 2; i < len; i++) {
            if (nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]) {
                if (dp == 0) {
                    dp = 1;
                } else {
                    dp += 2;
                }
            }
        }
        return dp;
    }

    public int numberOfArithmeticSlices(int[] nums) {
        int sum = 0, len = nums.length;
        Set<Integer> set = new HashSet<>();
        Map<String, Integer> map = new HashMap<>();
        for (int i = 0; i < len; i++) {
            for (int j = i + 1; j < len; j++) {
                int gap = nums[j] - nums[i];
                if (set.contains(gap)) continue;
                set.add(gap);
                int count = 2, pre = nums[j];
                for (int k = j + 1; k < len; k++) {
                    if (nums[k] - pre == gap) {
                        count++;
                        pre = nums[k];
                    }
                }
                if (gap == 0) {
                    sum += permutation(3, count, map);
                } else {
                    sum += (count - 1) * (count - 2) / 2;
                }
            }
        }
        return sum;
    }

    private int permutation(int k, int count, Map<String, Integer> map) {
        int sum = 0;
        for (int i = k; i <= count; i++) {
            sum += factorial(i, count, map);
        }
        return sum;
    }

    Map<Integer, Integer> fac = new HashMap<>();

    private int factorial(int head, int base, Map<String, Integer> map) {
        String tar = head + ":" + base;
        if (map.containsKey(tar)) {
            return map.get(tar);
        }
        int res = 0;
        if (base == head) {
            res = 1;
        } else {
            res = fac(base) / fac(head) / fac(base - head);
        }

        map.put(tar, res);
        return res;

    }

    private int fac(int base) {
        if (fac.containsKey(base)) {
            return fac.get(base);
        }
        if (base == 0) {
            return 0;
        }
        int res = 1;
        for (int i = 1; i <= base; i++) {
            res *= i;
        }
        fac.put(base, res);
        return res;
    }

    public ListNode mergeKLists(ListNode[] lists) {
        PriorityQueue<ListNode> queue = new PriorityQueue<>((a, b) -> a.val - b.val);
        for (ListNode each : lists) {
            if (each == null) continue;
            queue.offer(each);
        }
        ListNode dum = new ListNode(), cur = dum;
        while (!queue.isEmpty()) {
            ListNode tmp = queue.poll();
            cur.next = tmp;
            cur = cur.next;
            if (tmp.next != null) {
                queue.offer(tmp.next);
            }
        }
        return dum.next;
    }

//    public long maxProduct(String s) {
//        PriorityQueue<long[]> palin = new PriorityQueue<>((a, b) -> {
//            if (b[2] != a[2]) {
//                if (b[2] - a[2] > 0) return 1;
//                else return -1;
//            } else {
//                if (a[0] - b[0] > 0) return 1;
//                else if (a[0] - b[0] < 0) return -1;
//                else return 0;
//            }
//        });
//        int cur = 0, len = s.length();
//        while (cur < len) {
//            int side = 0;
//            while (isValid(s, cur, side)) {
//                side++;
//            }
//            palin.add(new long[]{cur - (side - 1), cur + (side - 1), 1 + (side - 1) * 2L});
//            cur++;
//        }
//        long[] first = palin.poll();
//
//    }

    public long maxProduct(String s) {
        List<long[]> palin = new ArrayList<>();
        int cur = 0, len = s.length();
        while (cur < len) {
            int side = 1;
            while (isValid(s, cur, side)) {
                palin.add(new long[]{cur - (side), cur + (side), 1 + (side) * 2L});
                side++;
            }
            if (side > 1) {
                palin.add(new long[]{cur - (side - 1), cur + (side - 1), 1 + (side - 1) * 2L});
            }
            cur++;
        }
        long max = 1L;
        for (int i = 0; i < palin.size(); i++) {
            long[] p1 = palin.get(i);
            for (int j = i + 1; j < palin.size(); j++) {
                long[] p2 = palin.get(j);
                if (p1[1] < p2[0] || p1[0] > p2[1]) {
                    max = Math.max(max, p1[2] * p2[2]);
                }

            }
            if (p1[2] < s.length()) {
                max = Math.max(max, p1[2]);
            }

        }
        return max;
    }

    private boolean isValid(String s, int cur, int side) {
        if (cur - side >= 0 && cur + side < s.length() && s.charAt(cur - side) == s.charAt(cur + side)) {
            return true;
        } else {
            return false;
        }
    }

    private boolean isValid2(String s, int left, int right) {
        if (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
            return true;
        } else {
            return false;
        }
    }

    int[][] memo;

    public int longestPalindromeSubseq(String s) {
        int len = s.length();
        memo = new int[len][len];
        helper(s, 0, len - 1);
        return memo[0][len - 1];
    }

    private int helper(String s, int start, int end) {
        if (start == end) return 1;
        if (start > end) return 0;
        if (memo[start][end] != 0) return memo[start][end];
        int ans = 0;
        if (s.charAt(start) == s.charAt(end)) {
            ans = helper(s, start + 1, end - 1) + 2;
        } else {
            ans = Math.max(helper(s, start, end - 1), helper(s, start + 1, end));
        }
        memo[start][end] = ans;
        return ans;
    }

    public int longestPalindromeSubseq2(String s) {
        int len = s.length();
        int[][] dp = new int[len][len];
        for (int i = 0; i < len; i++) {
            for (int j = i; j >= 0; j--) {
                if (i == j) {
                    dp[j][i] = 1;
                    continue;
                }
                if (s.charAt(i) == s.charAt(j)) {
//                    if(j>=1&&i+1<len)
                    if (i > j + 1)
                        dp[j][i] = dp[j + 1][i - 1] + 2;
                    if (i == j + 1)
                        dp[j][i] = 2;
                } else {
                    dp[j][i] = Math.max(dp[j + 1][i], dp[j][i - 1]);
                }
            }
        }
        return dp[0][len - 1];
    }


    public int[] sortItems(int n, int m, int[] group, List<List<Integer>> beforeItems) {
        // 第 1 步：数据预处理，给没有归属于一个组的项目编上组号
        for (int i = 0; i < group.length; i++) {
            if (group[i] == -1) {
                group[i] = m;
                m++;
            }
        }

        // 第 2 步：实例化组和项目的邻接表
        List<Integer>[] groupAdj = new ArrayList[m];
        List<Integer>[] itemAdj = new ArrayList[n];
        for (int i = 0; i < m; i++) {
            groupAdj[i] = new ArrayList<>();
        }
        for (int i = 0; i < n; i++) {
            itemAdj[i] = new ArrayList<>();
        }

        // 第 3 步：建图和统计入度数组
        int[] groupsIndegree = new int[m];
        int[] itemsIndegree = new int[n];

        int len = group.length;
        for (int i = 0; i < len; i++) {
            int currentGroup = group[i];
            for (int beforeItem : beforeItems.get(i)) {
                int beforeGroup = group[beforeItem];
                if (beforeGroup != currentGroup) {
                    groupAdj[beforeGroup].add(currentGroup);
                    groupsIndegree[currentGroup]++;
                }
            }
        }

        for (int i = 0; i < n; i++) {
            for (Integer item : beforeItems.get(i)) {
                itemAdj[item].add(i);
                itemsIndegree[i]++;
            }
        }

        // 第 4 步：得到组和项目的拓扑排序结果
        List<Integer> groupsList = topologicalSort(groupAdj, groupsIndegree, m);
        if (groupsList.size() == 0) {
            return new int[0];
        }
        List<Integer> itemsList = topologicalSort(itemAdj, itemsIndegree, n);
        if (itemsList.size() == 0) {
            return new int[0];
        }

        // 第 5 步：根据项目的拓扑排序结果，项目到组的多对一关系，建立组到项目的一对多关系
        // key：组，value：在同一组的项目列表
        Map<Integer, List<Integer>> groups2Items = new HashMap<>();
        for (Integer item : itemsList) {
            groups2Items.computeIfAbsent(group[item], key -> new ArrayList<>()).add(item);
        }

        // 第 6 步：把组的拓扑排序结果替换成为项目的拓扑排序结果
        List<Integer> res = new ArrayList<>();
        for (Integer groupId : groupsList) {
            List<Integer> items = groups2Items.getOrDefault(groupId, new ArrayList<>());
            res.addAll(items);
        }
        return res.stream().mapToInt(Integer::valueOf).toArray();
    }

    private List<Integer> topologicalSort(List<Integer>[] adj, int[] inDegree, int n) {
        List<Integer> res = new ArrayList<>();
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            if (inDegree[i] == 0) {
                queue.offer(i);
            }
        }

        while (!queue.isEmpty()) {
            Integer front = queue.poll();
            res.add(front);
            for (int successor : adj[front]) {
                inDegree[successor]--;
                if (inDegree[successor] == 0) {
                    queue.offer(successor);
                }
            }
        }

        if (res.size() == n) {
            return res;
        }
        return new ArrayList<>();
    }

    public int maxEnvelopes(int[][] envelopes) {
        int len = envelopes.length;
        int[] dp = new int[len];
        Arrays.fill(dp, 1);
        Arrays.sort(envelopes, (a, b) -> {
            if (a[0] != b[0]) return a[0] - b[0];
            else return a[1] - b[1];
        });
        for (int i = 1; i < len; i++) {
            for (int j = 0; j < i; j++) {
                if (envelopes[i][0] > envelopes[j][0] && envelopes[i][1] > envelopes[j][1]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
        }
        return Arrays.stream(dp).max().getAsInt();
    }

    public List<Integer> largestDivisibleSubset(int[] nums) {
        int len = nums.length;
        int[] dp = new int[len];
        int[] trace = new int[len];
        for (int i = 0; i < len; i++) {
            int pre = i;
            for (int j = 0; j < i; j++) {
                if (nums[j] % nums[i] == 0 || nums[i] % nums[j] == 0) {
                    if (dp[j] > dp[i]) {
                        dp[i] = dp[j];
                        pre = j;
                    }
                }
            }
            dp[i] += 1;
            trace[i] = pre;
        }
        int max = 0, traceid = 0;
        for (int i = 0; i < len; i++) {
            if (dp[i] > max) {
                max = dp[i];
                traceid = i;
            }
        }
        List<Integer> res = new ArrayList<>();
        while (true) {
            if (trace[traceid] == traceid) {
                res.add(nums[traceid]);
                break;
            }
            res.add(nums[traceid]);
            traceid = trace[traceid];
        }
        return res;
    }

    public List<Integer> largestDivisibleSubset2(int[] nums) {
        int len = nums.length;
        Set[] dp = new HashSet[len];
        Arrays.sort(nums);
        for (int i = 0; i < len; i++) {
            dp[i] = new HashSet<>();
        }
        for (int i = 0; i < len; i++) {
            int size = 0, tag = i;
            for (int j = i - 1; j >= 0; j--) {
//                if (i == j) {

                if (isValid3(dp[j], nums[i]) && dp[j].size() > size) {
                    size = dp[j].size();
                    tag = j;
                }
            }
            dp[i].add(nums[i]);
            if (tag != i)
                dp[i].addAll(dp[tag]);
        }
        int max = 0;
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < len; i++) {
            if (dp[i].size() > max) {
                res = new ArrayList<>(dp[i]);
                max = dp[i].size();
            }
        }
        return res;
    }

    private boolean isValid3(Set<Integer> integers, int num) {
        for (Integer cur : integers) {
            if (cur % num != 0 && num % cur != 0) return false;
        }
        return true;
    }

    public boolean canJump(int[] nums) {
        int jump = nums[0], len = nums.length, cur = 1;
        while (cur < len) {
            if (jump >= cur) {
                jump = Math.max(cur + nums[cur], jump);
            } else {
                return false;
            }
            cur++;
        }
        return true;
    }

    public int jump2(int[] nums) {
        int len = nums.length;
        int[] dp = new int[len];
        int jump = nums[0];
        int INF = 0x3f3f3f3f;
        Arrays.fill(dp, INF);
        dp[0] = 0;
        for (int i = 1; i < len; i++) {
            for (int j = 0; j < i; j++) {
                if (j + nums[j] < i) continue;
                dp[i] = Math.min(dp[i], dp[j] + 1);
            }
        }
        return dp[len - 1];
    }

    public int jump(int[] nums) {
        int len = nums.length;
        int jump = 0, cur = 0, time = 0;
        while (jump < len) {
            if (jump < cur + nums[cur]) {
                time++;
                jump = cur + nums[cur];
            }
            cur++;
        }
        return time;
    }

    public boolean canReach(int[] arr, int start) {
        int len = arr.length;
        int[][] memo = new int[len][2];
        // 0 for -, 1 for +
        // 0 for not initialized, 1 for false, 2 for true
        boolean r1 = helper2(arr, memo, start, 0, new HashSet<>());
        boolean r2 = helper2(arr, memo, start, 1, new HashSet<>());
        return r1 || r2;
    }

    private boolean helper2(int[] arr, int[][] memo, int start, int move, Set<Integer> set) {
        if (set.contains(start)) {
//            memo[start][move] = 1;
            return false;
        }
        set.add(start);
        if (memo[start][move] == 1) return false;
        if (arr[start] == 0) return true;
        boolean r1 = false, r2 = false;
        if (move == 1) {
            if (start - arr[start] >= 0) {
                if (helper2(arr, memo, start - arr[start], 0, set) || helper2(arr, memo, start - arr[start], 1, set)) {
                    return true;
                }
            }
        } else {
            if (start + arr[start] < arr.length) {
                if (helper2(arr, memo, start + arr[start], 0, set) || helper2(arr, memo, start + arr[start], 1, set)) {
                    return true;
                }
            }
        }
        set.remove(start);
        memo[start][move] = 1;
        return false;

    }

    int d = 0;
    int[] me;

    public int maxJumps(int[] arr, int _d) {
        int len = arr.length;
        d = _d;
        me = new int[len];
        int max = 0;
        for (int i = 0; i < len; i++) {
            max = Math.max(max, dfs2(arr, i));
        }
        return max;
    }

    private int dfs2(int[] arr, int start) {
        if (me[start] != 0) return me[start];
        int cur = arr[start];
        int max = 0;
        for (int i = 1; i <= d; i++) {
            if (start + i < arr.length && cur > arr[start + i]) {
                int tmp = dfs2(arr, start + i);
                max = Math.max(max, tmp);
            } else {
                break;
            }
        }
        for (int i = 1; i <= d; i++) {
            if (start - i >= 0 && cur > arr[start - i]) {
                int tmp = dfs2(arr, start - i);
                max = Math.max(max, tmp);
            } else {
                break;
            }
        }
        me[start] = max + 1;
        return max + 1;
    }

    //    public int minJumps(int[] arr) {
//        int len = arr.length;
//        int INF = 0x3f3f3f3f;
//        int[] dp = new int[len];
//        Arrays.fill(dp, INF);
//        dp[len - 1] = 0;
//        Map<Integer, List<Integer>> map = new HashMap<>();
//        for (int i = 0; i < len; i++) {
//            map.computeIfAbsent(arr[i], k -> new ArrayList<>()).add(i);
//        }
//        for (int i = len - 2; i >= 0; i--) {
//            List<Integer> cur = map.get(arr[i]);
//            for (int j = 0; j < cur.size(); j++) {
//                int same=cur.get(j);
//                if (same != i) {
//                    int tmp = dp[same]+1;
//                    dp[i]=Math.min(dp[i],dp[same]+1);
//                }
//            }
//            dp[i]=Math.min(dp[i],dp[i+1]+1);
//        }
//        for (int i = len - 2; i >= 0; i--) {
//            List<Integer> cur = map.get(arr[i]);
//            for (int j = 0; j < cur.size(); j++) {
//                int same=cur.get(j);
//                if (same != i) {
//                    int tmp = dp[same]+1;
//                    dp[i]=Math.min(dp[i],dp[same]+1);
//                }
//            }
//            if(i>=1)
//                dp[i]=Math.min(dp[i],dp[i-1]+1);
//        }
//        return dp[0];
//    }
    public int minJumps(int[] arr) {
        int len = arr.length;
        Queue<Integer> queue = new LinkedList<>();
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int i = 0; i < len; i++) {
            map.computeIfAbsent(arr[i], k -> new ArrayList<>()).add(i);
        }
        boolean[] vis = new boolean[len];
        queue.offer(0);
        int step = 0;
        vis[0] = true;
        while (!queue.isEmpty()) {
            int size = queue.size();
            step++;
            for (int i = 0; i < size; i++) {
                int cur = queue.poll();
//                if (vis[cur]) continue;
//                vis[cur] = true;
                if (cur == len - 1) {
                    return step - 1;
                }
                if (cur > 0) {
                    if (vis[cur - 1]) continue;
                    vis[cur - 1] = true;
                    queue.offer(cur - 1);
                }
                if (cur < len - 1) {
                    if (vis[cur + 1]) continue;
                    vis[cur + 1] = true;
                    queue.offer(cur + 1);
                }
                List<Integer> curmap = map.get(arr[cur]);
                for (int j = 0; j < curmap.size(); j++) {
                    int same = curmap.get(j);
                    if (same != cur) {
                        if (vis[same]) continue;
                        vis[same] = true;
                        queue.offer(same);
                    }
                }
            }

        }
        return step - 1;
    }

    public int countDigitOne(int n) {
        int count = 0;
        //依次考虑个位、十位、百位...是 1
        //k = 1000, 对应于上边举的例子
        for (int k = 1; k <= n; k *= 10) {
            // xyzdabc
            int abc = n % k;
            int xyzd = n / k;
            int d = xyzd % 10;
            int xyz = xyzd / 10;
            count += xyz * k;
            if (d > 1) {
                count += k;
            }
            if (d == 1) {
                count += abc + 1;
            }
            //如果不加这句的话，虽然 k 一直乘以 10，但由于溢出的问题
            //k 本来要大于 n 的时候，却小于了 n 会再次进入循环
            //此时代表最高位是 1 的情况也考虑完成了
            if (xyz == 0) {
                break;
            }
        }
        return count;
    }

    public int numberOf2sInRange(int n) {
        int count = 0;
        //依次考虑个位、十位、百位...是 1
        //k = 1000, 对应于上边举的例子
        for (int k = 1; k <= n; k *= 10) {
            //xyzdlow
            int low = n % k;
            int xyzd = n / k;
            int cur = xyzd % 10;
            int xyz = xyzd / 10;
            if (cur < 2) {
                count += xyz * k;
            } else if (cur == 2) {
                count += xyz * k + low + 1;
            } else {
                count += xyz * k + k;
            }
            if (xyz == 0) break;
        }
        return count;
    }
//    public int countDigitOne(int n) {
//        int sum = 0;
//        int len =0,tmp = n;
//        while(tmp>0){
//            len++;
//            tmp/=10;
//        }
//        for (int i = 0; i < len; i++) {
//
//            int pre = n;
//            int preop = i+1;
//            while(preop>0){
//                pre/=10;
//                preop--;
//            }
//            int behindbit = i;
//            int behindhundred = i*10;
//            int cur = (n%behindhundred)&1;
//            int behind =n%behindhundred;
//            if(behind==0){
//
//            }
//
//        }
//        return sum;
//    }

    public static void main(String[] args) {
        Lc15 lc15 = new Lc15();
        int[] s1 = {-2, 1, -1, -2, -2};
//        boolean r1 = lc15.circularArrayLoop(s1);
//        System.out.println(r1);
        int[][] s2 = {{9, 9, 4}, {6, 6, 8}, {2, 1, 1}};
        int[][] s3 = {{3, 4, 5}, {3, 2, 6}, {2, 2, 1}};
        int[][] s4 =
                {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {19, 18, 17, 16, 15, 14, 13, 12, 11, 10}, {20, 21, 22, 23, 24, 25, 26, 27, 28, 29}, {39, 38, 37, 36, 35, 34, 33, 32, 31, 30}, {40, 41, 42, 43, 44, 45, 46, 47, 48, 49}, {59, 58, 57, 56, 55, 54, 53, 52, 51, 50}, {60, 61, 62, 63, 64, 65, 66, 67, 68, 69}, {79, 78, 77, 76, 75, 74, 73, 72, 71, 70}, {80, 81, 82, 83, 84, 85, 86, 87, 88, 89}, {99, 98, 97, 96, 95, 94, 93, 92, 91, 90}, {100, 101, 102, 103, 104, 105, 106, 107, 108, 109}, {119, 118, 117, 116, 115, 114, 113, 112, 111, 110}, {120, 121, 122, 123, 124, 125, 126, 127, 128, 129}, {139, 138, 137, 136, 135, 134, 133, 132, 131, 130}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
//        int r2 = lc15.longestIncreasingPath(s2);
//        System.out.println(r2);
        char[][] s5 = {{'.', '.', '.', '.', '.', '.', '.', '.'}, {'.', 'B', '.', '.', 'W', '.', '.', '.'}, {'.', '.', 'W', '.', '.', '.', '.', '.'}, {'.', '.', '.', 'W', 'B', '.', '.', '.'}, {'.', '.', '.', '.', '.', '.', '.', '.'}, {'.', '.', '.', '.', 'B', 'W', '.', '.'}, {'.', '.', '.', '.', '.', '.', 'W', '.'}, {'.', '.', '.', '.', '.', '.', '.', 'B'}};
        char[][] s6 = {{'B', 'B', 'B', '.', 'W', 'W', 'B', 'W'}, {'B', 'B', '.', 'B', '.', 'B', 'B', 'B'}, {'.', 'W', '.', '.', 'B', '.', 'B', 'W'}, {'B', 'W', '.', 'W', 'B', '.', 'B', '.'}, {'B', 'W', 'W', 'B', 'W', '.', 'B', 'B'}, {'.', '.', 'W', '.', '.', 'W', '.', '.'}, {'W', '.', 'W', 'B', '.', 'W', 'W', 'B'}, {'B', 'B', 'W', 'W', 'B', 'W', '.', '.'}};

//        boolean r5 = lc15.checkMove(s5, 4, 4, 'W');
//        boolean r5 = lc15.checkMove(s6, 5, 6, 'B');
//        System.out.println(r5);
        int[] s7 = {10, 20, 30};
//        int r7 = lc15.minSpaceWastedKResizing(s7, 1);
//        System.out.println(r7);
        int[] s8 = {10, 20, 30};
//        int r8 = lc15.minSpaceWastedKResizing(s8, 1);
//        System.out.println(r8);

        int[] s9 = {2, 4, 6, 8, 10};
        int[] s10 = {7, 7, 7, 7, 7};
//        int r9 = lc15.numberOfArithmeticSlices(s10);
//        System.out.println(r9);

//        long r10 =lc15.maxProduct("rofcjxfkbzcvvlbkgcwtcjctwcgkblvvczbkfxjcfor");
//        System.out.println(r10);
        int[] s11 = {-1, -1, 1, 0, 0, 1, 0, -1};
        int[][] s12 = {{}, {6}, {5}, {6}, {3, 6}, {}, {}, {}};
        List<List<Integer>> beforeItems = new ArrayList<>();
        for (int[] each : s12) {
            beforeItems.add(Arrays.stream(each).boxed().collect(Collectors.toList()));
        }
//        lc15.sortItems(8, 2, s11, beforeItems);

//        int r12 = lc15.longestPalindromeSubseq("bbbab");
//        System.out.println(r12);

        int[] s13 = {3, 4, 16, 8};
        int[] s14 = {4, 8, 10, 240};
        int[] s15 = {5, 9, 18, 54, 108, 540, 90, 180, 360, 720};
//        lc15.largestDivisibleSubset(s15);

        int[] s16 = {4, 2, 3, 0, 3, 1, 2};
//        lc15.canReach(s16, 5);

        int[] s17 = {6, 4, 14, 6, 8, 13, 9, 7, 10, 6, 12};
//        lc15.maxJumps(s17, 2);

//        lc15.countDigitOne(13);

        int[] s18 = {100, -23, -23, 404, 100, 23, 23, 23, 3, 404};
        int[] s19 = {7, 6, 9, 6, 9, 6, 9, 7};
        int[] s20 = {6,1,9};
        lc15.minJumps(s19);

//        Twitter twitter = new Twitter();
//        twitter.postTweet(1, 5); // User 1 posts a new tweet (id = 5).
//        twitter.getNewsFeed(1);  // User 1's news feed should return a list with 1 tweet id -> [5]. return [5]
//        twitter.follow(1, 2);    // User 1 follows user 2.
//        twitter.postTweet(2, 6); // User 2 posts a new tweet (id = 6).
//        twitter.getNewsFeed(1);  // User 1's news feed should return a list with 2 tweet ids -> [6, 5]. Tweet id 6 should precede tweet id 5 because it is posted after tweet id 5.
//        twitter.unfollow(1, 2);  // User 1 unfollows user 2.
//        twitter.getNewsFeed(1);  // User 1's news feed should return a list with 1 tweet id -> [5], since user 1 is no longer following user 2.

    }
}


class Twitter {
    class Tweet {
        int tweetId;
        int timestamp;
        int userId;

        public Tweet(int tweetId, int timestamp, int userId) {
            this.tweetId = tweetId;
            this.timestamp = timestamp;
            this.userId = userId;
        }
    }

    //key: follwee  val: the follwer set
    Map<Integer, Set<Integer>> followers = new HashMap<>();
    // key user
    Map<Integer, List<Tweet>> tweets = new HashMap<>();
    // key user, its msg queue
//    Map<Integer, PriorityQueue<Tweet>> newsFeed = new HashMap<>();
    int time = 0;

    public Twitter() {

    }

    public void postTweet(int userId, int tweetId) {
        Set<Integer> followers = this.followers.computeIfAbsent(userId, k -> new HashSet<>());
        followers.add(userId);
        Tweet tweet = new Tweet(tweetId, time++, userId);
        tweets.computeIfAbsent(userId, k -> new ArrayList<>()).add(tweet);
//        newsFeed.computeIfAbsent(userId, k -> new LinkedList<>()).add(tweet);

    }

    public void follow(int followerId, int followeeId) {
        Set<Integer> followers = this.followers.get(followerId);
        if (followers == null) {
            this.followers.computeIfAbsent(followerId, k -> new HashSet<>());
            followers = this.followers.get(followerId);
        }
        followers.add(followeeId);

    }

    public void unfollow(int followerId, int followeeId) {
        Set<Integer> followers = this.followers.get(followerId);
        followers.remove(followeeId);
    }

    // the user and its follower should both enter the queue todo!
    public List<Integer> getNewsFeed(int userId) {
        List<Integer> res = new ArrayList<>();
        PriorityQueue<Tweet> queue = new PriorityQueue<>((a, b) -> (b.timestamp - a.timestamp));
        Set<Integer> set = followers.get(userId);
        int ten = 10;
        if (set == null) return res;
        for (int each : set) {
            List<Tweet> tmp = tweets.get(each);
            if (tmp == null) continue;
            queue.addAll(tmp);
        }
        while (ten > 0 && !queue.isEmpty()) {
            Tweet tmp = queue.poll();
            res.add(tmp.tweetId);
            ten--;
        }
        return res;
    }

}


// class Twitter {
//
//    /**
//     * 用户 id 和推文（单链表）的对应关系
//     */
//    private Map<Integer, Tweet> twitter;
//
//    /**
//     * 用户 id 和他关注的用户列表的对应关系
//     */
//    private Map<Integer, Set<Integer>> followings;
//
//    /**
//     * 全局使用的时间戳字段，用户每发布一条推文之前 + 1
//     */
//    private static int timestamp = 0;
//
//    /**
//     * 合并 k 组推文使用的数据结构（可以在方法里创建使用），声明成全局变量非必需，视个人情况使用
//     */
//    private static PriorityQueue<Tweet> maxHeap;
//
//    /**
//     * Initialize your data structure here.
//     */
//    public Twitter() {
//        followings = new HashMap<>();
//        twitter = new HashMap<>();
//        maxHeap = new PriorityQueue<>((o1, o2) -> -o1.timestamp + o2.timestamp);
//    }
//
//    /**
//     * Compose a new tweet.
//     */
//    public void postTweet(int userId, int tweetId) {
//        timestamp++;
//        if (twitter.containsKey(userId)) {
//            Tweet oldHead = twitter.get(userId);
//            Tweet newHead = new Tweet(tweetId, timestamp);
//            newHead.next = oldHead;
//            twitter.put(userId, newHead);
//        } else {
//            twitter.put(userId, new Tweet(tweetId, timestamp));
//        }
//    }
//
//    /**
//     * Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent.
//     */
//    public List<Integer> getNewsFeed(int userId) {
//        // 由于是全局使用的，使用之前需要清空
//        maxHeap.clear();
//
//        // 如果自己发了推文也要算上
//        if (twitter.containsKey(userId)) {
//            maxHeap.offer(twitter.get(userId));
//        }
//
//        Set<Integer> followingList = followings.get(userId);
//        if (followingList != null && followingList.size() > 0) {
//            for (Integer followingId : followingList) {
//                Tweet tweet = twitter.get(followingId);
//                if (tweet != null) {
//                    maxHeap.offer(tweet);
//                }
//            }
//        }
//
//        List<Integer> res = new ArrayList<>(10);
//        int count = 0;
//        while (!maxHeap.isEmpty() && count < 10) {
//            Tweet head = maxHeap.poll();
//            res.add(head.id);
//
//            // 这里最好的操作应该是 replace，但是 Java 没有提供
//            if (head.next != null) {
//                maxHeap.offer(head.next);
//            }
//            count++;
//        }
//        return res;
//    }
//
//
//    /**
//     * Follower follows a followee. If the operation is invalid, it should be a no-op.
//     *
//     * @param followerId 发起关注者 id
//     * @param followeeId 被关注者 id
//     */
//    public void follow(int followerId, int followeeId) {
//        // 被关注人不能是自己
//        if (followeeId == followerId) {
//            return;
//        }
//
//        // 获取我自己的关注列表
//        Set<Integer> followingList = followings.get(followerId);
//        if (followingList == null) {
//            Set<Integer> init = new HashSet<>();
//            init.add(followeeId);
//            followings.put(followerId, init);
//        } else {
//            if (followingList.contains(followeeId)) {
//                return;
//            }
//            followingList.add(followeeId);
//        }
//    }
//
//
//    /**
//     * Follower unfollows a followee. If the operation is invalid, it should be a no-op.
//     *
//     * @param followerId 发起取消关注的人的 id
//     * @param followeeId 被取消关注的人的 id
//     */
//    public void unfollow(int followerId, int followeeId) {
//        if (followeeId == followerId) {
//            return;
//        }
//
//        // 获取我自己的关注列表
//        Set<Integer> followingList = followings.get(followerId);
//
//        if (followingList == null) {
//            return;
//        }
//        // 这里删除之前无需做判断，因为查找是否存在以后，就可以删除，反正删除之前都要查找
//        followingList.remove(followeeId);
//    }
//
//    /**
//     * 推文类，是一个单链表（结点视角）
//     */
//    private class Tweet {
//        /**
//         * 推文 id
//         */
//        private int id;
//
//        /**
//         * 发推文的时间戳
//         */
//        private int timestamp;
//        private Tweet next;
//
//        public Tweet(int id, int timestamp) {
//            this.id = id;
//            this.timestamp = timestamp;
//        }
//    }

// public static void main(String[] args) {

//     Twitter twitter = new Twitter();
//     twitter.postTweet(1, 1);
//     List<Integer> res1 = twitter.getNewsFeed(1);
//     System.out.println(res1);

//     twitter.follow(2, 1);

//     List<Integer> res2 = twitter.getNewsFeed(2);
//     System.out.println(res2);

//     twitter.unfollow(2, 1);

//     List<Integer> res3 = twitter.getNewsFeed(2);
//     System.out.println(res3);
// }
//}


//public class LRUCache {
//    class DLinkedNode {
//        int key;
//        int value;
//        DLinkedNode prev;
//        DLinkedNode next;
//
//        public DLinkedNode() {
//        }
//
//        public DLinkedNode(int _key, int _value) {
//            key = _key;
//            value = _value;
//        }
//    }
//
//    private Map<Integer, DLinkedNode> cache = new HashMap<Integer, DLinkedNode>();
//    private int size;
//    private int capacity;
//    private DLinkedNode head, tail;
//
//    public LRUCache(int capacity) {
//        this.size = 0;
//        this.capacity = capacity;
//        // 使用伪头部和伪尾部节点
//        head = new DLinkedNode();
//        tail = new DLinkedNode();
//        head.next = tail;
//        tail.prev = head;
//    }
//
//    public int get(int key) {
//        DLinkedNode node = cache.get(key);
//        if (node == null) {
//            return -1;
//        }
//        // 如果 key 存在，先通过哈希表定位，再移到头部
//        moveToHead(node);
//        return node.value;
//    }
//
//    public void put(int key, int value) {
//        DLinkedNode node = cache.get(key);
//        if (node == null) {
//            // 如果 key 不存在，创建一个新的节点
//            DLinkedNode newNode = new DLinkedNode(key, value);
//            // 添加进哈希表
//            cache.put(key, newNode);
//            // 添加至双向链表的头部
//            addToHead(newNode);
//            ++size;
//            if (size > capacity) {
//                // 如果超出容量，删除双向链表的尾部节点
//                DLinkedNode tail = removeTail();
//                // 删除哈希表中对应的项
//                cache.remove(tail.key);
//                --size;
//            }
//        } else {
//            // 如果 key 存在，先通过哈希表定位，再修改 value，并移到头部
//            node.value = value;
//            moveToHead(node);
//        }
//    }
//
//    private void addToHead(DLinkedNode node) {
//        node.prev = head;
//        node.next = head.next;
//        head.next.prev = node;
//        head.next = node;
//    }
//
//    private void removeNode(DLinkedNode node) {
//        node.prev.next = node.next;
//        node.next.prev = node.prev;
//    }
//
//    private void moveToHead(DLinkedNode node) {
//        removeNode(node);
//        addToHead(node);
//    }
//
//    private DLinkedNode removeTail() {
//        DLinkedNode res = tail.prev;
//        removeNode(res);
//        return res;
//    }
//}

class LRUCache {
    //容量
    private int capacity;
    //缓存内容
    HashMap<Integer, Integer> map;
    //队列+延迟删除
    Queue<Integer> Q;
    HashMap<Integer, Integer> delay;

    public LRUCache(int capacity) {
        this.capacity = capacity;
        map = new HashMap<>();
        Q = new LinkedList<>();
        delay = new HashMap<>();
    }

    public int get(int key) {
        if (map.containsKey(key)) {
            deleteDelay();
            Q.add(key);
            delay.put(key, delay.getOrDefault(key, 0) + 1);
            return map.get(key);
        } else {
            return -1;
        }
    }
// queue we need offer instead of adding? todo

    public void put(int key, int value) {
        if (map.containsKey(key)) {
            deleteDelay();
            Q.add(key);
            delay.put(key, delay.getOrDefault(key, 0) + 1);
            map.put(key, value);
        } else {
            if (map.size() >= capacity) {
                deleteDelay();
                map.remove(Q.poll());
            }
            Q.add(key);
            map.put(key, value);
        }
    }

    private void deleteDelay() {
        while (delay.containsKey(Q.peek())) {
            int peek = Q.peek();
            delay.put(peek, delay.get(peek) - 1);
            if (delay.get(peek) == 0) {
                delay.remove(peek);
            }
            Q.poll();
        }
    }
}

//class LRUCache {
//    //容量
//    private int capacity;
//    //缓存内容
//    HashMap<Integer, Integer> map;
//    //队列+延迟删除
//    Queue<Integer> Q;
//    HashMap<Integer, Integer> delay;
//
//    public LRUCache(int capacity) {
//        this.capacity = capacity;
//        map = new HashMap<>();
//        Q = new LinkedList<>();
//        delay = new HashMap<>();
//    }
//
//    public int get(int key) {
//        if (map.containsKey(key)) {
//            Q.add(key);
//            delay.put(key, delay.getOrDefault(key, 0) + 1);
//            return map.get(key);
//        } else {
//            return -1;
//        }
//    }
//
//    public void put(int key, int value) {
//        if (map.containsKey(key)) {
//            map.put(key, value);
//            Q.add(key);
//            delay.put(key, delay.getOrDefault(key, 0) + 1);
//        } else {
//            if (map.size() < capacity) {
//                map.put(key, value);
//                Q.add(key);
//            } else {
//                delayDelete();
//                map.remove(Q.poll());
//                map.put(key, value);
//                Q.add(key);
//            }
//        }
//    }
//
//    void delayDelete() {
//        //延迟删除操作
//        while (delay.containsKey(Q.peek())) {
//            delay.put(Q.peek(), delay.get(Q.peek()) - 1);
//            if (delay.get(Q.peek()) == 0)
//                delay.remove(Q.peek());
//            Q.poll();
//        }
//    }
//}


//class Solution {
//    public int[][] dirs = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
//    public int rows, columns;
//
//    public int longestIncreasingPath(int[][] matrix) {
//        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
//            return 0;
//        }
//        rows = matrix.length;
//        columns = matrix[0].length;
//        int[][] memo = new int[rows][columns];
//        int ans = 0;
//        for (int i = 0; i < rows; ++i) {
//            for (int j = 0; j < columns; ++j) {
//                ans = Math.max(ans, dfs(matrix, i, j, memo));
//            }
//        }
//        return ans;
//    }
//
//    public int dfs(int[][] matrix, int row, int column, int[][] memo) {
//        if (memo[row][column] != 0) {
//            return memo[row][column];
//        }
//        ++memo[row][column];
//        for (int[] dir : dirs) {
//            int newRow = row + dir[0], newColumn = column + dir[1];
//            if (newRow >= 0 && newRow < rows && newColumn >= 0 && newColumn < columns && matrix[newRow][newColumn] > matrix[row][column]) {
//                memo[row][column] = Math.max(memo[row][column], dfs(matrix, newRow, newColumn, memo) + 1);
//            }
//        }
//        return memo[row][column];
//    }
//}


//class Solution {
//    public int[][] dirs = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
//    public int rows, columns;
//
//    public int longestIncreasingPath(int[][] matrix) {
//        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
//            return 0;
//        }
//        rows = matrix.length;
//        columns = matrix[0].length;
//        int[][] outdegrees = new int[rows][columns];
//        for (int i = 0; i < rows; ++i) {
//            for (int j = 0; j < columns; ++j) {
//                for (int[] dir : dirs) {
//                    int newRow = i + dir[0], newColumn = j + dir[1];
//                    if (newRow >= 0 && newRow < rows && newColumn >= 0 && newColumn < columns && matrix[newRow][newColumn] > matrix[i][j]) {
//                        ++outdegrees[i][j];
//                    }
//                }
//            }
//        }
//        Queue<int[]> queue = new LinkedList<int[]>();
//        for (int i = 0; i < rows; ++i) {
//            for (int j = 0; j < columns; ++j) {
//                if (outdegrees[i][j] == 0) {
//                    queue.offer(new int[]{i, j});
//                }
//            }
//        }
//        int ans = 0;
//        while (!queue.isEmpty()) {
//            ++ans;
//            int size = queue.size();
//            for (int i = 0; i < size; ++i) {
//                int[] cell = queue.poll();
//                int row = cell[0], column = cell[1];
//                for (int[] dir : dirs) {
//                    int newRow = row + dir[0], newColumn = column + dir[1];
//                    if (newRow >= 0 && newRow < rows && newColumn >= 0 && newColumn < columns && matrix[newRow][newColumn] < matrix[row][column]) {
//                        --outdegrees[newRow][newColumn];
//                        if (outdegrees[newRow][newColumn] == 0) {
//                            queue.offer(new int[]{newRow, newColumn});
//                        }
//                    }
//                }
//            }
//        }
//        return ans;
//    }
//}
