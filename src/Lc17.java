import javax.swing.*;
import java.util.*;
import java.util.stream.Collectors;

public class Lc17 {
    //    public int maxScore(int[] nums) {
//        int N = nums.length;//数组长度 以[3, 4, 6, 8]为例
//        int[] dp = new int[1 << N];//dp[st] 表示st这个状态下，N次操作后的最大分数和后
//        for (int i = 0; i < 1 << N; i++) {//枚举状态 从 0000 0001 ... 1111[表示全部数都在的]
//            int cnt = count(i);//计算当前i中1的个数，1表示这个索引下这个数存在
////                System.out.printf("i:%d,bin:%s,cnt:%d\n", i, PrintUtils.toBinaryString(i, 4), cnt);
//            if ((cnt & 1) == 1) continue;//奇数跳过 当1的个数是奇数个时，需要跳过，只有偶数个数才能做gcd
//            for (int j = 0; j < N; j++) {//第1个数
//                for (int k = j + 1; k < N; k++) {//当前数后面开始枚举第2个数
//                    //获取这个两个数组成的10进制的数，如j =0 , k =1
//                    //1<<j = 0001 1<<1 = 0010
//                    //0001 | 0010 = 0011 也就是十进制的3
//                    int st = (1 << j) | (1 << k);
////                        System.out.printf("  j:%d,k:%d,st:%d,st_bin:%s,if:%s\n", j, k, st, PrintUtils.toBinaryString(st, 4), (st & i) == st);
//                    if ((st & i) == st) {//i这个状态是否包含st这个状态，包含才有意义
//                        dp[i] = Math.max(dp[i], dp[i - st] + gcd(nums[j], nums[k]) * cnt / 2);//cnt是当前1的个数 取这个状态转移来的之前的状态 i-st这个状态
//                    }
//                }
//            }
//        }
//        return dp[(1 << N) - 1];//- 优先于 <<  相当于取二进制位上各位都为1的结果 恰好是整个数组
//
//    }
//    //计算i的1的个数
//    public int count(int i) {
//        int ans = 0;
//        while (i != 0) {
//            ans += i & 1;
//            i >>>= 1;
//        }
//        return ans;
//    }
//
//    //gcd
//    public int gcd(int a, int b) {
////            System.out.printf("a:%d,b:%d\n", a, b);
//        return b == 0 ? a : gcd(b, a % b);
//    }
    public int maxScore(int[] nums) {
        int len = nums.length;
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < len; i++) {
            for (int j = i + 1; j < len; j++) {
                int key = (1 << i) | (1 << j);
                map.put(key, gcd(nums[i], nums[j]));
            }
        }
        int[] dp = new int[(1 << len)];
        for (int i = 0; i < (1 << len); i++) {
            int bits = Integer.bitCount(i);
            if ((bits & 1) == 1) continue;
            for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
                if ((entry.getKey() & i) == 0) {
                    dp[(i | entry.getKey())] = Math.max(dp[(i | entry.getKey())], dp[i] + entry.getValue() * (bits / 2 + 1));
                }
            }
        }
        return dp[(1 << len) - 1];
    }

    public int maxScore2(int[] nums) {
        int n = nums.length;
        Map<Integer, Integer> gcdVal = new HashMap<>();
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                gcdVal.put((1 << i) + (1 << j), gcd(nums[i], nums[j]));
            }
        }

        int[] dp = new int[1 << n];

        for (int i = 0; i < (1 << n); ++i) {
            int bits = Integer.bitCount(i); // how many numbers are used
            if (bits % 2 != 0) // odd numbers, skip it
                continue;
            for (int k : gcdVal.keySet()) {
                if ((k & i) != 0) // overlapping used numbers
                    continue;
                dp[i ^ k] = Math.max(dp[i ^ k], dp[i] + gcdVal.get(k) * (bits / 2 + 1));
            }
        }

        return dp[(1 << n) - 1];
    }

    public int gcd(int a, int b) {
        if (b == 0)
            return a;
        return gcd(b, a % b);
    }

    public int unhappyFriends(int n, int[][] preferences, int[][] pairs) {
        int count = 0;
        List<List<Integer>> pre = new ArrayList<>();
        for (int i = 0; i < preferences.length; i++) {
            List<Integer> tmp = Arrays.stream(preferences[i]).boxed().collect(Collectors.toList());
            pre.add(tmp);
        }
        Map<Integer, Integer> map = new HashMap<>();
        for (int[] pair : pairs) {
            map.put(pair[0], pair[1]);
            map.put(pair[1], pair[0]);
        }
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            if (isUnHappy(entry, pre, map)) {
                count++;
            }
        }
        return count;
    }

    private boolean isUnHappy(Map.Entry<Integer, Integer> entry, List<List<Integer>> pre, Map<Integer, Integer> map) {
        List<Integer> keyList = pre.get(entry.getKey());
        int pairId = keyList.indexOf(entry.getValue());
        for (int i = 0; i < pairId; i++) {
            int u = keyList.get(i);
            List<Integer> uList = pre.get(u);
            if (uList.indexOf(map.get(u)) > uList.indexOf(entry.getKey())) {
                return true;
            }
        }
        return false;
    }

    //str1 is short
    public String gcdOfStrings(String str1, String str2) {
        if (str1.length() > str2.length()) {
            return gcdOfStrings(str2, str1);
        }
        int id = 0;
        StringBuilder res = new StringBuilder();
        for (int i = 0; i < str1.length(); i++) {
            StringBuilder sb = new StringBuilder(str1.substring(0, i + 1));
            if (str1.length() % sb.length() == 0 && str2.length() % sb.length() == 0) {
                StringBuilder sb1 = new StringBuilder(sb);
                int left1 = (str1.length() - sb.length()) / (sb.length());
                for (int j = 0; j < left1; j++) {
                    sb1.append(sb);
                }
                if (!str1.equals(sb1.toString())) {
                    continue;
                }
                StringBuilder sb2 = new StringBuilder(sb);
                int left2 = (str2.length() - sb.length()) / (sb.length());
                for (int j = 0; j < left2; j++) {
                    sb2.append(sb);
                }
                if (!str2.equals(sb2.toString())) {
                    continue;
                }
                res = res.length() < sb.length() ? sb : res;
            }
        }
        return res.toString();
    }

    public int findMaxForm(String[] strs, int m, int n) {
        int len = strs.length;
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 0; i < strs.length; i++) {
            int[] count = countOne(strs[i]);
            int zero = count[0];
            int one = count[1];
            for (int j = m; j >= zero; j--) {
                for (int k = n; k >= one; k--) {
                    dp[j][k] = Math.max(dp[j][k], dp[j - zero][k - one] + 1);
                }
            }
        }
        return dp[m][n];
    }

    private int[] countOne(String str) {
        int one = 0, zero = 0;
        for (int i = 0; i < str.length(); i++) {
            if (str.charAt(i) == '0') zero++;
            else one++;
        }
        return new int[]{zero, one};
    }

    public int minSteps(int n) {
        if (n == 1) return 0;
        int[][] dp = new int[n + 1][n + 1];
        int INF = 0x3f3f3f3f;
        for (int i = 0; i < n + 1; i++) {
            Arrays.fill(dp[i], INF);
        }
        dp[1][1] = 1;
        for (int i = 2; i <= n; i++) {
            for (int j = 1; j < i; j++) {
                dp[i][j] = Math.min(dp[i][j], dp[i - j][j] + 1);
            }
            dp[i][i] = Arrays.stream(dp[i]).min().getAsInt() + 1;
        }
        return Arrays.stream(dp[n]).min().getAsInt();
    }

    public int trap(int[] height) {
        int len = height.length;
        Stack<Integer> stack = new Stack<>();
        int res = 0, cur = 0;
        stack.push(0);
        while (cur < len) {
            while (stack.peek() != 0 && height[cur] >= height[stack.peek()]) {
                int pos = stack.pop();
                int incre = Math.min(height[stack.peek()], height[cur]) - height[pos];
                if (incre > 0)
                    res += incre * (cur - stack.peek() - 1);
            }
            stack.push(cur++);
        }
        return res;
    }


    public int numOfStrings(String[] patterns, String word) {
        int count = 0;
        for (int i = 0; i < patterns.length; i++) {
            if (word.contains(patterns[i])) {
                count++;
            }
        }
        return count;
    }

    public int[] rearrangeArray(int[] nums) {
        int len = nums.length;
        Arrays.sort(nums);
//        int[] res = new int[len];
        int left = 0, right = len - 1;

        while (left < right && left < len) {
            swap(nums, left, right);
            left += 2;
            right -= 2;
        }
        return nums;
    }

    private void swap(int[] nums, int left, int right) {
        int tmp = nums[left];
        nums[left] = nums[right];
        nums[right] = tmp;
    }

    public int[] rearrangeArray2(int[] nums) {
        int len = nums.length;
        boolean[] vis = new boolean[len];
        int[] res = new int[len];
        Arrays.sort(nums);
        dfs(res, vis, nums, 0);
        return res;
    }

    private boolean dfs(int[] res, boolean[] vis, int[] nums, int pos) {
        if (pos > 2) {
            if (res[pos - 3] + res[pos - 1] == res[pos - 2] * 2) {
                return false;
            }
        }
        if (pos == nums.length) {
            return true;
        }
        for (int i = 0; i < nums.length; i++) {
            if (vis[i]) continue;
            vis[i] = true;
            res[pos] = nums[i];
            if (dfs(res, vis, nums, pos + 1)) {
                return true;
            }
            vis[i] = false;
        }
        return false;
    }

//    public int maxStudents(char[][] seats) {
//        final int n = seats.length, m = seats[0].length;
//
//        // （空间还可以优化）
//        int[][] dp = new int[n + 1][1 << m];
//        for (int i = 1; i <= n; i++) {
//            // 位1表示坏了
//            int invalid = 0;
//            for (int j = 0; j < m; j++) {
//                if (seats[i - 1][j] == '#') {
//                    invalid |= 1 << j;
//                }
//            }
//            for (int j = 0; j < (1 << m); j++) {
//                // 来判断相邻位置
//                int adjacentMask = j << 1;
//                // 坐在坏椅子上或相邻座位已坐，舍弃该状态
//                if ((j & invalid) != 0 || (j & adjacentMask) != 0) {
//                    dp[i][j] = -1;
//                    continue;
//                }
//
//                int theOtherAdjacentMask = j >>> 1;
//                // 如果状态可行，遍历上一行的所有状态，寻找状态最大值
//                for (int s = 0; s < (1 << m); s++) {
//                    // 如果 s 不合法，舍弃状态 s
//                    if (dp[i - 1][s] == -1) {
//                        continue;
//                    }
//                    // 如果相邻列已坐，舍弃状态 s
//                    if ((s & adjacentMask) != 0 || (s & theOtherAdjacentMask) != 0) {
//                        continue;
//                    }
//                    // 状态转移
//                    dp[i][j] = Math.max(dp[i][j], dp[i - 1][s] + Integer.bitCount(j));
//                }
//            }
//        }
//
//        return Arrays.stream(dp[n]).max().getAsInt();
//    }

    public int maxStudents(char[][] seats) {
        int row = seats.length, col = seats[0].length;
        int[][] dp = new int[row + 1][1 << col];
        for (int i = 1; i <= row; i++) {
            int valid = 0;
            for (int j = 0; j < col; j++) {
                if (seats[i - 1][j] == '#') {
                    valid |= (1 << j);
                }
            }
            for (int j = 0; j < (1 << col); j++) {
                if ((j & valid) != 0) {
                    dp[i][j] = -1;
                    continue;
                }
                int shiftleft = j << 1;
                if ((shiftleft & j) != 0) continue;
                int shiftright = j >>> 1;
                for (int k = 0; k < (1 << col); k++) {
                    if (dp[i - 1][k] == -1) continue;
                    if ((k & shiftright) == 0 && (k & shiftleft) == 0) {
                        dp[i][j] = Math.max(dp[i][j], dp[i - 1][k] + Integer.bitCount(j));
                    }
                }
            }
        }
        return Arrays.stream(dp[row]).max().getAsInt();
    }

    int res1 = 0;
    int MOD = (int) 1e9 + 7;

    public int findPaths3(int m, int n, int maxMove, int startRow, int startColumn) {
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i++) {
            Arrays.fill(dp[i], -1);
        }
        dfs2(m, n, 1, maxMove, startRow, startColumn, dp);
        return res1;
    }

    private void dfs2(int m, int n, int curMove, int maxMove, int startRow, int startColumn, int[][] dp) {
        if (curMove > maxMove) {
            System.out.println("done");
            return;
        }
        if (dp[startRow][startColumn] == -1) {
            dp[startRow][startColumn] = countExits(m, n, startRow, startColumn);
        }
        System.out.println(startRow + ":" + startColumn);
        res1 += dp[startRow][startColumn];
        res1 %= MOD;
        for (int[] each : directions) {
            int nextx = startRow + each[0];
            int nexty = startColumn + each[1];
            if (nextx < 0 || nextx >= m || nexty < 0 || nexty >= n) continue;
            dfs2(m, n, curMove + 1, maxMove, nextx, nexty, dp);
        }
    }

    int[][] directions = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    public int countExits(int m, int n, int x, int y) {
        int count = 0;
        for (int[] each : directions) {
            int nextx = x + each[0];
            int nexty = y + each[1];
            if (nextx < 0 || nextx >= m || nexty < 0 || nexty >= n) {
                count++;
            }
        }
        return count;
    }

    int m, n, max;
    int[][] dirs = new int[][]{{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    public int findPaths(int _m, int _n, int _max, int r, int c) {
        m = _m;
        n = _n;
        max = _max;
        int[][] f = new int[m * n][max + 1];
        // 初始化边缘格子的路径数量
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i == 0) add(i, j, f);
                if (j == 0) add(i, j, f);
                if (i == m - 1) add(i, j, f);
                if (j == n - 1) add(i, j, f);
            }
        }
        // 从小到大枚举「可移动步数」
        for (int k = 1; k <= max; k++) {
            // 枚举所有的「位置」
            for (int idx = 0; idx < m * n; idx++) {
                int[] info = parseIdx(idx);
                int x = info[0], y = info[1];
                for (int[] d : dirs) {
                    int nx = x + d[0], ny = y + d[1];
                    if (nx < 0 || nx >= m || ny < 0 || ny >= n) continue;
                    int nidx = getIdx(nx, ny);
                    f[idx][k] += f[nidx][k - 1];
                    f[idx][k] %= MOD;
                }
            }
        }
        return f[getIdx(r, c)][max];
    }

    void add(int x, int y, int[][] f) {
        for (int k = 1; k <= max; k++) {
            f[getIdx(x, y)][k]++;
        }
    }

    int getIdx(int x, int y) {
        return x * n + y;
    }

    int[] parseIdx(int idx) {
        return new int[]{idx / n, idx % n};
    }


    public int maximumUnits(int[][] boxTypes, int truckSize) {
        int[] dp = new int[truckSize + 1];
        int count = 0;
        for (int[] each : boxTypes) {
            int boxNumber = each[0];
            int boxSize = each[1];
            count += boxNumber;
            for (int i = Math.min(count, truckSize); i > 0; i--) {

                for (int j = 1; j <= boxNumber; j++) {
                    if (i - j < 0) break;
                    dp[i] = Math.max(dp[i], dp[i - j] + boxSize * j);
                }
            }
        }
//        System.out.println(Arrays.toString(dp));
//        return dp[truckSize];
        return Arrays.stream(dp).max().getAsInt();
    }

    public int maximumUnits2(int[][] boxTypes, int truckSize) {
        int[] dp = new int[truckSize + 1];
        int count = 0;
        for (int[] each : boxTypes) {
            int boxNumber = each[0];
            int boxSize = each[1];
            count += boxNumber;
            for (int i = truckSize; i > 0; i--) {

                for (int j = 1; j <= boxNumber; j++) {
                    if (i - j < 0) break;
                    dp[i] = Math.max(dp[i], dp[i - j] + boxSize * j);
                }
            }
        }
        System.out.println(Arrays.toString(dp));

        return dp[truckSize];
    }

    List<int[]> trio = new ArrayList<>();

    public int minTrioDegree(int n, int[][] edges) {
        Map<Integer, List<Integer>> map = new HashMap<>();
        boolean[] vis = new boolean[n];
        int INF = 0x3f3f3f3f;
        int min = INF;
        for (int[] each : edges) {
            map.computeIfAbsent(each[0], k -> new ArrayList<>()).add(each[1]);
            map.computeIfAbsent(each[1], k -> new ArrayList<>()).add(each[0]);
        }
        for (Map.Entry<Integer, List<Integer>> entry : map.entrySet()) {
            vis[entry.getKey()] = true;
            if (isTrio(entry, map)) {

            }
        }
        return min == INF ? -1 : min;
    }

    private boolean isTrio(Map.Entry<Integer, List<Integer>> entry, Map<Integer, List<Integer>> map) {
        List<Integer> list = entry.getValue();
        for (int i = 0; i < list.size(); i++) {
            int third = list.get(i);
            List<Integer> thirdlist = map.get(third);
            for (int j = 0; j < thirdlist.size(); j++) {
                if (list.contains(thirdlist.get(j))) {

                    return true;
                }
            }
        }
        return false;
    }

    public char slowestKey(int[] releaseTimes, String keysPressed) {
        int max = releaseTimes[0];
        char maxChar = keysPressed.charAt(0);
        for (int i = 1; i < keysPressed.length(); i++) {
            int during = releaseTimes[i] - releaseTimes[i - 1];
            if (during > max) {
                max = during;
                maxChar = keysPressed.charAt(i);
            }
            if (during == max && keysPressed.charAt(i) > maxChar) {
                maxChar = keysPressed.charAt(i);
            }

        }
        return maxChar;
    }

    public int findLeastNumOfUniqueInts(int[] arr, int k) {
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> {
            if (a[1] != b[1]) return a[1] - b[1];
            else return a[0] - b[0];
        });
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < arr.length; i++) {
            map.put(arr[i], map.getOrDefault(arr[i], 0) + 1);
        }
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            pq.offer(new int[]{entry.getKey(), entry.getValue()});
        }
        while (k > 0) {
            int[] cur = pq.poll();
            if (cur[1] > 1) {
                pq.offer(new int[]{cur[0], cur[1] - 1});
            }
            k--;
        }
        return pq.size();
    }

    //    int res2 = 0;
//    int n2 = 0;
//
//    public int countArrangement(int _n) {
//        n2 = _n;
//        boolean[] vis = new boolean[n2 + 1];
////        int[] res = new int[n2 + 1];
//        backtrace(vis, 1);
//        return res2;
//    }
//
//    private void backtrace(boolean[] vis, int pos) {
//        if (pos == vis.length) {
//            res2++;
//            return;
//        }
//        for (int i = 1; i <= n2; i++) {
//            if (vis[i]) continue;
//            if (i % pos == 0 || pos % i == 0) {
//                vis[i] = true;
//                backtrace(vis, pos + 1);
//                vis[i] = false;
//            }
//        }
//    }
    public int countArrangement(int n) {
        int[][] dp = new int[n + 1][1 << n];
        dp[0][0] = 1;
        for (int i = 1; i <= n; i++) {
            for (int j = 0; j < (1 << n); j++) {
                //choose the number to fill in the array
                for (int k = 1; k <= n; k++) {
                    if ((j >> (k - 1) & 1) == 0) continue;
                    if (i % k != 0 && k % i != 0) continue;
                    dp[i][j] += dp[i - 1][j & (~(1 << (k - 1)))];
                }
            }
        }
        return dp[n][(1 << n) - 1];
    }

    public ListNode reverseBetween(ListNode head, int left, int right) {
        if (left == right) return head;
        ListNode dum = new ListNode(), pre = dum, cur = head;
        dum.next = head;
        int curId = 1;
        Stack<ListNode> stack = new Stack<>();
        while (curId < left) {
            curId++;
            cur = cur.next;
            pre = pre.next;
        }
        while (curId <= right) {
            ListNode tmp = cur;
            stack.push(tmp);
            cur = cur.next;
            tmp.next = null;
            curId++;
        }
        ListNode end = cur;
        while (!stack.isEmpty()) {
            pre.next = stack.pop();
            pre = pre.next;
        }
        pre.next = end;
        return dum.next;
    }

    public int countSubstrings2(String s) {
        int count = s.length();
        char[] arr = s.toCharArray();
        for (int i = 0; i < arr.length; i++) {
            int tmp = 1;
            while (i - tmp >= 0 && i + tmp < arr.length && arr[i - tmp] == arr[i + tmp]) {
                count++;
                tmp++;
            }
            tmp = 0;
            while (i >= 1 && arr[i - 1 - tmp] == arr[i + tmp]) {
                count++;
                tmp++;
            }
        }
        return count;
    }

    public int countSubstrings(String s) {
        int len = s.length();
        char[] arr = s.toCharArray();
        boolean[][] dp = new boolean[len][len];
        int res = 0;
        for (int i = len - 1; i >= 0; i--) {
            for (int j = i; j < len; j++) {
                if (arr[i] == arr[j]) {
                    if ((i == j || i + 1 == j || dp[i + 1][j - 1])) {
                        res++;
                        dp[i][j] = true;
                    }
                }

            }
        }
        return res;
    }

    public static void main(String[] args) {
        Lc17 lc17 = new Lc17();
//        int[] s3 = {3, 4, 6, 8};
//        lc17.maxScore(s3);

//        int[][] s4 = {{1, 2, 3}, {3, 2, 0}, {3, 1, 0}, {1, 2, 0}};
//        int[][] s5 = {{0, 1}, {2, 3}};
//        int r4 = lc17.unhappyFriends(4, s4, s5);
//        System.out.println(r4);

//        lc17.gcdOfStrings("ABCABC", "ABC");

//        int r5=lc17.minSteps(3);
//        System.out.println(r5);

//        int[] s6 = {0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1};
//        int r6 = lc17.trap(s6);
//        System.out.println(r6);

//        int[] s7 = {1, 2, 3, 4, 5};
//        int[] s8 = {6, 2, 0, 9, 7};
//        int[] r7 = lc17.rearrangeArray(s8);
//        System.out.println(Arrays.toString(r7));

//        char[][] s9 = {{'#', '.', '#', '#', '.', '#'}, {'.', '#', '#', '#', '#', '.'}, {'#', '.', '#', '#', '.', '#'}};
//        lc17.maxStudents(s9);

//        int r10 =lc17.findPaths(2,2,2,0,0);
//        int r11 = lc17.findPaths(1, 3, 3, 0, 1);
////        int r12 = lc17.findPaths(8, 7, 16, 1, 5);
//        System.out.println(r11);

        int[][] s13 = {{1, 3}, {5, 5}, {2, 5}, {4, 2}, {4, 1}, {3, 1}, {2, 2}, {1, 3}, {2, 5}, {3, 2}};
//        lc17.maximumUnits(s13, 35);
//        lc17.maximumUnits2(s13, 35);

//        int r14 = lc17.countArrangement(2);
//        System.out.println(r14);

        ListNode s15 = new ListNode(1, new ListNode(2, new ListNode(3, new ListNode(4, new ListNode(5)))));
        lc17.reverseBetween(s15, 2, 4);

    }
}


//class Solution {
//    public int maxScore(int[] nums) {
//        int n = nums.length;
//        Map<Integer, Integer> gcdVal = new HashMap<>();
//        for (int i = 0; i < n; ++i) {
//            for (int j = i + 1; j < n; ++j) {
//                gcdVal.put((1 << i) + (1 << j), gcd(nums[i], nums[j]));
//            }
//        }
//
//        int[] dp = new int[1 << n];
//
//        for (int i = 0; i < (1 << n); ++i) {
//            int bits = Integer.bitCount(i); // how many numbers are used
//            if (bits % 2 != 0) // odd numbers, skip it
//                continue;
//            for (int k : gcdVal.keySet()) {
//                if ((k & i) != 0) // overlapping used numbers
//                    continue;
//                dp[i ^ k] = Math.max(dp[i ^ k], dp[i] + gcdVal.get(k) * (bits / 2 + 1));
//            }
//        }
//
//        return dp[(1 << n) - 1];
//    }
//
//    public int gcd(int a, int b) {
//        if (b == 0)
//            return a;
//        return gcd(b, a % b);
//    }
//}

// Time: O(2^n * n^2)
// Space: O(2 ^ n)