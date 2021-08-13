import java.util.*;

public class Lc5 {
    int ans = 0;

    public int maxLength2(List<String> arr) {
        List<Integer> masks = new ArrayList<Integer>();
        for (String s : arr) {
            int mask = 0;
            for (int i = 0; i < s.length(); ++i) {
                int ch = s.charAt(i) - 'a';
                if (((mask >> ch) & 1) != 0) { // 若 mask 已有 ch，则说明 s 含有重复字母，无法构成可行解
                    mask = 0;
                    break;
                }
                mask |= 1 << ch; // 将 ch 加入 mask 中
            }
            if (mask > 0) {
                masks.add(mask);
            }
        }

        backtrack(masks, 0, 0);
        return ans;
    }

    public void backtrack(List<Integer> masks, int pos, int mask) {
        if (pos == masks.size()) {
            ans = Math.max(ans, Integer.bitCount(mask));
            return;
        }
        if ((mask & masks.get(pos)) == 0) { // mask 和 masks[pos] 无公共元素
            backtrack(masks, pos + 1, mask | masks.get(pos));
        }
        backtrack(masks, pos + 1, mask);
    }

    int res = 0;

    public int maxLength(List<String> arr) {
        List<Integer> masks = new ArrayList<>();
        for (String each : arr) {
            int mask = 0;
            for (int i = 0; i < each.length(); i++) {
                int cur = each.charAt(i) - '0';
                if ((mask & cur) != 1) {
                    mask |= 1 << cur;
                } else {
                    mask = 0;
                    break;
                }
            }
            if (mask > 0) {
                masks.add(mask);
            }
        }
        back(masks, 0, 0);
        return res;
    }

    private void back(List<Integer> masks, int pos, int mask) {
        if (pos == masks.size()) {
            res = Math.max(res, Integer.bitCount(mask));
            return;
        }
        if ((masks.get(pos) & mask) == 0) {
            back(masks, pos + 1, mask | masks.get(pos));
        }
        back(masks, pos + 1, mask);
    }
//    public int maxLength(List<String> arr) {
//        List<Integer> masks = new ArrayList<>();
//        for (String each : arr) {
//            int mask = 0;
//            char[] array = each.toCharArray();
//            for (char eachChar : array) {
//                int cur = eachChar - '0';
//                if (((mask >> cur) & 1) != 0) {
//                    mask = 0;
//                    break;
//                } else {
//                    mask |= 1 << cur;
//                }
//            }
//            if (mask != 0) {
//                masks.add(mask);
//            }
//        }
//        return back(masks, 0, 0, 0);
//    }
//
//    private int back(List<Integer> masks, int pos, int res, int mask) {
//        if (pos == masks.size()) {
//            return Math.max(Integer.bitCount(mask), res);
//        }
//        if ((masks.get(pos) & mask) == 0) {
//            return back(masks, pos + 1, res, mask | masks.get(pos));
//        }
//        return back(masks, pos + 1, res, mask);
//
//    }

    int[] jobs;
    int n, k;
    int ans2 = 0x3f3f3f3f;

    public int minimumTimeRequired(int[] _jobs, int _k) {
        jobs = _jobs;
        n = jobs.length;
        k = _k;
        int[] sum = new int[k];
        dfs(0, 0, sum, 0);
        return ans;
    }

    private void dfs(int pos, int used, int[] sum, int max) {
        if (max >= ans2) {
            return;
        }
        if (used < sum.length) {
            sum[used] += jobs[pos];
            dfs(pos + 1, used + 1, sum, Math.max(max, sum[used]));
            sum[used] -= jobs[pos];
        }
        for (int i = 0; i < used; i++) {
            sum[i] += jobs[pos];
            dfs(pos + 1, used, sum, Math.max(max, sum[i]));
            sum[i] -= jobs[pos];
        }
    }


    public int[][] rotateGrid(int[][] grid, int k) {
        // 矩阵尺寸
        int m = grid.length, n = grid[0].length;
        // 计算一共有多少层
        int layerNumber = Math.min(m, n) / 2;
        for (int i = 0; i < layerNumber; i++) {
            int[] data = new int[((m - 2 * i) * (n - 2 * i)) - (m - (i + 1) * 2) * (n - (i + 1) * 2)];
            int id = 0;
            for (int start = i; i < n - i - 1; i++) {
                data[id++] = grid[i][start];
            }
            for (int start = i; i < m - i - 1; i++) {
                data[id++] = grid[start][n - i - 1];
            }
            for (int start = n - i - 1; i > 0; i--) {
                data[id++] = grid[m - i - 1][start];
            }
            for (int start = m - i - 1; i > 0; i--) {
                data[id++] = grid[start][i];
            }
            Integer[] ret = rotateVector(data, k);

        }
        return null;
    }

    public int[][] rotateGrid2(int[][] grid, int k) {
        // 矩阵尺寸
        int m = grid.length, n = grid[0].length;
        // 计算一共有多少层
        int layerNumber = Math.min(m, n) / 2;
        // 逐层处理
        for (int i = 0; i < layerNumber; ++i) {
            // 计算出来当前层需要多大的数组来存放, 计算方法是当前层最大尺寸 - 内部下一层尺寸.
            int[] data = new int[(m - 2 * i) * (n - 2 * i) - (m - (i + 1) * 2) * (n - (i + 1) * 2)];
            int idx = 0;
            // 从左往右
            for (int offset = i; offset < n - i - 1; ++offset)
                data[idx++] = grid[i][offset];
            // 从上往下
            for (int offset = i; offset < m - i - 1; ++offset)
                data[idx++] = grid[offset][n - i - 1];
            // 从右往左
            for (int offset = n - i - 1; offset > i; --offset)
                data[idx++] = grid[m - i - 1][offset];
            // 从下往上
            for (int offset = m - i - 1; offset > i; --offset)
                data[idx++] = grid[offset][i];
            // 把旋转完的放回去
            Integer[] ret = rotateVector(data, k);
            idx = 0;
            // 从左往右
            for (int offset = i; offset < n - i - 1; ++offset)
                grid[i][offset] = ret[idx++];
            // 从上往下
            for (int offset = i; offset < m - i - 1; ++offset)
                grid[offset][n - i - 1] = ret[idx++];
            // 从右往左
            for (int offset = n - i - 1; offset > i; --offset)
                grid[m - i - 1][offset] = ret[idx++];
            // 从下往上
            for (int offset = m - i - 1; offset > i; --offset)
                grid[offset][i] = ret[idx++];
        }
        return grid;
    }

    private Integer[] rotateVector(int[] vector, int offset) {
        // 取余, 否则会有无用功, 超时
        offset = offset % vector.length;
        Deque<Integer> deque = new ArrayDeque<>();
        for (int item : vector) deque.add(item);
        // 旋转操作
        while (offset-- > 0) {
            deque.addLast(deque.pollFirst());
        }
//        deque.toArray();
        return (Integer[]) deque.toArray();
//        return deque.toArray(new Integer[0]);
    }

    public int subarraySum(int[] nums, int k) {
        Set<Integer> set = new HashSet<>();
        int sum = 0;
        set.add(0);
        int res = 0;
        for (int num : nums) {
            sum += num;
            if (set.contains(sum - k)) {
                res++;
            }
            set.add(sum);
        }
        return res;
    }

    public int numSubarraysWithSum(int[] nums, int goal) {
        int res = 0, sum = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        for (Integer each : nums) {
            if (map.containsKey(sum - goal)) {
                res += map.get(sum - goal);
            }
            map.put(sum, map.getOrDefault(sum, 0) + 1);
        }
        return res;
    }

    public int pivotIndex(int[] nums) {
        int[] sum = new int[nums.length + 2];
        for (int i = 0; i < nums.length; i++) {
            sum[i + 1] = sum[i] + nums[i];
        }
        sum[nums.length + 1] = sum[nums.length];
        for (int i = 1; i <= nums.length; i++) {
            if (sum[i - 1] == sum[nums.length + 1] - sum[i]) {
                return i;
            }
        }
        return -1;
    }

    public int numberOfSubarrays(int[] nums, int k) {
        int sum = 0, res = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        for (Integer each : nums) {
            sum += each & 1;
            if (map.containsKey(sum - k)) {
                res += map.get(sum - k);
            }
            map.put(sum, map.getOrDefault(sum, 0) + 1);
        }
        return res;
    }

    public int subarraysDivByK(int[] nums, int k) {
        int sum = 0, res = 0;
        Map<Integer, Integer> map = new HashMap<>();
        for (Integer each : nums) {
            sum += each;
            if (map.containsKey(sum % k)) {
                res += map.get(sum % k);
            }
            map.put(sum % k, map.getOrDefault(sum % k, 0) + 1);
        }
        return res;
    }


    public int minSubarray2(int[] nums, int p) {
        long sum = 0, mod = 0;
        int res = Integer.MAX_VALUE;
        for (Integer num : nums) {
            sum += num;
        }
        mod = sum % p;
        if (mod == 0) {
            return 0;
        }
        sum = 0;
        Map<Long, Integer> map = new HashMap<>();
        map.put(0L, 0); //此题不需要！！
        for (int i = 1; i <= nums.length; i++) {
            sum += nums[i - 1];
            long tmp = (sum + p - mod) % p;
            if (map.containsKey(tmp)) {
                res = Math.min(res, i - map.get(tmp));
            }
            map.put(sum % p, i);
        }
        return res == Integer.MAX_VALUE ? -1 : res;
    }

    public int minSubarray(int[] nums, int p) {
        int n = nums.length, mod = 0;
        for (int num : nums) {
            mod = (mod + num) % p;
        }
        if (mod == 0) {
            return 0;
        }
        int res = n, subMod = 0;
        Map<Integer, Integer> hashmap = new HashMap<>();
        hashmap.put(0, -1);
        for (int i = 0; i < n; i++) {
            subMod = (subMod + nums[i]) % p;
            int target = (subMod - mod + p) % p;
            if (hashmap.containsKey(target)) {
                res = Math.min(res, i - hashmap.get(target));
                if (res == 1 && res != n) {
                    return res;
                }
            }
            hashmap.put(subMod, i);
        }
        if (res == n) {
            return -1;
        }
        return res;
    }


    public int longestAwesome(String s) {
        int res = 0, sum = 0;
        int[] map = new int[s.length() + 1];
        for (int i = 1; i < s.length() + 1; i++) {
            int cur = (1 << (s.charAt(i - 1) - '0'));
            sum ^= cur;
            map[i] = sum;
        }
        for (int i = 0; i < s.length() + 1; i++) {
            for (int j = i + 1; j < s.length() + 1; j++) {
                if (Integer.bitCount(map[i] ^ map[j]) <= 1) {
                    res = Math.max(res, j - i);
                }
            }
        }
        return res;
    }

    public int longestAwesome2(String s) {
        int res = 0, sum = 0;
        int[] map = new int[s.length()];
        for (int i = 0; i < s.length(); i++) {
            int cur = (1 << (s.charAt(i) - '0'));
            sum ^= cur;
            map[i] = sum;
        }
        for (int i = 0; i < s.length(); i++) {
            for (int j = i + 1; j < s.length(); j++) {
                if (Integer.bitCount(map[i] ^ map[j]) <= 1) {
                    res = Math.max(res, j - i);
                }
            }
        }
        return res;
    }

    public String convertToTitle(int columnNumber) {
        StringBuffer res = new StringBuffer();
        while (columnNumber > 0) {
            int cur = columnNumber % 26;
            if (cur == 0) {
                res.append((char) 'Z');
            } else {
                res.append((char) ('A' + cur - 1));
            }
            columnNumber /= 26;
        }
        return res.reverse().toString();
    }

    int[][] directions = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};

    public boolean exist(char[][] board, String word) {
        char[] array = word.toCharArray();
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
//                if (dfs2(board, array, 0, i, j)) {
//                    return true;
//                }
                if (dfs(board, array, i, j, 0)) {
                    return true;
                }
            }
        }
        return false;
    }

    boolean dfs(char[][] board, char[] word, int i, int j, int k) {
        if (i >= board.length || i < 0 || j >= board[0].length || j < 0 || board[i][j] != word[k]) return false;
        if (k == word.length - 1) return true;
        board[i][j] = '\0';
        boolean res = dfs(board, word, i + 1, j, k + 1) || dfs(board, word, i - 1, j, k + 1) ||
                dfs(board, word, i, j + 1, k + 1) || dfs(board, word, i, j - 1, k + 1);
        board[i][j] = word[k];
        return res;
    }

    private boolean dfs2(char[][] board, char[] array, int pos, int i, int j) {
        if (pos == array.length) {
            return true;
        }
        if (i < 0 || i >= board.length || j < 0 || j >= board[0].length || array[pos] != board[i][j]) {
            return false;
        }

        for (int k = 0; k < 4; k++) {
            int x = i + directions[k][0], y = j + directions[k][1];
            if (dfs2(board, array, pos + 1, x, y)) {
                return true;
            }
        }
        return false;
    }

    /**
     * 1.如果当前不是vowel，就看沿用上次sum，计算map.get(sum)来求res
     * 2.如果当前是vowel，计算sum，遍历vowel，看map2中是否有，有的话，
     *
     * @param s
     * @return
     */
    public int findTheLongestSubstring(String s) {
        int res = 0, sum = 0;
        Map<Integer, Integer> map = new HashMap<>();
        Map<Character, Integer> map2 = new HashMap<>();
        map2.put('a', 1);
        map2.put('e', 2);
        map2.put('i', 3);
        map2.put('o', 4);
        map2.put('u', 5);
        map.put(0, -1);
        for (int i = 0; i < s.length(); i++) {
            if (map2.containsKey(s.charAt(i))) {
                sum ^= (1 << map2.get(s.charAt(i)));
                for (Map.Entry<Character, Integer> entry : map2.entrySet()) {
                    if (map.containsKey(sum ^ (1 << entry.getValue()))) {
                        int tmp = sum ^ (1 << entry.getValue());
                        res = Math.max(res, i - map.get(tmp));
                    }
                }
                if (!map.containsKey(sum)) {
                    map.put(sum, i);
                }
            } else {
                if (map.containsKey(sum)) {
                    res = Math.max(res, i - map.get(sum));
                }
            }
        }
        return res;
    }

    public int findTheLongestSubstring2(String s) {
        int ans = 0, n = s.length();
        int[] idxs = new int[0b100000];
        Arrays.fill(idxs, -2);
        idxs[0b11111] = -1;

        int val = 0b11111;
        for (int i = 0; i < n; i++) {
            switch (s.charAt(i)) {
                case 'a':
                    val ^= 0b10000;
                    break;
                case 'e':
                    val ^= 0b01000;
                    break;
                case 'i':
                    val ^= 0b00100;
                    break;
                case 'o':
                    val ^= 0b00010;
                    break;
                case 'u':
                    val ^= 0b00001;
                    break;
            }
            if (idxs[val] == -2) idxs[val] = i;
            else ans = Math.max(ans, i - idxs[val]);
        }
        return ans;
    }

//    public int numWays(int n, int[][] relation, int k) {
////        int[] visited = new int[relation.length];
////        visited[0]=1;
//        List<Integer> trace = new ArrayList<>();
//        trace.add(relation[0][0]);
//        dfs3(relation, 0, k - 1, n,trace);
//        return res;
//    }
//
//    private void dfs3(int[][] relation, int pos, int k, int n, List<Integer> trace) {
//        if (k == 0 && relation[pos][1] == n - 1) {
//            res++;
//            trace.add(relation[pos][1]);
//            StringBuilder tmp = new StringBuilder();
//            for (Integer integer : trace) {
//                tmp.append(integer);
//            }
//            System.out.println(tmp);
//            trace.remove(trace.size()-1);
//            return;
//        }
//        if (pos >= relation.length || k < 0) {
//            return;
//        }
//        for (int i = 0; i < relation.length; i++) {
//            if (relation[pos][1] == relation[i][0]) {
////                visited[i] = 1;
//                trace.add(relation[i][0]);
//                dfs3(relation, i, k - 1, n, trace);
//                trace.remove(trace.size()-1);
////                visited[i] = 0;
//            }
//        }
//    }


    //    int ways, n, k;
//    List<List<Integer>> edges;
//    List<Integer> trace = new ArrayList<>();
//    public int numWays(int n, int[][] relation, int k) {
//        ways = 0;
//        this.n = n;
//        this.k = k;
//        edges = new ArrayList<List<Integer>>();
//        for (int i = 0; i < n; i++) {
//            edges.add(new ArrayList<Integer>());
//        }
//        for (int[] edge : relation) {
//            int src = edge[0], dst = edge[1];
//            edges.get(src).add(dst);
//        }
//        trace.add(relation[0][0]);
//        dfs(0, 0);
//        return ways;
//    }
//
//    public void dfs(int index, int steps) {
//        if (steps == k) {
//            if (index == n - 1) {
//                ways++;
//                trace.add(index);
//                StringBuilder tmp = new StringBuilder();
//                for (Integer integer : trace) {
//                    tmp.append(integer);
//                }
//                System.out.println(tmp);
//                trace.remove(trace.size()-1);
//            }
//            return;
//        }
//        List<Integer> list = edges.get(index);
//        for (int nextIndex : list) {
//            trace.add(nextIndex);
//            dfs(nextIndex, steps + 1);
//            trace.remove(trace.size()-1);
//
//        }
//    }exist
    public int numWays(int n, int[][] relation, int k) {
        int[][] dp = new int[k + 1][n];
        dp[0][0] = 0;
        for (int i = 0; i < k; i++) {
            for (int[] each : relation) {
                int src = each[0], to = each[1];
                dp[i + 1][to] = dp[i][src];
            }
        }
        return dp[k][n - 1];
    }


    public String minWindow2(String s, String t) {
        StringBuilder sb = new StringBuilder();
        int min = Integer.MAX_VALUE;
        Map<Character, Integer> map = new HashMap<>();
        for (int i = 0; i < t.length(); i++) {
            map.put(t.charAt(i), map.getOrDefault(t.charAt(i), 0) + 1);
        }
        for (int i = 0; i < s.length(); i++) {
            for (int j = i + t.length(); j < s.length(); j++) {
//              [i,j)  如果包含就要在里面把map都--
                if (containSub(i, j, s, t, map)) {
                    if (j - i < min) {
                        min = j - i;
                        sb = new StringBuilder();
                        sb.append(s, i, j);
                    }
                }
            }
            if (map.containsKey(s.charAt(i))) {
                map.put(s.charAt(i), map.get(s.charAt(i)) + 1);
            }
        }
        return sb.toString();
    }

    private boolean containSub(int start, int end, String s, String t, Map<Character, Integer> map) {
        for (int i = start; i < end; i++) {
            if (map.containsKey(s.charAt(i)) && map.get(s.charAt(i)) > 0) {
                map.put(s.charAt(i), map.get(s.charAt(i)) - 1);
            }
        }
        boolean res = true;
        for (Map.Entry<Character, Integer> entry : map.entrySet()) {
            if (entry.getValue() > 0) {
                res = false;
                break;
            }
        }
        return res;
    }


    public String minWindow(String s, String t) {
        if (s == null || s.length() == 0 || t == null || t.length() == 0) {
            return "";
        }
        int[] array = new int[26];
        for (int i = 0; i < t.length(); i++) {
            array[t.charAt(i) - 'A'] += 1;
        }
        int l = 0, r = 0, size = Integer.MAX_VALUE, count = t.length(), start = 0;
        while (r < s.length()) {
            if (array[s.charAt(r) - 'A'] > 0) {
                count--;
            }
            array[s.charAt(r) - 'A']--;

            if (count == 0) {
                while (array[s.charAt(l) - 'A'] < 0) {
                    l++;
                    array[s.charAt(l) - 'A']++;

                }
                if (r - l + 1 < size) {
                    size = r - l + 1;
                    start = l;
                }
                array[s.charAt(l) - 'A']++;
                count++;
                l++;
            }
            r++;
        }
        return size == Integer.MAX_VALUE ? "" : s.substring(start, start + size + 1);
    }

    //    public String minWindow(String s, String t) {
//        if (s == null || s.length() == 0 || t == null || t.length() == 0) {
//            return "";
//        }
//        int[] need = new int[128];
//        //记录需要的字符的个数
//        for (int i = 0; i < t.length(); i++) {
//            need[t.charAt(i)]++;
//        }
//        //l是当前左边界，r是当前右边界，size记录窗口大小，count是需求的字符个数，start是最小覆盖串开始的index
//        int l = 0, r = 0, size = Integer.MAX_VALUE, count = t.length(), start = 0;
//        //遍历所有字符
//        while (r < s.length()) {
//            char c = s.charAt(r);
//            if (need[c] > 0) {//需要字符c
//                count--;
//            }
//            need[c]--;//把右边的字符加入窗口
//            if (count == 0) {//窗口中已经包含所有字符
//                while (l < r && need[s.charAt(l)] < 0) {
//                    need[s.charAt(l)]++;//释放右边移动出窗口的字符
//                    l++;//指针右移
//                }
//                if (r - l + 1 < size) {//不能右移时候挑战最小窗口大小，更新最小窗口开始的start
//                    size = r - l + 1;
//                    start = l;//记录下最小值时候的开始位置，最后返回覆盖串时候会用到
//                }
//                //l向右移动后窗口肯定不能满足了 重新开始循环
//                need[s.charAt(l)]++;
//                l++;
//                count++;
//            }
//            r++;
//        }
//        return size == Integer.MAX_VALUE ? "" : s.substring(start, start + size);
//    }

    public int lengthOfLongestSubstring(String s) {
        if (s == null || s.length() == 0) {
            return 0;
        }
        s = s.trim();
        if (s.length() == 0) {
            return 0;
        }
        int res = 0, l = 0, r = 0;
        Map<Character, Integer> map = new HashMap<>();
        while (r < s.length()) {
            if (map.getOrDefault(s.charAt(r), 0) == 0) {
                map.put(s.charAt(r), map.getOrDefault(s.charAt(r), 0) + 1);
                r++;
            } else {
                if (r - l > res) {
                    res = r - l;
                }
                map.put(s.charAt(l), map.get(s.charAt(l)) - 1);
                l++;
            }
        }
        return res;
    }


//    List<Integer> res2 = new ArrayList<>();
//
//    public List<Integer> findSubstring(String s, String[] words) {
//        if (s == null || s.length() == 0) {
//            return res2;
//        }
//        int l = 0, size = 0;
//        Map<String, Integer> map = new HashMap<>();
//        for (String each : words) {
//            size += each.length();
//            map.put(each, 1);
//        }
//        while (l < s.length()) {
//            if (containSub2(s, l, size, map)) {
//                res2.add(l);
//            } else {
//                l++;
//            }
//        }
//        return res2;
//    }
//
//    private boolean containSub2(String s, int start, int size, Map<String, Integer> map) {
//        int l = start, r = start;
//        while (r <= start + size && r < s.length()) {
//            if (map.containsKey(s.substring(l, r)) && map.getOrDefault(s.substring(l, r), 0) == 1) {
//                map.put(s.substring(l, r), 0);
//                res2.add(l);
//                l = r;
//            }
//            r++;
//        }
//        boolean result = true;
//        for (Map.Entry<String, Integer> entry : map.entrySet()) {
//            if (entry.getValue() == 1) {
//                result = false;
//            }
//            map.put(entry.getKey(), 1);
//        }
//        if (!result) {
//            res2 = new ArrayList<>();
//        }
//        return result;
//    }


    public List<Integer> findSubstring(String s, String[] words) {
        List<Integer> res = new ArrayList<>();
        if (s == null || s.length() == 0 || words == null || words.length == 0) return res;
        HashMap<String, Integer> map = new HashMap<>();
        int one_word = words[0].length();
        int word_num = words.length;
        int all_len = one_word * word_num;
        for (String word : words) {
            map.put(word, map.getOrDefault(word, 0) + 1);
        }
        for (int i = 0; i < one_word; i++) {
            HashMap<String, Integer> tmpMap = new HashMap<>();
            int l = i, r = i, count = 0;
            while (r + one_word <= s.length()) {
                String cur = s.substring(r, r + one_word);
                r += one_word;
                if (!map.containsKey(cur)) {
                    l = r;
                    count = 0;
                    tmpMap.clear();
                } else {
                    tmpMap.put(cur, tmpMap.getOrDefault(cur, 0) + 1);
                    count++;
                    while (map.get(cur) < tmpMap.get(cur)) {
                        count--;
                        l += one_word;
                        tmpMap.put(cur, tmpMap.get(cur) - 1);
                    }
                    if (count == word_num) {
                        res.add(l);
                    }
                }
            }
        }
        return res;
    }

    public int lengthOfLongestSubstring2(String s) {
        if (s.length() == 0) return 0;
        HashMap<Character, Integer> map = new HashMap<Character, Integer>();
        int max = 0;
        int left = 0;
        for (int i = 0; i < s.length(); i++) {
            if (map.containsKey(s.charAt(i))) {
                left = Math.max(left, map.get(s.charAt(i)) + 1);
            }
            map.put(s.charAt(i), i);
            max = Math.max(max, i - left + 1);
        }
        return max;
    }

    public int minSubArrayLen(int target, int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int l = 0, r = 0, min = Integer.MAX_VALUE, sum = 0;
        while (r < nums.length) {
            if (sum < target) {
                sum += nums[r];
                r++;
            }
            while (sum >= target) {
                min = Math.min(min, r - l);
                sum -= nums[l];
                l++;
            }

        }
        return min==Integer.MAX_VALUE?0:min;
    }
    public int minSubArrayLen2(int s, int[] nums) {
        int length = nums.length;
        int min = Integer.MAX_VALUE;
        int[] sums = new int[length + 1];
        for (int i = 1; i <= length; i++) {
            sums[i] = sums[i - 1] + nums[i - 1];
        }
        for (int i = 0; i <= length; i++) {
            int target = s + sums[i];
            int index = Arrays.binarySearch(sums, target);
            if (index < 0)
                index = -index+1;
            if (index <= length) {
                min = Math.min(min, index - i);
            }
        }
        return min == Integer.MAX_VALUE ? 0 : min;
    }



    public static void main(String[] args) {
        Lc5 lc5 = new Lc5();
//        int res = lc5.maxLength(Arrays.asList("cha", "r", "act", "ers"))
//        int res = lc5.findTheLongestSubstring2("eleetminicoworoep");
//        int[][] pa = {{0, 2}, {2, 1}, {3, 4}, {2, 3}, {1, 4}, {2, 0}, {0, 4}};
//        int res = lc5.numWays(5, pa, 3);
        int[] pa = {2,3,1,2,4,3};
        int res = lc5.minSubArrayLen2(7,pa);
        System.out.println(res);
//        int res =lc5.longestAwesome("3242415");
//        StringBuffer sb = new StringBuffer();
//        sb.append('A');
//        sb.append((char) ('A' + 1));
//        System.out.println(sb.toString());
//        System.out.println(-1 % 2);
    }
}
