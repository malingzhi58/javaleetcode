import java.lang.reflect.Array;
import java.util.*;

import static com.sun.tools.javac.jvm.ByteCodes.ret;
import static com.sun.tools.javac.jvm.ByteCodes.swap;

public class Lc8 {
    public int countTriples(int n) {
//        int[] dp = new int[n+1];
        int count = 0;
        Set<Integer> set = new HashSet<>();
        for (int i = 1; i <= n; i++) {
            int curSquare = i * i;
            for (int j = 1; j < i; j++) {
                int loopSquare = j * j;
                if (set.contains(curSquare - loopSquare)) {
                    count++;
                }
            }
            set.add(i * i);
        }
        return count;
    }

    int count = Integer.MAX_VALUE;
    int[] entrance;
    int min;

    public int nearestExit(char[][] maze, int[] _entrance) {
        entrance = _entrance;
        min = Math.min(entrance[0], Math.min(entrance[1], Math.min(maze.length - entrance[0] - 1, maze[0].length - 1 - entrance[1])));
        dfs(maze, entrance[0], entrance[1], 0);
        return count == Integer.MAX_VALUE ? -1 : count;
    }

    private void dfs(char[][] maze, int row, int col, int depth) {
        if (count == min) return;
        if (depth >= count) return;
        if (row != entrance[0] || col != entrance[1]) {
            if (row < 0 || row >= maze.length || col < 0 || col >= maze[0].length || maze[row][col] == '+') return;
        }
        if (row != entrance[0] || col != entrance[1]) {
            if (row == 0 || row == maze.length - 1 || col == 0 || col == maze[0].length - 1) {
                count = Math.min(count, depth);
                return;
            }
        }

        maze[row][col] = '+';
        dfs(maze, row + 1, col, depth + 1);
        dfs(maze, row - 1, col, depth + 1);
        dfs(maze, row, col + 1, depth + 1);
        dfs(maze, row, col - 1, depth + 1);
        maze[row][col] = '.';
    }

    int[][] direction = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    boolean[][] visited;
    int step;

    public int nearestExit2(char[][] maze, int[] entrance) {
        int m = maze.length;
        int n = maze[0].length;
        visited = new boolean[m][n];
        int x = entrance[0];
        int y = entrance[1];
        step = 1;
        int res = bfs(maze, x, y, step);
        return res;
    }

    private int bfs(char[][] maze, int x, int y, int step) {
        Queue<int[]> queue = new LinkedList<>();
        queue.offer(new int[]{x, y});
        while (!queue.isEmpty()) {
            int[] cur = queue.poll();
            int row = cur[0], col = cur[1];
            for (int i = 0; i < direction.length; i++) {
                int nextrow = row + direction[i][0];
                int nextcol = col + direction[i][1];
                if (nextrow < 0 || nextrow >= maze.length || nextcol < 0 || nextcol >= maze[0].length || visited[nextrow][nextcol]) {
                    continue;
                }
                visited[nextrow][nextcol] = true;
                queue.offer(new int[]{nextrow, nextcol});
            }
            step++;
        }
        return -1;
    }

    public boolean validPalindrome(String s) {
        if (isPalindrome(s)) {
            return true;
        }
        int l = 0, r = s.length() - 1;

        while (l < r) {
            if (s.charAt(l) != s.charAt(r)) {
                if (isPalindrome(s.substring(l, r))) {
                    return true;
                }
                if (isPalindrome(s.substring(l + 1, r + 1))) {
                    return true;
                }
                return false;
            }
            l++;
            r--;
        }
        return true;
    }

    private boolean isPalindrome(String s) {
        int l = 0, r = s.length() - 1;
        while (l < r) {
            if (s.charAt(l) != s.charAt(r)) {
                return false;
            }
            l++;
            r--;
        }
        return true;
    }

    public int[] getConcatenation(int[] nums) {
        int[] res = new int[nums.length * 2];
        for (int i = 0; i < nums.length * 2; i++) {
            if (i < nums.length) {
                res[i] = nums[i];
            } else {
                res[i] = nums[i - nums.length];
            }
        }
        return res;

    }

    public int countPalindromicSubsequence(String s) {
        int count = 0;
        HashSet<String> set = new HashSet<>();
        StringBuffer sb = new StringBuffer();
        StringBuffer sb2 = new StringBuffer();
        StringBuffer sb3 = new StringBuffer();
        for (int i = 0; i < s.length() - 2; i++) {
            char first = s.charAt(i);
            for (int j = i + 1; j < s.length() - 1; j++) {
                int right = s.length() - 1;
                char sec = s.charAt(j);
                sb = new StringBuffer();
                sb.append(first).append(first).append(first);
                sb2 = new StringBuffer();
                sb2.append(first).append(sec).append(first);
                if (first == sec && !set.contains(sb.toString())) {
                    while (right > j) {
                        if (s.charAt(right) == sec && !set.contains(sb.toString())) {
                            count++;
                            sb3 = new StringBuffer();
                            sb3.append(first).append(first).append(first);
                            set.add(sb3.toString());
                        }
                        right--;
                    }
                } else if (first != sec && !set.contains(sb2.toString())) {
                    while (right > j) {
                        if (s.charAt(right) == first && !set.contains(sb2.toString())) {
                            count++;
                            sb3 = new StringBuffer();
                            sb3.append(first).append(sec).append(first);
                            set.add(sb3.toString());
                        }
                        right--;
                    }
                }
            }
        }
        return count;
    }


    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        if (!wordList.contains(endWord)) {
            return 0;
        }
        Set<String> visited = new HashSet<>();
        Queue<String> frontQueue = new LinkedList<>();
        Queue<String> endQueue = new LinkedList<>();
        frontQueue.offer(beginWord);
        endQueue.offer(endWord);
        visited.add(beginWord);
        visited.add(endWord);
        int count = 0;
        while (!frontQueue.isEmpty() && !endQueue.isEmpty()) {
            if (endQueue.size() < frontQueue.size()) {
                Queue<String> tmp = frontQueue;
                frontQueue = endQueue;
                endQueue = tmp;
            }
            int size = frontQueue.size();
            ++count;
            for (int i = 0; i < size; ++i) {
                String start = frontQueue.poll();
                for (String s : wordList) {

                    // 已经遍历的不再重复遍历
                    if (visited.contains(s)) {
                        continue;
                    }
                    // 不能转换的直接跳过
                    if (!canConvert(start, s)) {
                        continue;
                    }
                    // 用于调试
                    // System.out.println(count + ": " + start + "->" + s);
                    // 可以转换，并且能转换成 endWord，则返回 count
                    if (endQueue.contains(s)) {
                        return count + 1;
                    }
                    // 保存访问过的单词，同时把单词放进队列，用于下一层的访问
                    visited.add(s);
                    frontQueue.offer(s);
                }
            }


        }
        return 0;
    }

    public boolean canConvert(String s1, String s2) {
        if (s1.length() != s2.length()) return false;
        int count = 0;
        for (int i = 0; i < s1.length(); ++i) {
            if (s1.charAt(i) != s2.charAt(i)) {
                ++count;
                if (count > 1) {
                    return false;
                }
            }
        }
        return count == 1;
    }

    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dum = new ListNode(-1);
        dum.next = head;
        ListNode pre = dum;
        for (int i = 0; i < n; i++) {
            head = head.next;
        }
        while (head != null) {
            head = head.next;
            pre = pre.next;
        }
        pre.next = pre.next.next;
        return dum.next;
    }

    public int[][] insert(int[][] intervals, int[] newInterval) {
        if (intervals.length == 0) {
            return new int[][]{newInterval};
        }
        List<int[]> res = new ArrayList<>();
        for (int i = 0; i < intervals.length; ) {
            if (intervals[i][1] > newInterval[0]) {
                if (newInterval[1] < intervals[i][0]) {
                    res.add(intervals[i]);
                    i++;
                } else {
                    int left = Math.min(intervals[i][0], newInterval[0]);
                    int right = Math.min(intervals[i][1], newInterval[1]);
                    int j = i;
                    for (; j < intervals.length; j++) {
//                        find the first the left greater than the newInterval
                        if (intervals[j][0] > newInterval[1]) {
                            right = Math.max(intervals[j - 1][1], newInterval[1]);
                            break;
                        }
                    }
                    res.add(new int[]{left, right});
                    i = j;
                }
            } else {
                res.add(intervals[i]);
                i++;
            }

        }
        int[][] result = new int[res.size()][2];
        for (int i = 0; i < res.size(); i++) {
            result[i][0] = res.get(i)[0];
            result[i][1] = res.get(i)[1];
        }
        return result;
    }

    public int leastInterval(char[] tasks, int n) {
        Map<Character, Integer> map = new HashMap<>();

        for (Character each : tasks) {
            map.put(each, map.getOrDefault(each, 0) + 1);
        }
        int max = 0;
        for (Map.Entry<Character, Integer> entry : map.entrySet()) {
            max = Math.max(max, entry.getValue());
        }
        int count1 = 0;
        for (Map.Entry<Character, Integer> entry : map.entrySet()) {
            if (entry.getValue() == max) {
                count1++;
            }
        }
        count1 += (max - 1) * (n + 1);
        return Math.max(count1, tasks.length);
    }

    public boolean searchMatrix2(int[][] matrix, int target) {
        int row = 0, col = matrix[0].length - 1;
        while (row < matrix.length && col >= 0) {
            if (matrix[row][col] == target) {
                return true;
            } else if (matrix[row][col] > target) {
                col--;
            } else {
                row++;
            }
        }
        return false;
    }

    public boolean searchMatrix3(int[][] matrix, int target) {
        int row = matrix.length, col = matrix[0].length;
        int l = 0, r = row - 1;
        while (l < r) {
            int mid = (r - l + 1) / 2 + l;
            if (matrix[mid][0] <= target) {
                l = mid;
            } else {
                r = mid - 1;
            }
        }
        if (matrix[l][0] > target) return false;
        if (matrix[l][0] == target) return true;
        int targetRow = l;
        l = 0;
        r = col - 1;
        while (l < r) {
            int mid = (r - l) / 2 + l;
            if (matrix[targetRow][mid] == target) {
                return true;
            } else if (matrix[targetRow][mid] > target) {
                r = mid - 1;
            } else {
                l = mid + 1;
            }
        }
        if (matrix[targetRow][l] == target) return true;
        else return false;

    }

    public boolean searchMatrix(int[][] matrix, int target) {
        if (matrix == null || matrix.length == 0) {
            return false;
        }
        int shortDim = Math.min(matrix.length, matrix[0].length);
        for (int i = 0; i < shortDim; i++) {
            boolean verticalRes = binary(matrix, i, target, true);
            boolean horizontalRes = binary(matrix, i, target, false);
            if (verticalRes || horizontalRes) return true;
        }
        return false;
    }

    private boolean binary(int[][] matrix, int start, int target, boolean isVertical) {
        int l = start, r = isVertical ? matrix.length - 1 : matrix[0].length - 1;
        while (l < r) {
            if (isVertical) {
                int mid = (r - l) / 2 + r;
                if (matrix[mid][start] == target) {
                    return true;
                } else if (matrix[mid][start] > target) {
                    r = mid - 1;
                } else {
                    l = mid + 1;
                }
            } else {
                int mid = (r - l) / 2 + r;
                if (matrix[start][mid] == target) {
                    return true;
                } else if (matrix[start][mid] > target) {
                    r = mid - 1;
                } else {
                    l = mid + 1;
                }
            }
        }
        return isVertical ? matrix[l][start] == target : matrix[start][l] == target;
    }

    public int minMeetingRooms(int[][] intervals) {
        new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] - o2[0];
            }
        };
        Arrays.sort(intervals, (o1, o2) -> o1[0] - o2[0]);
        PriorityQueue<Integer> pq = new PriorityQueue<>((x1, x2) -> x1 - x2);
//        PriorityQueue<Integer> pq2 = new PriorityQueue<>((x1,x2)->x2-x1);
        pq.offer(intervals[0][1]);
        for (int i = 1; i < intervals.length; i++) {
            if (intervals[i][0] >= pq.peek()) {
                pq.poll();
            }
            pq.offer(intervals[i][1]);
        }
        return pq.size();
    }

    //    int[][] direction = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    public int numIslands(char[][] grid) {
        Queue<int[]> queue = new LinkedList<>();
        int count = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == '1') {
                    count++;
                    queue.offer(new int[]{i, j});
                    while (!queue.isEmpty()) {
                        int[] cur = queue.poll();
                        int curx = cur[0], cury = cur[1];
                        if (grid[curx][cury] == '0') continue;
                        grid[curx][cury] = '0';
                        for (int k = 0; k < direction.length; k++) {
                            int targetx = curx + direction[k][0];
                            int targety = cury + direction[k][1];
                            if (targety < 0 || targetx < 0 || targety >= grid[0].length || targetx >= grid.length || grid[targetx][targety] == '0') {
                                continue;
                            }
                            System.out.println(" x " + targetx + " y " + targety);
                            queue.offer(new int[]{targetx, targety});
                        }
                    }
                }
            }
        }
        return count;
    }

    public int numIslands2(char[][] grid) {
        Queue<int[]> queue = new LinkedList<>();
//        queue.offer(new int[]{0,0});
        int count = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == '1') {
                    count++;
                    queue.offer(new int[]{i, j});
                    while (!queue.isEmpty()) {
                        int[] cur = queue.poll();
                        int curx = cur[0], cury = cur[1];
                        if (curx >= 0 && cury >= 0 && cury < grid[0].length && curx < grid.length && grid[curx][cury] == '1') {
                            grid[curx][cury] = '0';
                            for (int k = 0; k < direction.length; k++) {
                                int targetx = curx + direction[k][0];
                                int targety = cury + direction[k][1];
                                queue.offer(new int[]{targetx, targety});
                            }
                        }
                    }
                }
            }
        }
        return count;
    }

    public int hIndex(int[] citations) {
        int len = citations.length;
        for (int i = len - 1; i >= 0; ) {
            if (citations[i] >= len - i) {
                i--;
            } else {
                return i + 1;
            }
        }
        return 0;
    }

    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null) return true;
        if (p == null || q == null) return false;
        if (p.val != q.val) return false;
        return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
    }

    int preval = 0;
    int curval = 0;
    boolean done = false;
    List<Integer> res = new ArrayList<>();

    public void recoverTree(TreeNode root) {
        formlist(root);
        int l = 0, r = res.size() - 1;
        while (l + 1 < res.size()) {
            if (res.get(l) > res.get(l + 1)) {
                break;
            }
            l++;
        }
        while (r - 1 >= 0) {
            if (res.get(r) < res.get(r - 1)) {
                break;
            }
            r--;
        }
        swap2(root, res.get(l), res.get(r));
        return;
    }

    private void swap2(TreeNode root, Integer l, Integer r) {
        if (root == null) return;
        swap2(root.left, l, r);
        if (root.val == l) {
            root.val = r;
        } else if (root.val == r) {
            root.val = l;
        }
        swap2(root.right, l, r);
    }

    private void formlist(TreeNode root) {
        if (root == null) return;
        formlist(root.left);
        res.add(root.val);
        formlist(root.right);
    }


    public void recoverTree2(TreeNode root) {
        findthevals(root);
        swap(root);
        return;
    }

    private void findthevals(TreeNode root) {
        if (root == null) return;
        findthevals(root.left);
        if (preval > root.val) {
            curval = root.val;
            return;
        }
        preval = root.val;
        findthevals(root.right);
    }

    private void swap(TreeNode root) {
        if (root == null) return;
        findthevals(root.left);
        if (root.val == preval) {
            root.val = curval;
        } else if (root.val == curval) {
            root.val = preval;
            done = true;
        }
        if (done) return;
        findthevals(root.right);

    }

    List<String> res2 = new ArrayList<>();

    public List<String> binaryTreePaths3(TreeNode root) {
        dfs2(root, new StringBuffer());
        return res2;
    }

    private void dfs2(TreeNode root, StringBuffer sb) {
        if (root == null) {
            return;
        }
        if (root.left == null && root.right == null) {
//            sb.deleteCharAt(sb.length()-1).deleteCharAt(sb.length()-1);
            sb.append(root.val);
            res2.add(sb.toString());
            sb.deleteCharAt(sb.length() - 1);
            return;
        }
        sb.append(root.val);
        sb.append("->");
        dfs2(root.left, sb);
        dfs2(root.right, sb);
        sb.deleteCharAt(sb.length() - 1).deleteCharAt(sb.length() - 1).deleteCharAt(sb.length() - 1);

    }

    public List<String> binaryTreePaths4(TreeNode root) {
        List<String> res = new ArrayList<>();
        dfs3(root, res, new StringBuffer());
        return res;
    }

    private void dfs3(TreeNode root, List<String> res, StringBuffer sb) {
        if (root == null) return;
        if (root.left == null && root.right == null) {
            sb.append(root.val);
            res.add(sb.toString());
            return;
        }
        dfs3(root.left, res, new StringBuffer(sb).append(root.val).append("->"));
        dfs3(root.right, res, new StringBuffer(sb).append(root.val).append("->"));
    }

    public List<String> binaryTreePaths(TreeNode root) {
        List<String> res = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        dfs4(root, res, path);
        return res;
    }

    private void dfs4(TreeNode root, List<String> res, List<Integer> path) {
        if (root == null) return;
        if (root.left == null && root.right == null) {
            path.add(root.val);
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < path.size(); i++) {
                sb.append(path.get(i)).append("->");
            }
            sb.deleteCharAt(sb.length() - 1).deleteCharAt(sb.length() - 1);
            res.add(sb.toString());
            return;
        }
        path.add(root.val);
        if (root.left != null) {
            dfs4(root.left, res, path);
            path.remove(path.size() - 1);
        }
        if (root.right != null) {
            dfs4(root.right, res, path);
            path.remove(path.size() - 1);
        }
    }

    public TreeNode invertTree(TreeNode root) {
        if (root == null) return root;
        TreeNode right = invertTree(root.right);
        TreeNode left = invertTree(root.left);
        root.left = right;
        root.right = left;
        return root;
    }

    public int sumNumbers(TreeNode root) {
        List<List<Integer>> list = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        dfs5(root, list, path);
        int res = 0;
        for (int i = 0; i < list.size(); i++) {
            List<Integer> tmp = list.get(i);
            int count = 0;
            for (int j = 0; j < tmp.size(); j++) {
                count = count * 10 + tmp.get(j);
            }
            res += count;
        }
        return res;
    }

    private void dfs5(TreeNode root, List<List<Integer>> list, List<Integer> path) {
        if (root == null) return;
        if (root.left == null && root.right == null) {
            path.add(root.val);
            list.add(new ArrayList<>(path));
            return;
        }
        path.add(root.val);
        if (root.left != null) {
            dfs5(root.left, list, path);
            path.remove(path.size() - 1);
        }
        if (root.right != null) {
            dfs5(root.right, list, path);
            path.remove(path.size() - 1);
        }
    }

    int res3 = 0;

    public int countNodes(TreeNode root) {
        dfs6(root);
        return res3;
    }

    private void dfs6(TreeNode root) {
        if (root == null) return;
        dfs6(root.left);
        res3++;
        dfs6(root.right);
    }

    public List<List<Integer>> getSkyline(int[][] buildings) {
        List<List<Integer>> res = new ArrayList<>();
        List<int[]> buildLines = new ArrayList<>();
        for (int[] points : buildings) {
            buildLines.add(new int[]{points[0], -points[2]});
            buildLines.add(new int[]{points[1], points[2]});
        }
        Collections.sort(buildLines, (a, b) -> {
            if (a[0] != b[0]) return a[0] - b[0];
            else return a[1] - b[1];
        });
        PriorityQueue<Integer> pq = new PriorityQueue<>((a, b) -> b - a);
        pq.add(0);
        int pre = 0;
        for (int[] point : buildLines) {
            if (point[1] < 0) {
                pq.add(-point[1]);
            } else {
                pq.remove(point[1]);
            }
            int cur = pq.peek();
            if (cur != pre) {
                res.add(Arrays.asList(point[0], cur));
                pre = cur;
            }
        }
        return res;
    }

    public int minAbsoluteSumDiff(int[] nums1, int[] nums2) {
        final int MOD = 1000000007;
        int len = nums1.length;
        int[] rep = new int[len];
        System.arraycopy(nums1, 0, rep, 0, len);
        Arrays.sort(rep);
        int sum = 0, diff = 0;
        for (int i = 0; i < len; i++) {
            int oriDiff = Math.abs(nums1[i] - nums2[i]);
            int close = binarysearch(rep, nums2[i]);
            if (close > 1) {
                diff = Math.max(diff, oriDiff - Math.abs(rep[close - 1] - nums2[i]));
            }
            if (close < len) {
                diff = Math.max(diff, oriDiff - Math.abs(rep[close] - nums2[i]));
            }
            sum += (oriDiff) % MOD;
        }
        return (sum - diff + MOD) % MOD;
    }

    private int binarysearch(int[] nums1, int target) {
        int l = 0, r = nums1.length - 1;
        if (nums1[r] < target) {
            return r + 1;
        }
        while (l < r) {
            int mid = (r - l) / 2 + l;
            if (nums1[mid] == target) {
                return mid;
            } else if (nums1[mid] < target) {
                l = mid + 1;
            } else {
                r = mid;
            }
        }
        return l;
    }

    public int minAbsoluteSumDiff2(int[] nums1, int[] nums2) {
        boolean[] visited = new boolean[nums1.length];
        int sum = 0;
        for (int k = 0; k < nums1.length; k++) {
            int max = Integer.MIN_VALUE, maxIndex = 0, originSum = 0;
            for (int i = 0; i < nums1.length; i++) {
                int tmp = nums1[i] > nums2[i] ? nums1[i] - nums2[i] : nums2[i] - nums1[i];
                if (tmp > max && !visited[i]) {
                    max = tmp;
                    maxIndex = i;
                }
                originSum += tmp;
                originSum %= 1e9 + 7;
            }
            int gap = Integer.MAX_VALUE, replaceIndex = 0;
            for (int i = 0; i < nums1.length; i++) {
                int tmp = nums1[i] > nums2[maxIndex] ? nums1[i] - nums2[maxIndex] : nums2[maxIndex] - nums1[i];
                if (tmp < gap) {
                    gap = tmp;
                    replaceIndex = i;
                }
            }
            if (replaceIndex == maxIndex) {
                visited[maxIndex] = true;
                continue;
            }
            nums1[maxIndex] = nums1[replaceIndex];

            for (int i = 0; i < nums1.length; i++) {
                sum += nums1[i] > nums2[i] ? nums1[i] - nums2[i] : nums2[i] - nums1[i];
                sum %= 1e9 + 7;
            }
            if (sum < originSum) break;
            else visited[maxIndex] = true;
        }
        return sum;
    }

    public int minAbsoluteSumDiff3(int[] nums1, int[] nums2) {
        final int MOD = 1000000007;
        int n = nums1.length;
        int[] rec = new int[n];
        System.arraycopy(nums1, 0, rec, 0, n);
        Arrays.sort(rec);
        int sum = 0, maxn = 0;
        for (int i = 0; i < n; i++) {
            int diff = Math.abs(nums1[i] - nums2[i]);
            sum = (sum + diff) % MOD;
            int j = binarySearch(rec, nums2[i]);
            if (j < n) {
                maxn = Math.max(maxn, diff - (rec[j] - nums2[i]));
            }
            if (j > 0) {
                maxn = Math.max(maxn, diff - (nums2[i] - rec[j - 1]));
            }
        }
        return (sum - maxn + MOD) % MOD;
    }

    public int binarySearch(int[] rec, int target) {
        int low = 0, high = rec.length - 1;
        // if (rec[high] < target) {
        //     return high + 1;
        // }
        while (low < high) {
            int mid = (high - low) / 2 + low;
            if (rec[mid] < target) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        return low;
    }

    public void sortColors(int[] nums) {
        int n = nums.length;
        int first = 0, last = n - 1;
        for (int i = 0; i <= last; ) {
            if (nums[i] == 0) {
                nums[i] = nums[first];
                nums[first] = 0;
                first++;
                i++;
            } else if (nums[i] == 1) {
                i++;
            } else {
                nums[i] = nums[last];
                nums[last] = 2;
                last--;
            }
        }
    }

    public void sortColors2(int[] nums) {
        quickSort(nums, 0, nums.length - 1);
    }

    private void quickSort(int[] nums, int start, int end) {
        if (start >= end) return;
        int l = start, r = end;
        while (l < r) {
            while (l < r && nums[r] >= nums[start]) r--;
            while (l < r && nums[l] <= nums[start]) l++;
            int tmp = nums[l];
            nums[l] = nums[r];
            nums[r] = tmp;
        }
        int tmp = nums[start];
        nums[start] = nums[l];
        nums[l] = tmp;
        quickSort(nums, start, l - 1);
        quickSort(nums, l + 1, end);
    }

    public List<List<Integer>> optionsGroup(int n, int k) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        dfs7(res, path, 1, n, n, k);
        return res;
    }

    private void dfs7(List<List<Integer>> res, List<Integer> path, int start, int sum, int n, int k) {
        if (path.size() == k) {
            if (sum == 0) {
                StringBuilder sb = new StringBuilder();
                for (int i = 0; i < path.size(); i++) {
                    sb.append(path.get(i));
                }
                System.out.println(sb.toString());
                res.add(new ArrayList<>(path));
            }
            return;
        }
        for (int i = start; i < n; i++) {
            if (i > sum) break;
            path.add(i);
            dfs7(res, path, i, sum - i, n, k);
            path.remove(path.size() - 1);
        }
    }

    public static void main(String[] args) {
        Lc8 lc8 = new Lc8();
//        char[][] tmp = new char[][]{{'+', '+', '.', '+'}, {'.', '.', '.', '+'}, {'+', '+', '+', '.'}};
//        lc8.nearestExit(tmp, new int[]{1, 2});
//        lc8.validPalindrome("abc");
//        lc8.countPalindromicSubsequence("bbcbaba");
//        int res = lc8.ladderLength("hit", "cog", Arrays.asList("hot", "dot", "dog", "lot", "log", "cog"));
//        System.out.println(res);
//        lc8.insert(new int[][]{{1, 3}, {6, 9}}, new int[]{2, 5});
        int[][] ma = new int[][]{{1, 3, 5, 7}, {10, 11, 16, 20}, {23, 30, 34, 60}};
//        lc8.searchMatrix(ma, 3);
//        new char[][]{{'1', '1'}, {'1', '0'}}
        char[][] pa = new char[][]{{'1', '1', '1', '1', '0'},
                {'1', '1', '0', '1', '0'},
                {'1', '1', '0', '0', '0'},
                {'0', '0', '0', '0', '0'}};
//        lc8.numIslands(pa);
//        int[] a = new int[]{0, 1, 3, 5, 6};
//        lc8.hIndex(a);

//        StringBuffer sb = new StringBuffer();
//        sb.append('a').append('1').append('2');
//        sb.deleteCharAt(sb.length()-1).deleteCharAt(sb.length()-1);
//        System.out.println(sb.toString());

        TreeNode root = new TreeNode(1);
//        root.left = new TreeNode(2,  new TreeNode(6), new TreeNode(5));
        root.left = new TreeNode(2);
        root.right = new TreeNode(3);
//        lc8.binaryTreePaths(root);
//        lc8.sumNumbers(root);

        int[][] b = new int[][]{{2, 9, 10}, {3, 7, 15}, {5, 12, 12}, {15, 20, 10}, {19, 24, 8}};
//        lc8.getSkyline(b);
//        lc8.minAbsoluteSumDiff(new int[]{1, 28, 21}, new int[]{9, 21, 20});
//        lc8.minAbsoluteSumDiff3(new int[]{5, 2, 7}, new int[]{10, 7, 12});

//        lc8.optionsGroup(8, 4);
        lc8.sortColors(new int[]{2, 0, 2, 1, 1, 0});

//        PriorityQueue<ListNode> pq = new PriorityQueue<>();
//        for(ListNode each:pq){
//            each.val--;
//        }
    }
}

class NumArray {
    int[] res;

    public NumArray(int[] nums) {
        res = nums;
    }

    public void update(int index, int val) {
        res[index] = val;
    }

    public int sumRange(int left, int right) {
        int sum = 0;
        for (int i = left; i <= right; i++) {
            sum += res[i];
        }
        return sum;
    }
}

class Solution {
    private boolean binarySearch(int[][] matrix, int target, int start, boolean vertical) {
        int lo = start;
        int hi = vertical ? matrix[0].length - 1 : matrix.length - 1;

        while (hi >= lo) {
            int mid = (lo + hi) / 2;
            if (vertical) { // searching a column
                if (matrix[start][mid] < target) {
                    lo = mid + 1;
                } else if (matrix[start][mid] > target) {
                    hi = mid - 1;
                } else {
                    return true;
                }
            } else { // searching a row
                if (matrix[mid][start] < target) {
                    lo = mid + 1;
                } else if (matrix[mid][start] > target) {
                    hi = mid - 1;
                } else {
                    return true;
                }
            }
        }

        return false;
    }

    public boolean searchMatrix(int[][] matrix, int target) {
        // an empty matrix obviously does not contain `target`
        if (matrix == null || matrix.length == 0) {
            return false;
        }

        // iterate over matrix diagonals
        int shorterDim = Math.min(matrix.length, matrix[0].length);
        for (int i = 0; i < shorterDim; i++) {
            boolean verticalFound = binarySearch(matrix, target, i, true);
            boolean horizontalFound = binarySearch(matrix, target, i, false);
            if (verticalFound || horizontalFound) {
                return true;
            }
        }

        return false;
    }
}

class Solution2 {
    Map<String, Integer> wordId = new HashMap<String, Integer>();
    List<List<Integer>> edge = new ArrayList<List<Integer>>();
    int nodeNum = 0;

    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        for (String word : wordList) {
            addEdge(word);
        }
        addEdge(beginWord);
        if (!wordId.containsKey(endWord)) {
            return 0;
        }
        int[] dis = new int[nodeNum];
        Arrays.fill(dis, Integer.MAX_VALUE);
        int beginId = wordId.get(beginWord), endId = wordId.get(endWord);
        dis[beginId] = 0;

        Queue<Integer> que = new LinkedList<Integer>();
        que.offer(beginId);
        while (!que.isEmpty()) {
            int x = que.poll();
            if (x == endId) {
                return dis[endId] / 2 + 1;
            }
            for (int it : edge.get(x)) {
                if (dis[it] == Integer.MAX_VALUE) {
                    dis[it] = dis[x] + 1;
                    que.offer(it);
                }
            }
        }
        return 0;
    }

    public void addEdge(String word) {
        addWord(word);
        int id1 = wordId.get(word);
        char[] array = word.toCharArray();
        int length = array.length;
        for (int i = 0; i < length; ++i) {
            char tmp = array[i];
            array[i] = '*';
            String newWord = new String(array);
            addWord(newWord);
            int id2 = wordId.get(newWord);
            edge.get(id1).add(id2);
            edge.get(id2).add(id1);
            array[i] = tmp;
        }
    }

    public void addWord(String word) {
        if (!wordId.containsKey(word)) {
            wordId.put(word, nodeNum++);
            edge.add(new ArrayList<Integer>());
        }
    }
}

class NumArray2 {
    // 参考 https://www.bilibili.com/video/av47331849/
    int[] treeNode;
    int[] nums;

    public NumArray2(int[] nums) {
        this.nums = nums;
        treeNode = new int[nums.length * 4];
        buildTree(0, 0, nums.length - 1);
    }

    public void buildTree(int curNode, int start, int end) {
        if (start == end) {
            treeNode[curNode] = nums[start];
            return;
        }

        int mid = start + ((end - start) >> 1);

        int leftNode = 2 * curNode + 1, rightNode = 2 * curNode + 2;
        buildTree(leftNode, start, mid);
        buildTree(rightNode, mid + 1, end);

        treeNode[curNode] = treeNode[leftNode] + treeNode[rightNode];
    }

    public int sumRange(int i, int j) {
        return sum(0, 0, nums.length - 1, i, j);
    }

    public int sum(int curNode, int start, int end, int L, int R) {
        if (R < start || L > end) {
            return 0;
        } else if (start >= L && end <= R) {
            return treeNode[curNode];
        } else if (start == end) {
            return treeNode[curNode];
        }

        int mid = start + ((end - start) >> 1);

        int leftNode = 2 * curNode + 1, rightNode = 2 * curNode + 2;
        int leftSum = sum(leftNode, start, mid, L, R);
        int rightSum = sum(rightNode, mid + 1, end, L, R);

        return leftSum + rightSum;
    }

    public void update(int i, int val) {
        updateHelper(0, 0, nums.length - 1, i, val);
    }

    public void updateHelper(int curNode, int start, int end, int index, int val) {
        if (start > end) return;
        if (start == end) {
            nums[index] = val;
            treeNode[curNode] = val;
            return;
        }

        int mid = start + ((end - start) >> 1);
        int leftNode = 2 * curNode + 1, rightNode = 2 * curNode + 2;
        if (index >= start && index <= mid) {
            updateHelper(leftNode, start, mid, index, val);
        } else {
            updateHelper(rightNode, mid + 1, end, index, val);
        }

        treeNode[curNode] = treeNode[leftNode] + treeNode[rightNode];
    }
}

