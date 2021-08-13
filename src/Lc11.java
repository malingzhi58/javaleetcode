import java.io.File;
import java.io.FileNotFoundException;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Lc11 {
    public static void main2() {
        List<String> list = new LinkedList<>();
        Scanner scan = null;
        try {
            scan = new Scanner(new File("/Users/qtt/IdeaProjects/demo/src/log.txt"));
            while (scan.hasNextLine()) {
                String tmp = scan.nextLine();
                Pattern pattern = Pattern.compile("qtt:map\\[_active\\_time\\_:(.*?) \\_exist\\_:true");
                Matcher matcher = pattern.matcher(tmp);
                String tmpRes = "";
                if (matcher.find()) {
                    tmpRes = matcher.group(1);
                }
                String[] t1 = tmpRes.split("\\.");
                String t2 = t1[0] + t1[1];
                String[] t3 = t2.split("e");
                StringBuilder sb = new StringBuilder(t3[0]);
                while (sb.length() < 10) {
                    sb.append('0');
                }
                String formats = "yyyy-MM-dd HH:mm:ss";
                long timestamp = Long.parseLong(sb.toString()) * 1000;
                String date = new SimpleDateFormat(formats, Locale.CHINA).format(new Date(timestamp));
                System.out.println(date);

            }


        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

    }

    public String smallestSubsequence(String s) {
        Stack<Character> stack = new Stack<>();
        Map<Character, Integer> map = new HashMap<>();
        for (int i = 0; i < s.length(); i++) {
            map.put(s.charAt(i), map.getOrDefault(s.charAt(i), 0) + 1);
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            while (!stack.isEmpty() && map.get(stack.peek()) > 0 && stack.peek() >= s.charAt(i)) {
                stack.pop();
            }
            if (!stack.contains(s.charAt(i))) {

                stack.push(s.charAt(i));
            }
            map.put(s.charAt(i), map.get(s.charAt(i)) - 1);
        }
        while (!stack.isEmpty()) {
            sb.append(stack.pop());
        }
        sb.reverse();
        return sb.toString();
    }


    int max = 0;

    public int maxCompatibilitySum(int[][] stu, int[][] men) {
        int row = stu.length, col = stu[0].length;
        boolean[] visited = new boolean[row];
        int[][] score = new int[row][row];
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < row; j++) {
                score[i][j] = computeScore(stu[i], men[j]);
            }
        }
        dfs(visited, score, 0, 0);
        return max;

    }

    // cur means the current student, each student will loop through each mentor
    // visited means mentor
    private void dfs(boolean[] visited, int[][] score, int cur, int sum) {
        if (cur == visited.length) {
            max = Math.max(max, sum);
            return;
        }

        for (int i = 0; i < visited.length; i++) {
            if (visited[i]) continue;
            visited[i] = true;
            dfs(visited, score, cur + 1, sum + score[cur][i]);
            visited[i] = false;
        }
    }

    private int computeScore(int[] ints, int[] men) {
        int res = 0;
        for (int i = 0; i < ints.length; i++) {
            if (ints[i] == men[i]) {
                res++;
            }
        }
        return res;
    }

    public List<List<Integer>> getSkyline(int[][] buildings) {
        List<List<Integer>> res = new ArrayList<>();
        PriorityQueue<Integer> pq = new PriorityQueue<>((a, b) -> (b - a));
        List<int[]> sortB = new ArrayList<>();
        for (int i = 0; i < buildings.length; i++) {
            sortB.add(new int[]{buildings[i][0], -buildings[i][2]});
            sortB.add(new int[]{buildings[i][1], buildings[i][2]});
        }
        Collections.sort(sortB, (a, b) -> {
                    if (a[0] != b[0]) return a[0] - b[0];
                    else return a[1] - b[1];
                }
        );
        int pre = 0;
        pq.add(0);
        for (int[] each : sortB) {
            if (each[1] < 0) {
                pq.add(-each[1]);
            } else {
                pq.remove(each[1]);
            }
            int cur = pq.peek();
            if (cur != pre) {
                res.add(new ArrayList<>(Arrays.asList(each[0], cur)));
                pre = cur;
            }
        }
        return res;
    }

    List<Integer> res = new ArrayList<>();

    public List<Integer> pathInZigZagTree(int label) {
        List<Integer> path = new ArrayList<>();
        if (label < 1) return path;
        TreeNode root = makeTree(label);
        dfs2(root, label, path);
        return res;
    }

    private boolean dfs2(TreeNode root, int label, List<Integer> path) {
        if (root == null) return false;
        path.add(root.val);
        if (root.val == label) {
            res = new ArrayList<>(path);
            return true;
        }
        if (dfs2(root.left, label, path)) {
            return true;
        }
        if (dfs2(root.right, label, path)) {
            return true;
        }
        path.remove(path.size() - 1);
        return false;
    }

    private TreeNode makeTree(int label) {
        Deque<TreeNode> queue = new LinkedList<>();
        TreeNode root = new TreeNode(1);
        queue.offer(root);
        int depth = 0, cur = 2;
        while (!queue.isEmpty()) {
            int size = queue.size();
            depth++;
            while (size-- > 0) {
                if ((depth & 1) == 0) {
                    TreeNode current = queue.removeLast();
                    current.left = new TreeNode(cur);
                    cur++;
                    queue.addFirst(current.left);
                    current.right = new TreeNode(cur);
                    cur++;
                    queue.addFirst(current.right);
                } else {
                    TreeNode current = queue.removeFirst();
                    current.right = new TreeNode(cur);
                    queue.addLast(current.right);
                    cur++;
                    current.left = new TreeNode(cur);
                    queue.addLast(current.left);
                    cur++;
                }
            }
            if (cur > label) break;

        }
        return root;
    }

//    public List<List<Long>> splitPainting(int[][] segments) {
//        List<List<Long>> res = new ArrayList<>();
//        TreeMap<Integer,Long> map = new TreeMap<>();
//        for(int[]each:segments){
//            map.put(each[0],map.getOrDefault (each[0],0L)+(long)each[2]);
//            map.put(each[1],map.getOrDefault (each[1],0L)-(long)each[2]);
//        }
//        long sum =0;
//        int pre =0;
//        boolean start = false;
//        for(Map.Entry<Integer,Long>entry:map.entrySet()){
//            if(sum!=0){
//                res.add(new ArrayList<>(Arrays.asList((long)pre,(long)entry.getKey(),sum)));
//            }
//            sum+=entry.getValue();
//            pre = entry.getKey();
//        }
//        return res;
//    }

    public List<List<Long>> splitPainting(int[][] segments) {
        long[] diff = new long[100001];
        Set<Integer> set = new HashSet<>();
        for (int[] s : segments) {
            diff[s[0]] += s[2];
            diff[s[1]] -= s[2];
            set.add(s[0]);
        }
        long[] res = new long[100002];
        for (int i = 0; i < diff.length; i++) {
            long d = diff[i];
            res[i + 1] = res[i];
            if (d != 0) {
                res[i + 1] += d;
            }
        }
        List<List<Long>> result = new ArrayList<>();
        int pre = 0;
        for (int i = 0; i < res.length; i++) {
            List<Long> tempList = new ArrayList<>();
            if (res[i] != 0) {
                tempList.add((long) i - 1);
                long temp = res[i];
                while (res[i + 1] == temp && !set.contains(i)) i++;
                tempList.add((long) i);
                tempList.add(temp);
                result.add(tempList);
            }
        }
        return result;
    }

//    public int[] canSeePersonsCount(int[] heights) {
//        int len = heights.length;
//        int[] res = new int[len];
//        for (int i = 0; i < len; i++) {
//            int tmp = 0;
//            for (int j = i + 1; j < len; j++) {
//                int min = Math.min(heights[i], heights[j]);
//                boolean add = true;
//                for (int k = i + 1; k < j - 1; k++) {
//                    if (min <= heights[k]) {
//                        add = false;
//                        break;
//                    }
//                }
//                if (add) tmp++;
//                else break;
//            }
//            res[i] = tmp;
//        }
//        return res;
//    }

    public int[] canSeePersonsCount(int[] heights) {
        int len = heights.length;
        int[] res = new int[len];
        Stack<Integer> stack = new Stack<>();
        Stack<Integer> tmpstack = new Stack<>();
        stack.push(heights[len - 1]);
        for (int i = len - 2; i >= 0; i--) {
            int cur = 0;
            tmpstack = new Stack<>();
            while (!stack.isEmpty()) {
                if (stack.peek() > heights[i]) {
                    cur++;
                    break;
                } else {
                    tmpstack.push(stack.pop());
                    cur++;
                }

            }
            res[i] = cur;
            while (!tmpstack.isEmpty() && tmpstack.peek() > heights[i]) {
                stack.push(tmpstack.pop());
            }
            stack.push(heights[i]);
        }
        return res;
    }

//    int res3 = Integer.MAX_VALUE, partition = 1;
//
//    public int minCost(int[] houses, int[][] cost, int m, int n, int target) {
////        int[] colo
//        int minCost = 0;
//        dfs(houses, cost, m, n, target, minCost, 0);
//        return res3 == Integer.MAX_VALUE ? -1 : res3;
//    }
//
//    private void dfs(int[] houses, int[][] cost, int m, int n, int target, int minCost, int start) {
//        if (partition > target) return;
//        if (start == m) {
//            if (target == partition) {
//                res3 = Math.min(res3, minCost);
//            }
//            return;
//        }
//        if (houses[start] == 0) {
//            for (int j = 0; j < n; j++) {
//                houses[start] = j + 1;
//                if (start > 0 && houses[start - 1] != houses[start]) {
//                    partition++;
//                }
//                dfs(houses, cost, m, n, target, minCost + cost[start][j], start + 1);
//                if (start > 0 && houses[start - 1] != houses[start]) {
//                    partition--;
//                }
//                houses[start] = 0;
//            }
//        } else {
//            if (start > 0 && houses[start - 1] != houses[start]) {
//                partition++;
//            }
//            dfs(houses, cost, m, n, target, minCost, start + 1);
//            if (start > 0 && houses[start - 1] != houses[start]) {
//                partition--;
//            }
//        }
//    }

    public int titleToNumber(String columnTitle) {
        int sum = 0;
        for (int i = columnTitle.length() - 1; i >= 0; i--) {
            char cur = columnTitle.charAt(i);
            int count = 0;
            count = (cur - 'A' + 1) * (int) Math.pow(26, columnTitle.length() - i - 1);

            sum += count;
        }
        return sum;
    }

    public int minCost(int[][] costs) {
        // write your code here
        int row = costs.length, col = costs[0].length;
        int[] dp = new int[col];
        int[] dpnew = new int[col];
        dp = Arrays.copyOf(costs[0], col);
//        int[] minRight =new int[col];
//        int[] minLeft =new int[col];

        for (int i = 1; i < row; i++) {
            for (int j = 0; j < col; j++) {
                dpnew[j] = Integer.MAX_VALUE;
                for (int k = 0; k < col; k++) {
                    if (j != k) {
                        dpnew[j] = Math.min(dpnew[j], dp[k] + costs[i][j]);
                    }
                }
            }
            dp = dpnew.clone();
        }
        int min = Integer.MAX_VALUE;
        for (int i = 0; i < col; i++) {
            min = Math.min(min, dp[i]);
        }
        return min;

    }

    public int[] relativeSortArray(int[] arr1, int[] arr2) {
        List<Integer> notin = new ArrayList<>();
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < arr2.length; i++) {
            map.put(arr2[i], 0);
        }
        for (int i = 0; i < arr1.length; i++) {
            if (!map.containsKey(arr1[i])) {
                notin.add(arr1[i]);
            } else {
                map.put(arr1[i], map.get(arr1[i]) + 1);
            }
        }
        Collections.sort(notin);
        int[] res = new int[arr1.length];
        int id = 0;
        for (int i = 0; i < arr2.length; i++) {
            for (int j = 0; j < map.get(arr2[i]); j++) {
                res[id++] = arr2[i];
            }
        }
        for (int i = 0; i < notin.size(); i++) {
            res[id++] = notin.get(i);
        }
        return res;
    }

    int INF = 0x3f3f3f3f;

    public int minCost(int[] houses, int[][] cost, int m, int n, int target) {
        int[][][] f = new int[m + 1][n + 1][target + 1];

        for (int i = 0; i <= m; i++) {
            for (int j = 0; j <= n; j++) {
                f[i][j][0] = INF;
            }
        }

        for (int i = 1; i <= m; i++) {
            int color = houses[i - 1];
            if (color == 0) {
                for (int j = 1; j <= n; j++) {
                    for (int k = 1; k <= target; k++) {
                        if (k > i) {
                            f[i][j][k] = INF;
                            continue;
                        }
                        int min = INF;
                        for (int l = 1; l <= n; l++) {
                            if (l != j) {
                                min = Math.min(min, f[i - 1][l][k - 1]);
                            }
                        }
                        f[i][j][k] = Math.min(f[i - 1][j][k], min) + cost[i - 1][j - 1];
                    }
                }
            } else {
                for (int j = 1; j <= n; j++) {
                    if (j != color - 1) {
                        for (int k = 1; k <= target; k++) {
                            f[i][j][k] = INF;
                        }
                    } else {
                        for (int k = 1; k <= target; k++) {
                            if (k > i) {
                                f[i][j][k] = INF;
                                continue;
                            }
                            int min = INF;
                            for (int l = 1; l <= n; l++) {
                                if (l != j) {
                                    min = Math.min(min, f[i - 1][l][k - 1]);
                                }
                            }
                            f[i][j][k] = Math.min(f[i - 1][j][k], min);
                        }
                    }
                }
            }
        }
        int min = INF;
        for (int i = 1; i <= n; i++) {
            min = Math.min(f[m][i][target], min);
        }
        return min == INF ? -1 : min;
    }

    public int wiggleMaxLength(int[] nums) {
        int[] dp = new int[nums.length];
        dp[0] = 1;
        int pre = 0;
        // dp[i] = dp[i-1]+1 if pre*cur<0
        // dp[i] =dp[i-1] else
        for (int i = 1; i < nums.length; i++) {
            int cur = nums[i] - nums[i - 1];
            if (cur * pre < 0 || pre == 0 && cur != 0) {
                dp[i] = dp[i - 1] + 1;
            } else {
                dp[i] = dp[i - 1];
            }
            pre = cur;
        }
        return dp[nums.length - 1];
    }

    //    public int wiggleMaxLength(int[] nums) {
////        int[] dp = new int[nums.length];
////        dp[0] = 1;
//        if (nums == null || nums.length <= 1) {
//            return nums.length;
//        }
//        int count =1;
//        int pre = 0;
//        for (int i = 1; i < nums.length; i++) {
//            int cur = nums[i] - nums[i - 1];
//            if (cur * pre < 0 || pre == 0 && cur != 0) {
//                count++;
//            }
//            pre = cur;
//        }
//        return count;
//    }
    int leftlen=0,rightlen=0;
    public List<List<Integer>> verticalTraversal(TreeNode root) {
        computeLen(root,0);
        List<Node2>[] list = new ArrayList[rightlen-leftlen+1];
        for (int i = 0; i < list.length; i++) {
            list[i]=new ArrayList<>();
        }
        recur(list,root,0,0,-leftlen);
        List<List<Integer>> res = new ArrayList<>();
        for (int i = 0; i < list.length; i++) {
            Collections.sort(list[i]);
            List<Integer> tmp = new ArrayList<>();
            for (Node2 each:list[i]) {
                tmp.add(each.val);
            }
            res.add(tmp);
        }
        return res;
    }
    private void computeLen(TreeNode root,int offset) {
        if(root==null) return;
        leftlen = Math.min(leftlen,offset);
        rightlen = Math.max(rightlen,offset);
        computeLen(root.left,offset-1);
        computeLen(root.right,offset+1);
    }

    private void recur(List<Node2>[] list, TreeNode root, int depth, int offset, int base) {
        if(root==null)return;
        Node2 cur = new Node2(root.val,depth, offset);
        list[base+offset].add(cur);
        recur(list,root.left,depth+1,offset-1,base);
        recur(list,root.right,depth+1,offset+1,base);
    }

    class Node2 extends TreeNode implements Comparable<Node2>{
        int depth,offset;

        public Node2(int depth, int offset) {
            this.depth = depth;
            this.offset = offset;
        }

        public Node2(int val, int depth, int offset) {
            super(val);
            this.depth = depth;
            this.offset = offset;
        }

        public Node2(int val, TreeNode left, TreeNode right, int depth, int offset) {
            super(val, left, right);
            this.depth = depth;
            this.offset = offset;
        }



        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            Node2 node2 = (Node2) o;
            return depth == node2.depth && offset == node2.offset;
        }

        @Override
        public int hashCode() {
            return Objects.hash(depth, offset);
        }

        @Override
        public int compareTo(Node2 o) {
            return this.depth!=o.depth?this.depth-o.depth:this.val-o.val;
        }
//        @Override
//        public int compare(Node2 o1, Node2 o2) {
//            return o1.depth!=o2.depth?o1.depth-o2.depth:o1.val-o2.val;
//        }
    }



    public static void main(String[] args) {
        Lc11 lc11 = new Lc11();
        int[][] s = {{1, 1, 0}, {1, 0, 1}, {0, 0, 1}};
        int[][] m = {{1, 0, 0}, {0, 0, 1}, {1, 1, 0}};
//        lc11.maxCompatibilitySum(s, m);
        int[][] sk = {{2, 9, 10}, {3, 7, 15}, {5, 12, 12}, {15, 20, 10}, {19, 24, 8}};
//        lc11.getSkyline(sk);
//        lc11.pathInZigZagTree(14);
        int[][] s2 = {{1, 4, 5}, {4, 7, 7}, {1, 7, 9}};
//        lc11.splitPainting(s2);
        int[] s3 = {10, 6, 8, 5, 11, 9};
//        lc11.canSeePersonsCount(s3);
        int[] s4 = {0, 0, 0, 0, 0};
        int[][] s5 = {{1, 10}, {10, 1}, {10, 1}, {1, 10}, {5, 1}};
//        lc11.minCost(s4, s5, 5, 2, 3);

        int[] s6 = {2, 3, 1, 3, 2, 4, 6, 7, 9, 2, 19};
        int[] s7 = {2, 1, 4, 3, 9, 6};
//        lc11.relativeSortArray(s6,s7);
//        PriorityQueue<int[]> pq = new PriorityQueue<>();
//        pq.add(new int[]{1,2,3});
//        pq.peek()[2]=5;
//        System.out.println(pq.peek()[2]);
        LFUCache lfuCache = new LFUCache(2);
//        lfuCache.put(1, 1);
//        lfuCache.put(2, 2);
//        lfuCache.get(1);
//        lfuCache.put(3, 3);
//        lfuCache.get(2);
//        lfuCache.put(4, 4);
//        lfuCache.get(3);
////        int i = lfuCache.get(4);
////        System.out.println(i);
//        lfuCache.put(3, 1);
//        lfuCache.put(2, 1);
//        lfuCache.put(2, 2);
//        lfuCache.put(4, 4);
//        System.out.println(lfuCache.get(2));

//        lc11.wiggleMaxLength(new int[]{1, 17, 5, 10, 13, 15, 10, 5, 16, 8});
        TreeNode root = new TreeNode(3,new TreeNode(9),new TreeNode(20,new TreeNode(15),new TreeNode(7)));
        lc11.verticalTraversal(root);

    }
}


class LFUCache {
    // 缓存容量，时间戳
    int capacity, time;
    Map<Integer, Node> key_table;
    TreeSet<Node> S;
    TreeMap<Integer,Integer> N;
    public LFUCache(int capacity) {
        this.capacity = capacity;
        this.time = 0;
        key_table = new HashMap<Integer, Node>();
        S = new TreeSet<Node>();
    }

    public int get(int key) {
        // 如果哈希表中没有键 key，返回 -1
        if (!key_table.containsKey(key)) {
            return -1;
        }
        // 从哈希表中得到旧的缓存
        Node cache = key_table.get(key);
        // 从平衡二叉树中删除旧的缓存
        S.remove(cache);
        // 将旧缓存更新
        cache.cnt += 1;
        cache.time = ++time;
        // 将新缓存重新放入哈希表和平衡二叉树中
        S.add(cache);
        key_table.put(key, cache);
        return cache.value;
    }

    public void put(int key, int value) {
        if (capacity == 0) {
            return;
        }
        if (!key_table.containsKey(key)) {
            // 如果到达缓存容量上限
            if (key_table.size() == capacity) {
                // 从哈希表和平衡二叉树中删除最近最少使用的缓存
                key_table.remove(S.first().key);
                S.remove(S.first());
            }
            // 创建新的缓存
            Node cache = new Node(1, ++time, key, value);
            // 将新缓存放入哈希表和平衡二叉树中
            key_table.put(key, cache);
            S.add(cache);
        } else {
            // 这里和 get() 函数类似
            Node cache = key_table.get(key);
            S.remove(cache);
            cache.cnt += 1;
            cache.time = ++time;
            cache.value = value;
            S.add(cache);
            key_table.put(key, cache);
        }
    }

    static class Node implements Comparable<Node> {
        int cnt, time, key, value;

        Node(int cnt, int time, int key, int value) {
            this.cnt = cnt;
            this.time = time;
            this.key = key;
            this.value = value;
        }

        public boolean equals(Object anObject) {
            if (this == anObject) {
                return true;
            }
            if (anObject instanceof Node) {
                Node rhs = (Node) anObject;
                return this.cnt == rhs.cnt && this.time == rhs.time;
            }
            return false;
        }

        public int compareTo(Node rhs) {
            return cnt == rhs.cnt ? time - rhs.time : cnt - rhs.cnt;
        }

        public int hashCode() {
            return cnt * 1000000007 + time;
        }
    }
}


//class LFUCache {
//    //    1. key 2. freq 3.time
//    PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> {
//        if (a[1] != b[1]) {
//            return a[1] - b[1];
//        } else {
//            return a[2] - b[2];
//        }
//    });
//    TreeMap<Integer, Integer> map = new TreeMap<>();
//    int limit = 0;
//    int time = 0;
//
//    public LFUCache(int capacity) {
//        limit = capacity;
//    }
//
//    public int get(int key) {
//        if (!map.containsKey(key)) {
//            return -1;
//        }
//        time++;
//        List<int[]> tmp = new ArrayList<>();
//        while (!pq.isEmpty()) {
//            int[] each = pq.poll();
//            if (each[0] == key) {
//                each[1]++;
//                each[2] = time;
//            }
//            tmp.add(each);
//        }
//        pq.addAll(tmp);
//        return map.get(key);
//    }
//
//    public void put(int key, int value) {
//        time++;
////        if(pq.size()<limit){
////
////        }
//        List<int[]> tmp = new ArrayList<>();
//        boolean find = false;
//        int[] in = new int[3];
//        while (!pq.isEmpty()) {
//            int[] cur = pq.poll();
//            if (cur[0] == key) {
//                in = cur;
//                find = true;
//            } else {
//                tmp.add(cur);
//            }
//        }
//        map.put(key, value);
//        pq.addAll(tmp);
//        if (find) {
//            pq.add(new int[]{key, in[1] + 1, time});
//        } else {
//            if (pq.size() == limit) {
//                int[] toberemoved = pq.poll();
//                map.remove(toberemoved[0]);
//            }
//            pq.add(new int[]{key, 1, time});
//        }
//    }
//}


//class Solution {
//    boolean[] f; // 标记数组
//    int ans = 0;
//    int[][] t; // 兼容评分
//    int m, n;
//    public int maxCompatibilitySum(int[][] stu, int[][] men) {
//        m = stu.length;
//        n = stu[0].length;
//
//        f = new boolean[m];
//        t = new int[m][m];
//
//        // 预处理兼容和
//        for(int i=0; i<m; i++){
//            for(int j=0; j<m; j++){
//                t[i][j] = get(stu[i], men[j]);
//            }
//        }
//        dfs(0, 0, 0);
//        return ans;
//    }
//
//    public void dfs(int idx, int s, int sum){
//        if(s == m){
//            ans = Math.max(ans, sum);
//            return;
//        }
//        for(int i=0; i<m; i++){
//            if(f[i]) continue;
//            f[i] = true;
//            dfs(idx + 1, s + 1, sum + t[idx][i]);
//            f[i] = false;
//        }
//    }
//
//    // 返回两数组的兼容和
//    public int get(int[] a, int[] b){
//        int ret = 0;
//        for(int i=0; i<a.length; i++){
//            if(a[i] == b[i]) ret++;
//        }
//        return ret;
//    }
//}

//class Solution {
//    private int ans = 0;
//    private int m;
//    private int n;
//    private int[][] covs;
//
//    public int maxCompatibilitySum(int[][] students, int[][] mentors) {
//        m = students.length;
//        n = students[0].length;
//        int[] stu = new int[m], men = new int[m];
//        for (int i = 0; i < m; i++) {
//            stu[i] = cal(students[i], n);
//            men[i] = cal(mentors[i], n);
//        }
//        covs = new int[m][m];
//        for (int i = 0; i < m; i++) {
//            for (int j = 0; j < m; j++) {
//                covs[i][j] = cov(stu[i], men[j], n);
//            }
//        }
//        get(stu, men, 0, 0);
//        return ans;
//    }
//
//    private void get(int[] stu, int[] men, int stu_id, int point) {
//        if (stu_id == m) {
//            ans = Math.max(ans, point);
//            return;
//        }
//        for (int i = 0; i < m; i++) {
//            if (men[i] == -1) continue;
//            int p = covs[stu_id][i];
//            int t = men[i];
//            men[i] = -1;
//            get(stu, men, stu_id + 1, point + p);
//            men[i] = t;
//        }
//    }
//
//    private int cov(int a, int b, int n) {
//        int s = a ^ b;
//        return n - Integer.bitCount(s);
//    }
//
//    private int cal(int[] p, int n) {
//        int sum = 0;
//        for (int i = 0; i < n; i++) {
//            sum = (sum << 1) + p[i];
//        }
//        return sum;
//    }
//}


//class Solution {
//    public List<List<Long>> splitPainting(int[][] segments) {
//        TreeMap<Integer, Long> i_diff = new TreeMap<>();
//        for (int[] s : segments) {
//            int l = s[0], r = s[1], c = s[2];
//            i_diff.put(l, i_diff.getOrDefault(l, 0L) + (long) c);
//            i_diff.put(r, i_diff.getOrDefault(r, 0L) - (long) c);
//        }
//
//        List<List<Long>> res = new ArrayList<>();
//        Long last_i = 0L;             //区间的左端点
//        Long cur_color = 0L;       //区间的颜色（就是个大数值）
//        for (Integer i : i_diff.keySet()) {
//
//            if (cur_color != 0) {
//                List<Long> tmp = new ArrayList<>();
//                tmp.add(last_i);
//                tmp.add((long) i);
//                tmp.add(cur_color);
//                res.add(tmp);
//            }
//            Long diff = i_diff.get(i);
//            last_i = (long) i;
//            cur_color += diff;
//        }
//
//        return res;
//    }
//}


//class Solution {
//    int INF = 0x3f3f3f3f;
//    public int minCost(int[] hs, int[][] cost, int m, int n, int t) {
//        int[][][] f = new int[m + 1][n + 1][t + 1];
//
//        // 不存在分区数量为 0 的状态
//        for (int i = 0; i <= m; i++) {
//            for (int j = 0; j <= n; j++) {
//                f[i][j][0] = INF;
//            }
//        }
//
//        for (int i = 1; i <= m; i++) {
//            int color = hs[i - 1];
//            for (int j = 1; j <= n; j++) {
//                for (int k = 1; k <= t; k++) {
//                    // 形成分区数量大于房子数量，状态无效
//                    if (k > i) {
//                        f[i][j][k] = INF;
//                        continue;
//                    }
//
//                    // 第 i 间房间已经上色
//                    if (color != 0) {
//                        if (j == color) { // 只有与「本来的颜色」相同的状态才允许被转移
//                            int tmp = INF;
//                            // 先从所有「第 i 间房形成新分区」方案中选最优（即与上一房间颜色不同）
//                            for (int p = 1; p <= n; p++) {
//                                if (p != j) {
//                                    tmp = Math.min(tmp, f[i - 1][p][k - 1]);
//                                }
//                            }
//                            // 再结合「第 i 间房不形成新分区」方案中选最优（即与上一房间颜色相同）
//                            f[i][j][k] = Math.min(f[i - 1][j][k], tmp);
//
//                        } else { // 其余状态无效
//                            f[i][j][k] = INF;
//                        }
//
//                        // 第 i 间房间尚未上色
//                    } else {
//                        int u = cost[i - 1][j - 1];
//                        int tmp = INF;
//                        // 先从所有「第 i 间房形成新分区」方案中选最优（即与上一房间颜色不同）
//                        for (int p = 1; p <= n; p++) {
//                            if (p != j) {
//                                tmp = Math.min(tmp, f[i - 1][p][k - 1]);
//                            }
//                        }
//                        // 再结合「第 i 间房不形成新分区」方案中选最优（即与上一房间颜色相同）
//                        // 并将「上色成本」添加进去
//                        f[i][j][k] = Math.min(tmp, f[i - 1][j][k]) + u;
//                    }
//                }
//            }
//        }
//
//        // 从「考虑所有房间，并且形成分区数量为 t」的所有方案中找答案
//        int ans = INF;
//        for (int i = 1; i <= n; i++) {
//            ans = Math.min(ans, f[m][i][t]);
//        }
//        return ans == INF ? -1 : ans;
//    }
//}