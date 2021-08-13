import java.util.*;

public class Lc10 {
    public boolean findWhetherExistsPath2(int n, int[][] graph, int start, int target) {
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int i = 0; i < graph.length; i++) {
            int s = graph[i][0];
            int e = graph[i][1];
            if (map.containsKey(s)) {
                map.get(s).add(e);
            } else {
                map.put(s, new ArrayList<>(Arrays.asList(e)));
            }
        }
        Set<Integer> set = new HashSet<>();
        Queue<Integer> queue = new LinkedList<>();
        queue.offer(start);
        while (!queue.isEmpty()) {
            int cur = queue.poll();
            if (!map.containsKey(cur)) continue;
            List<Integer> val = map.get(cur);
            for (Integer each : val) {
                if (target == each) return true;
                if (set.contains(each)) {
                    continue;
                } else {
                    set.add(each);
                    queue.offer(each);
                }
            }
        }
        return false;
    }

    public boolean findWhetherExistsPath(int n, int[][] graph, int start, int target) {
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int i = 0; i < graph.length; i++) {
            int s = graph[i][0];
            int e = graph[i][1];
            if (map.containsKey(s)) {
                map.get(s).add(e);
            } else {
                map.put(s, new ArrayList<>(Arrays.asList(e)));
            }
        }
        Set<Integer> set = new HashSet<>();
        Queue<Integer> startQueue = new LinkedList<>();
        Queue<Integer> endQueue = new LinkedList<>();
        startQueue.offer(start);
        endQueue.offer(target);
        while (!startQueue.isEmpty() && !endQueue.isEmpty()) {
            if (startQueue.size() > endQueue.size()) {
                Queue<Integer> tmp = startQueue;
                startQueue = endQueue;
                endQueue = tmp;
            }
            for (int i = 0; i < startQueue.size(); i++) {
                int cur = startQueue.poll();
                if (endQueue.contains(cur)) {
                    return true;
                }
                if (set.contains(cur)) {
                    continue;
                } else {
                    set.add(cur);
                    if (map.containsKey(cur)) {
                        List<Integer> list = map.get(cur);
                        for (Integer each : list) {
                            startQueue.offer(each);
                        }
                    }
                }
            }

        }
        return false;
    }

    public int canBeTypedWords(String text, String brokenLetters) {
        String[] list = text.split(" ");
        Map<String, Boolean> map = new HashMap<>();
        int count = 0;
        for (String each : list) {
            if (map.containsKey(each)) {
                if (map.get(each) == true) {
                    count++;
                }
            } else {
                boolean suc = true;
                for (Character ch : brokenLetters.toCharArray()) {
                    if (each.indexOf(ch) != -1) {
                        map.put(each, false);
                        suc = false;
                        break;
                    }
                }
                if (suc) {
                    map.put(each, true);
                    count++;
                }
            }
        }
        return count;
    }

    public int addRungs(int[] rungs, int dist) {
        int count = 0;
        int max = 0;
        for (Integer each : rungs) {
            if (max + dist >= each) {
                max = each;
            } else {
//                while(max+dist<each){
//                    max+=dist;
//                    count++;
//                }

                count += (each - max) / dist;
                max = each;
            }
        }
        return count;
    }

    public long maxPoints2(int[][] points) {
        int row = points.length, col = points[0].length;
        long[][] dp = new long[row][col];
        long[][] tmp = new long[col][col];

        for (int i = 0; i < col; i++) {
            dp[0][i] = points[0][i];
        }

        for (int i = 1; i < row; i++) {
            for (int j = 0; j < col; j++) {
                for (int k = 0; k < col; k++) {
                    dp[i][j] = Math.max(dp[i][j], dp[i - 1][k] - Math.abs(k - j) + points[i][j]);
                    tmp[j][k] = dp[i - 1][k] - Math.abs(k - j) + points[i][j];
                }
            }
        }
        long max = Long.MIN_VALUE;
        for (int i = 0; i < col; i++) {
            max = Math.max(max, dp[row - 1][i]);
        }
        return max;
//        return curmaxdp==Long.MIN_VALUE?0:curmaxdp;
    }

    public long maxPoints(int[][] points) {
        int row = points.length, col = points[0].length;
        long[][] dp = new long[row][col];
        long oldmaxdp = Long.MIN_VALUE, curmaxdp = oldmaxdp;
        int oldmaxdpindex = -1, curmaxdpindex = oldmaxdpindex;
        for (int i = 0; i < col; i++) {
            dp[0][i] = points[0][i];
            if (dp[0][i] > oldmaxdp) {
                oldmaxdp = dp[0][i];
                oldmaxdpindex = i;
            }
        }

        for (int i = 1; i < row; i++) {
            curmaxdp = Long.MIN_VALUE;
            for (int j = 0; j < col; j++) {
                if (j > oldmaxdpindex) {
                    for (int k = oldmaxdpindex; k < col; k++) {
                        dp[i][j] = Math.max(dp[i][j], dp[i - 1][k] - Math.abs(k - j) + points[i][j]);
                        if (dp[i][j] > curmaxdp) {
                            curmaxdp = dp[i][j];
                            curmaxdpindex = j;
                        }
                    }
                } else if (j < oldmaxdpindex) {
                    for (int k = 0; k <= oldmaxdpindex; k++) {
                        dp[i][j] = Math.max(dp[i][j], dp[i - 1][k] - Math.abs(k - j) + points[i][j]);
                        if (dp[i][j] > curmaxdp) {
                            curmaxdp = dp[i][j];
                            curmaxdpindex = j;
                        }
                    }
                } else {
                    dp[i][j] = Math.max(dp[i][j], dp[i - 1][oldmaxdpindex] + points[i][j]);
                    if (dp[i][j] > curmaxdp) {
                        curmaxdp = dp[i][j];
                        curmaxdpindex = j;
                    }
                }
            }
            oldmaxdp = curmaxdp;
            oldmaxdpindex = curmaxdpindex;
        }
        long max = Long.MIN_VALUE;
        for (int i = 0; i < col; i++) {
            max = Math.max(max, dp[row - 1][i]);
        }
        return max;
//        return curmaxdp==Long.MIN_VALUE?0:curmaxdp;
    }

    public TreeNode sortedArrayToBST(int[] nums) {
        TreeNode root = toBst(nums, 0, nums.length - 1);
        return root;
    }

    private TreeNode toBst(int[] nums, int start, int end) {
        if (start >= end) return null;
        int mid = (start + end) / 2;
        TreeNode root = new TreeNode(nums[mid]);
        root.left = toBst(nums, start, mid - 1);
        root.right = toBst(nums, mid + 1, end);
        return root;
    }

    public long maxPoints3(int[][] points) {
        int m = points.length;
        int n = points[0].length;
        long[][] f = new long[m][n];
        for (int j = 0; j < n; ++j) {
            f[0][j] = points[0][j];
        }
        for (int i = 1; i < m; ++i) {
            long ret = f[i - 1][0];
            int retIndex = 0;
            for (int j = 1; j < n; ++j) {
                if (f[i - 1][j] > ret) {
                    ret = f[i - 1][j];
                    retIndex = j;
                }
            }
            for (int j = 0; j < n; j++) {
                if (retIndex <= j) {
                    f[i][j] = points[i][j] - j + ret + retIndex;
                } else {
                    f[i][j] = points[i][j] + j + ret - retIndex;

                }
            }
//            ret = f[i - 1][n - 1] - (n - 1);
//            for (int j = n - 2; j >= 0; --j) {
//                f[i][j] = Math.max(f[i][j], points[i][j] + j + ret);
//                ret = Math.max(ret, f[i - 1][j] - j);
//            }
        }
        long ans = 0;
        for (int j = 0; j < n; ++j) {
            ans = Math.max(ans, f[m - 1][j]);
        }
        return ans;
    }

//    public long maxPoints(int[][] points) {
//        int m = points.length;
//        int n = points[0].length;
//        long[][] dp = new long[m][n];
//        for (int j = 0; j < n; ++j) {
//            dp[0][j] = points[0][j];
//        }
//        long[] maxLeft = new long[n];
//        long[] maxRight = new long[n];
//        for (int i = 1; i < m; ++i) {
//            maxLeft[0] = dp[i - 1][0];
//            for (int j = 1; j < n; j++) {
//                maxLeft[j] = Math.max(dp[i - 1][j] + j, maxLeft[j - 1]);
//            }
//            maxRight[n - 1] = dp[i - 1][n - 1] - (n - 1);
//            for (int j = n - 2; j >= 0; j--) {
//                maxRight[j] = Math.max(dp[i - 1][j] - j, maxRight[j + 1]);
//            }
//            for (int j = 0; j < n; j++) {
//                dp[i][j] = Math.max(points[i][j] - j + maxLeft[j], points[i][j] + j + maxRight[j]);
//            }
//        }
//        long ans = 0;
//        for (int j = 0; j < n; ++j) {
//            ans = Math.max(ans, dp[m - 1][j]);
//        }
//        return ans;
//    }

    public int maxFrequency(int[] nums, int k) {
        int count = 0;
        int l = 0, r = 0, len = nums.length, remain = k;
        Arrays.sort(nums);
        while (r < len) {
            int tmp = r == 0 ? 0 : r - 1;
            int dif = nums[r] - nums[tmp];
            int num = r - l;
            if (remain - dif * num >= 0) {
                count = Math.max(count, r - l + 1);
                r++;
                remain -= dif * num;
            } else {
                remain += nums[r - 1] - nums[l];
                l++;

            }
        }
        return count;
    }

    public boolean isBalanced(TreeNode root) {
        return Math.abs(depth(root.left) - depth(root.right)) <= 1 && isBalanced(root.left) && isBalanced(root.right);
    }

    private int depth(TreeNode root) {
        if (root == null) return 0;
        return Math.max(depth(root.left), depth(root.right)) + 1;
    }

//    public boolean isValidBST2(TreeNode root) {
//        if(root==null) return true;
//        if(root.left!=null&&root.left.val>root.val){
//            return false;
//        }
//        if(root.right!=null&&root.right.val<root.val){
//            return false;
//        }
//        return isValidBST(root.left)&&isValidBST(root.right);
//    }
//
//    public boolean isValidBST3(TreeNode root) {
//        return isGreaterThan(root.left,root.val)&&isSmallerThan(root.right,root.val);
//    }

//    private boolean isSmallerThan(TreeNode root, int val) {
//
//        if(root==null){
//            return true;
//        }
//        return isGreaterThan(root.left,root.val)&&isSmallerThan(root.right,root.val)&&isSmallerThan(root.left,val)&&isSmallerThan(root.right,val);
//    }
//
//    private boolean isGreaterThan(TreeNode root, int val) {
//        if(root==null){
//            return true;
//        }
//        return isGreaterThan(root.left,root.val)&&isSmallerThan(root.right,root.val)&&isGreaterThan(root.left,val)&&isGreaterThan(root.right,val);
//    }
//    public boolean isValidBST(TreeNode root) {
//        if(root==null) return true;
//        if(root.left!=null){
//            int leftmax = leftMax(root.left);
//            if(root.val<=leftmax){
//                return false;
//            }
//        }
//        if(root.right!=null){
//            int rightmax = righMin(root.right);
//            if(root.val>=rightmax){
//                return false;
//            }
//        }
//
//        return true;
//    }
//
//    private int righMin(TreeNode root) {
//
//    }

//    private int leftMax(TreeNode root) {
//        if(root==null) return 0;
//    }

    public int minPairSum(int[] nums) {
        Arrays.sort(nums);
        int left = 0, right = nums.length - 1, max = Integer.MIN_VALUE;
        while (left < right) {
            max = Math.max(max, nums[left] + nums[right]);
            left++;
            right--;
        }
        return max;
    }

    TreeNode target = null;

    public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
        inOrder(root, p);
        return target;
    }

    private void inOrder(TreeNode root, TreeNode p) {
        if (root == null) {
            return;
        }
        inOrder(root.left, p);
        if (root.val > p.val && (target == null || target.val > root.val)) {
            target = root;
        }
        inOrder(root.right, p);
    }

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) return null;
        if (root == p || root == q) return root;
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if (left != null && right != null) return root;
        if (left == null) return right;
        else return left;

    }

    //    List<List<Integer>> res = new LinkedList<>();
//
//    public List<List<Integer>> BSTSequences(TreeNode root) {
//        if (root == null) {
//            res.add(new LinkedList<>());
//            return res;
//        }
//
//        LinkedList<Integer> path = new LinkedList<>();
//        path.add(root.val);
//
//        helper(root, new LinkedList<>(), path);
//
//        return res;
//    }
//
//    public void helper(TreeNode root, LinkedList<TreeNode> queue, LinkedList<Integer> path) {
//        if (root == null) return;
//
//        if (root.left != null) queue.add(root.left);
//        if (root.right != null) queue.add(root.right);
//
//        if (queue.isEmpty()) {
//            res.add(new LinkedList<>(path));
//            return;
//        }
//
//        int lens = queue.size();
//        for (int i = 0; i < lens; i++) {
//            TreeNode cur = queue.get(i);
//            queue.remove(i);
//            path.add(cur.val);
//
//            helper(cur, new LinkedList<>(queue), path);
//
//            queue.add(i, cur);
//            path.removeLast();
//        }
//    }
//
//    public ListNode deleteDuplicates(ListNode head) {
//        if (head == null || head.next == null) return head;
//        ListNode cur = head;
//        while (cur != null) {
//            while (cur.next != null && cur.val == cur.next.val) {
//                cur.next = cur.next.next;
//            }
//            cur = cur.next;
//        }
//        return head;
//    }

    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null || headB == null) return null;
        ListNode a = headA, b = headB;
        while (a != null && b != null) {
            if (a == b) return a;
            a = a == null ? headB : a.next;
            b = b == null ? headA : b.next;
        }
        return null;

    }


    //    private List<List<Integer>> res;
//
    public List<List<Integer>> BSTSequences2(TreeNode root) {
        res = new ArrayList<>();
        Deque<TreeNode> path = new LinkedList<>();
        if (root != null) {
            path.add(root);
        }
        dfs(path, new ArrayList<>());
        return res;
    }

    private void dfs(Deque<TreeNode> path, List<Integer> ans) {
        if (path.isEmpty()) {
            res.add(new ArrayList<>(ans));
            return;
        }
        int size = path.size();
        for (int i = 0; i < size; i++) {
            TreeNode node = path.removeFirst();
            if (node.left != null) {
//                path.addLast(node.left);
                path.offer(node.left);
            }
            if (node.right != null) {
//                path.addLast(node.right);
                path.offer(node.right);
            }
            ans.add(node.val);
            dfs(path, ans);
            if (node.left != null) {
//                path.removeLast();
                path.remove(node.left);
            }
            if (node.right != null) {
//                path.removeLast();
                path.remove(node.right);

            }
            path.addLast(node);
            ans.remove(ans.size() - 1);
        }
    }


    private List<List<Integer>> res;

    public List<List<Integer>> BSTSequences(TreeNode root) {
        res = new ArrayList<>();
        Deque<TreeNode> queue = new LinkedList<>();
        if (root != null) {
            queue.add(root);
        }
        dfs(queue, new ArrayList<>());
        return res;
    }

    //    private void dfs(Deque<TreeNode> queue, ArrayList<Integer> path) {
//        if(!queue.isEmpty()){
//            res.add(new ArrayList<>(path));
//            return;
//        }
//        int size = queue.size();
//        for (int i = 0; i < size; i++) {
//            TreeNode cur = queue.poll();
//            if(cur.left!=null){
//                queue.offer(cur.left);
//            }
//            if(cur.right!=null){
//                queue.offer(cur.right);
//            }
//            path.add(cur.val);
//            dfs(queue,path);
//            if(cur.left!=null){
//                queue.remove(cur.left);
//            }
//            if(cur.right!=null){
//                queue.remove(cur.right);
//            }
//            queue.offer(cur);
//            path.remove(cur.val);
//        }
//    }
    class Node {
        int val;
        Node next;
        Node random;

        public Node(int val) {
            this.val = val;
            this.next = null;
            this.random = null;
        }
    }

    //    public Node copyRandomList(Node head) {
//        Node dum = new Node(-1), cur = head, ori = dum;
//        //1. cur.val -> random.val 2. cur.val-> new Node
//        Map<Integer, Integer> map = new HashMap<>();
//        Map<Integer, Node> map2 = new HashMap<>();
//        while (cur != null) {
//            dum.next = new Node(cur.val);
//            if (cur.random != null) {
//                map.put(cur.val, cur.random.val);
//            }
//            map2.put(cur.val, dum.next);
//            dum = dum.next;
//            cur = cur.next;
//        }
//        dum = ori.next;
//        while (dum != null) {
//            dum.random = map2.getOrDefault(map.getOrDefault(dum.val, null), null);
//            dum = dum.next;
//        }
//        return ori.next;
//    }
    public Node copyRandomList(Node head) {
        Node dum = new Node(-1), cur = head, res = dum;
        while (cur != null) {
            Node next = cur.next;
            cur.next = new Node(cur.val);
            cur.next.next = next;
            cur = cur.next.next;
        }
        cur = head;
        while (cur != null) {
            if (cur.random != null) {
                cur.next.random = cur.random.next;
            }
            cur = cur.next.next;
        }
        cur = head;
        while (cur != null) {
            Node next = cur.next.next;
            res.next = cur.next;
            res = res.next;
            cur.next = next;
            cur = cur.next;
        }

        return dum.next;
    }

    public boolean checkSubTree(TreeNode t1, TreeNode t2) {
        if (t2 == null) return true;
        return isSub(t1, t2);
    }

    private boolean isSub(TreeNode t1, TreeNode t2) {
        if (t1 == null && t2 == null) return true;
        if (t1 == null || t2 == null) return false;
        if (t1.val == t2.val) {
            return isSub(t1.left, t2.left) && isSub(t1.right, t2.right);
        } else {
            return isSub(t1.left, t2) || isSub(t1.right, t2);
        }
    }

    //    pre sum, but need backtracking
    int res3 = 0;

    public int pathSum(TreeNode root, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        int sum = 0;
        backtrack(map, root, target, sum);
        return res3;
    }

    private void backtrack(Map<Integer, Integer> map, TreeNode root, int target, int sum) {
        if (map.containsKey(sum - target)) {
            res3 += map.get(sum - target);
        }
        if (root == null) return;
        sum += root.val;
        map.put(sum, map.getOrDefault(sum, 0) + 1);
        backtrack(map, root.left, target, sum);
        backtrack(map, root.right, target, sum);
        map.put(sum, map.getOrDefault(sum, 0) - 1);
    }

    public boolean isCovered(int[][] ranges, int left, int right) {
        Arrays.sort(ranges, (a, b) -> (a[0] - b[0]));
        int start = left, id = 0, len = ranges.length;
        while (start <= right) {
            if (id < len && ranges[id][0] <= start && ranges[id][1] >= start) {
                start++;
            } else if (ranges[id][0] > start) {
                return false;
            } else {
                id++;
            }
            if (id == len) return false;
        }
        return true;
    }

    public int waysToStep(int n) {
        int[] dp = new int[n + 1];
        dp[0] = 1;
        for (int i = 1; i < n + 1; i++) {
            for (int j = 1; j <= 3; j++) {
                if (j > i) break;
                dp[i] = (dp[i] + dp[i - j]) % 1000000007;
            }
        }
        return dp[n];
    }

    //    List<List<Integer>> res5 = new ArrayList<>();
//    boolean finish = false;
//
//    public List<List<Integer>> pathWithObstacles(int[][] obstacleGrid) {
//        dfs2(0, 0, obstacleGrid);
//        return res5;
//    }
//
//    private void dfs2(int x, int y, int[][] obstacleGrid) {
//        if (finish||x<0||x>=obstacleGrid.length||y<0||y>=obstacleGrid[0].length) return;
//
//        if (obstacleGrid[x][y] == 0) {
//            if (x == obstacleGrid.length - 1 && y == obstacleGrid[0].length - 1) {
//                res5.add(new ArrayList<>(Arrays.asList(x, y)));
//                finish = true;
//                return;
//            }
//            res5.add(new ArrayList<>(Arrays.asList(x, y)));
//            dfs2(x,y+1,obstacleGrid);
//            dfs2(x+1,y,obstacleGrid);
//            if(!finish)
//                res5.remove(res5.size()-1);
//        } else {
//            return;
//        }
//    }
//    List<List<Integer>> res5 = new ArrayList<>();

    public List<List<Integer>> pathWithObstacles(int[][] obstacleGrid) {
        int row = obstacleGrid.length, col = obstacleGrid[0].length;
        if (obstacleGrid[0][0] == 1) return new ArrayList<>();
        List<List<Integer>>[][] dp = new List[row][col];
        Arrays.fill(dp[0], new ArrayList<>());
        dp[0][0].add(new ArrayList<>(Arrays.asList(0, 0)));

        for (int i = 1; i < col; i++) {
            if (obstacleGrid[0][i] != 1) {
                dp[0][i] = new ArrayList<>(dp[0][i - 1]);
                dp[0][i].add(new ArrayList<>(Arrays.asList(0, i)));
            } else {
                for (int j = i; j < col; j++) {
                    dp[0][j] = new ArrayList<>();
                }
                break;
            }
        }
        for (int i = 1; i < row; i++) {
            if (obstacleGrid[i][0] != 1) {
                dp[i][0] = new ArrayList<>(dp[i - 1][0]);
                dp[i][0].add(new ArrayList<>(Arrays.asList(i, 0)));
            } else {
                for (int j = i; j < row; j++) {
                    dp[j][0] = new ArrayList<>();
                }
                break;
            }
        }
        for (int i = 1; i < row; i++) {
            for (int j = 1; j < col; j++) {
                if (obstacleGrid[i][j] == 0) {
                    if (dp[i - 1][j].size() != 0) {
                        dp[i][j] = new ArrayList<>(dp[i - 1][j]);
                        dp[i][j].add(new ArrayList<>(Arrays.asList(i, j)));
                    } else if (dp[i][j - 1].size() != 0) {
                        dp[i][j] = new ArrayList<>(dp[i][j - 1]);
                        dp[i][j].add(new ArrayList<>(Arrays.asList(i, j)));
                    } else {
                        dp[i][j] = new ArrayList<>();
                    }
                } else {
                    dp[i][j] = new ArrayList<>();
                }
            }
        }
        return dp[row - 1][col - 1];
    }

    public int findMagicIndex(int[] nums) {
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == i) {
                return i;
            }
        }
        return -1;
    }

    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        dfs2(nums, 0, path, res);
        return res;
    }

    private void dfs2(int[] nums, int pos, List<Integer> path, List<List<Integer>> res) {
        if (pos == nums.length) {
            res.add(new ArrayList<>(path));
            return;
        }
        path.add(nums[pos]);
        dfs2(nums, pos + 1, path, res);
        path.remove(path.size() - 1);
        dfs2(nums, pos + 1, path, res);

    }

    private static double EPSILON = 1e-6;
    private static double TARGET = 24;

    public boolean judgePoint24(int[] cards) {
        double[] num = new double[]{cards[0], cards[1], cards[2], cards[3]};
        return helper(num);
    }

    private boolean helper(double[] num) {
        int len = num.length;
        if (len == 1) return Math.abs(num[0] - TARGET) < EPSILON;
        for (int i = 0; i < len; i++) {
            for (int j = i + 1; j < len; j++) {
                double[] newNum = new double[len - 1];
                int newId = 0;
                for (int k = 0; k < len; k++) {
                    if (k != i && k != j) newNum[newId++] = num[k];
                }
                for (double each : calculate(num[i], num[j])) {
                    newNum[len - 2] = each;
                    if (helper(newNum)) {
                        return true;
                    }
                }

            }
        }
        return false;
    }

    private List<Double> calculate(double a, double b) {
        List<Double> res = new ArrayList<>();
        res.add(a + b);
        res.add(a - b);
        res.add(b - a);
        res.add(a * b);
        if (Math.abs(b) > EPSILON) res.add(a / b);
        if (Math.abs(a) > EPSILON) res.add(b / a);
        return res;
    }

    int res5 = 0;

    public int findTargetSumWays(int[] nums, int target) {
        dfs3(nums, 0, 0, target);
        return res5;
    }

    private void dfs3(int[] nums, int pos, int sum, int target) {
        if (pos == nums.length) {
            if (sum == target) {
                res5++;
            }
            return;
        }
        dfs3(nums, pos + 1, sum + nums[pos], target);
        dfs3(nums, pos + 1, sum - nums[pos], target);
    }

    public int maxProfit(int[] prices) {
        if (prices.length < 2) return 0;
        int[][] dp = new int[4][prices.length];
        dp[0][0] = -prices[0];
        dp[2][1] = -prices[1];
        for (int i = 1; i < prices.length; i++) {
            dp[0][i] = Math.max(dp[0][i - 1], -prices[i]);
            dp[1][i] = Math.max(dp[1][i - 1], dp[0][i - 1] + prices[i]);
            if (i > 1) {
                dp[2][i] = Math.max(dp[2][i - 1], dp[1][i - 1] - prices[i]);
                dp[3][i] = Math.max(dp[3][i - 1], dp[2][i - 1] + prices[i]);
            }
        }
        System.out.println();
        return Math.max(dp[3][prices.length - 1], dp[1][prices.length - 1]);
    }

//    public int maxProfit(int[] prices) {
//        int n = prices.length;
//        int buy1 = -prices[0], sell1 = 0;
//        int buy2 = -prices[0], sell2 = 0;
//        for (int i = 1; i < n; ++i) {
//            buy1 = Math.max(buy1, -prices[i]);
//            sell1 = Math.max(sell1, buy1 + prices[i]);
//            buy2 = Math.max(buy2, sell1 - prices[i]);
//            sell2 = Math.max(sell2, buy2 + prices[i]);
//            System.out.printf("%d %d %d %d \n", buy1, sell1, buy2, sell2);
//
//        }
//        return sell2;
//    }

    public boolean areOccurrencesEqual(String s) {
        Map<Character, Integer> map = new HashMap<>();
        for (int i = 0; i < s.length(); i++) {
            map.put(s.charAt(i), map.getOrDefault(s.charAt(i), 0) + 1);
        }
        int num = -1;
        for (Map.Entry<Character, Integer> entry : map.entrySet()) {
            if (num == -1) {
                num = entry.getValue();
            } else {
                if (entry.getValue() != num) {
                    return false;
                }
            }
        }
        return true;

    }

    //    public int smallestChair(int[][] times, int targetFriend) {
//        int[] target = times[targetFriend];
//        Arrays.sort(times,(a,b)->(a[0]-b[0]));
//        PriorityQueue<int[]> pq = new PriorityQueue<>((a,b)->(a[1]-b[1]));
//        for (int i = 0; i < times.length; i++) {
//            int[] cur = times[i];
//            if(target[0]==times[i][0]&&target[1]==times[i][1]){
//                while(!pq.isEmpty()&&times[i][0]>=pq.peek()[1]){
//                    pq.poll();
//                }
//                return pq.size();
//            }
//            while(!pq.isEmpty()&&times[i][0]>=pq.peek()[1]){
//                pq.poll();
//            }
//            pq.offer(times[i]);
//        }
//        return -1;
//     }
    public int smallestChair(int[][] times, int targetFriend) {
        int[] target = times[targetFriend];
        Arrays.sort(times, (a, b) -> (a[0] - b[0]));
        int[][] seat = new int[times.length][2];
        Arrays.fill(seat[0], -1);
        for (int i = 0; i < times.length; i++) {
            int index = 0;
            if (target[0] == times[i][0] && target[1] == times[i][1]) {
                for (int j = 0; j < seat.length; j++) {
                    if (seat[j][0] == -1) {
                        return j;
                    }
                    if (seat[j][0] != -1 && seat[j][1] <= times[i][0]) {
                        return j;
                    }
                }
            }
            for (int j = 0; j < seat.length; j++) {
                if (seat[j][0] == -1) {
                    seat[j] = times[i];
                    break;
                }
                if (seat[j][0] != -1 && seat[j][1] <= times[i][0]) {
                    seat[j] = times[i];
                    break;
                }
            }
        }
        return -1;
    }

    public List<List<Long>> splitPainting(int[][] segments) {
        List<List<Long>> res = new ArrayList<>();
        Arrays.sort(segments);
        res.add(new ArrayList<>(Arrays.asList((long) segments[0][0], (long) segments[0][1], (long) segments[0][2])));
        for (int i = 1; i < segments.length; i++) {
            for (int j = 0; j < res.size(); j++) {
                List<Long> cur = res.get(j);
//                if(cur.get(0)>=segments[i][1]){
//                    res.add(j,new ArrayList<>(Arrays.asList((long)segments[i][0],(long)segments[i][1],(long)segments[i][2])));
//                    break;
//                }else
                if (cur.get(1) <= segments[i][0]) {
                    if (j == res.size() - 1) {
                        res.add(new ArrayList<>(Arrays.asList((long) segments[i][0], (long) segments[i][1], (long) segments[i][2])));
                    } else {
                        continue;
                    }
                } else {
//                    it needs to split!
                    cur.add(cur.remove(2) + segments[i][2]);
                    if (cur.get(1) > segments[i][2]) {
                        break;
                    } else {
                        segments[i][0] = Math.toIntExact(cur.get(1));
                    }
                }
            }
        }
        return res;
    }

    public int getLucky(String s, int k) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            sb.append(s.charAt(i) - 'a' + 1);
        }
//        double target = Double.parseDouble(sb.toString());
        String target = sb.toString();
        while (k > 0) {
            int sum = 0;
            for (int i = 0; i < target.length(); i++) {
                sum += Integer.parseInt(String.valueOf(target.charAt(i)));
            }

            target = String.valueOf(sum);
            System.out.println(target);
            k--;
        }
        return Integer.parseInt(target);
    }

    public String maximumNumber(String num, int[] change) {
        int start = 0, end = 0;
        boolean find = false;
        for (int i = 0; i < num.length(); i++) {
            if (change[num.charAt(i) - '0'] > (num.charAt(i) - '0')) {
                start = i;
                find = true;
            }
            if (find) {

                while (i < num.length() && change[num.charAt(i) - '0'] > (num.charAt(i) - '0')) {
                    end = i;
                    i++;
                }
                break;
            }
        }

        return find ? mutate(num, change, start, end) : num;
    }

    private String mutate(String num, int[] change, int start, int end) {
        StringBuilder sb = new StringBuilder();
        int id = 0;
        while (id < start) {
            sb.append(num.charAt(id));
            id++;
        }
        while (id >= start && id <= end) {
            sb.append(change[num.charAt(id) - '0']);
            id++;
        }
        while (id < num.length()) {
            sb.append(num.charAt(id));
            id++;
        }
        return sb.toString();
    }

//    public int maxCompatibilitySum(int[][] students, int[][] mentors) {
//        Comparator<int[]> comparator = new Comparator<int[]>() {
//            @Override
//            public int compare(int[] o1, int[] o2) {
//                int id = 0;
//                while (id < o1.length && o1[id] == o2[id]) {
//                    id++;
//                }
//                return id == o1.length ? 0 : o1[id] - o2[id];
//            }
//        };
//        Arrays.sort(students, comparator);
//        Arrays.sort(mentors, comparator);
//        int sum = 0;
//        for (int i = 0; i < students.length; i++) {
//
//            for (int j = 0; j < students[0].length; j++) {
//                sum += students[i][j] == mentors[i][j] ? 1 : 0;
//            }
//        }
//        return sum;
//    }


    public int maxCompatibilitySum(int[][] students, int[][] mentors) {
        int len = students.length, col = students[0].length;
        int[] s = new int[len];
        int[] m = new int[len];
        for (int i = 0; i < len; i++) {
            int tmp = 0;
            for (int j = 0; j < col; j++) {
                if (students[i][j] == 1) {
                    tmp |= (1 << j);
                }
            }
            s[i] = tmp;
        }
        for (int i = 0; i < len; i++) {
            int tmp = 0;
            for (int j = 0; j < col; j++) {
                if (mentors[i][j] == 1) {
                    tmp |= (1 << j);
                }
            }
            m[i] = tmp;
        }
        boolean[] vis = new boolean[len];
        int res = 0;
        for (int i = 0; i < len; i++) {
            int min = Integer.MAX_VALUE, id = 0;
            for (int j = 0; j < len; j++) {
                if (!vis[j]) {
                    int xor = s[i] ^ m[j];
                    int count = Integer.bitCount(xor);
                    if (count < min) {
                        min = count;
                        id = j;
                    }
                }
            }
            vis[id] = true;
            res += (col - min);
        }
        return res;
    }


//    public String maximumNumber(String num, int[] change) {
////        int max = Integer.parseInt(num);
//        String max = num;
//        for (int i = 0; i < num.length(); i++) {
//            for (int j = i; j < num.length(); j++) {
////                int res = mutate(num,change,i,j);
//                String res = mutate(num,change,i,j);
//                if(max.compareTo(res)<0){
//                    max = res;
//                }
//            }
//        }
//        return String.valueOf(max);
//    }
//

    public int[] restoreArray2(int[][] aps) {
        int m = aps.length, n = m + 1;
        Map<Integer, Integer> cnts = new HashMap<>();
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int i = 0; i < m; i++) {
            int[] cur = aps[i];
            cnts.put(cur[0], cnts.getOrDefault(cur[0], 0) + 1);
            cnts.put(cur[1], cnts.getOrDefault(cur[1], 0) + 1);
            List<Integer> a = map.getOrDefault(cur[0], new ArrayList<>());
            a.add(cur[1]);
            map.put(cur[0], a);
            List<Integer> b = map.getOrDefault(cur[1], new ArrayList<>());
            b.add(cur[0]);
            map.put(cur[1], b);
        }
        int start = -1;
        for (Map.Entry<Integer, Integer> entry : cnts.entrySet()) {
            if (entry.getValue() == 1) {
                start = entry.getKey();
                break;
            }
        }
        int[] res = new int[n];
        res[0] = start;
        res[1] = map.get(start).get(0);
        for (int i = 2; i < n; i++) {
            int pre = res[i - 1];
            List<Integer> cur = map.get(pre);
            for (Integer num : cur
            ) {
                if (num != res[i - 2]) {
                    res[i] = num;
                }
            }
        }
        return res;
    }

    public int[] restoreArray(int[][] adjacentPairs) {
        Map<Integer, List<Integer>> map = new HashMap<Integer, List<Integer>>();
        for (int[] adjacentPair : adjacentPairs) {
            map.putIfAbsent(adjacentPair[0], new ArrayList<Integer>());
            map.putIfAbsent(adjacentPair[1], new ArrayList<Integer>());
            map.get(adjacentPair[0]).add(adjacentPair[1]);
            map.get(adjacentPair[1]).add(adjacentPair[0]);
        }

        int n = adjacentPairs.length + 1;
        int[] ret = new int[n];
        for (Map.Entry<Integer, List<Integer>> entry : map.entrySet()) {
            int e = entry.getKey();
            List<Integer> adj = entry.getValue();
            if (adj.size() == 1) {
                ret[0] = e;
                break;
            }
        }

        ret[1] = map.get(ret[0]).get(0);
        for (int i = 2; i < n; i++) {
            List<Integer> adj = map.get(ret[i - 1]);
            ret[i] = ret[i - 2] == adj.get(0) ? adj.get(1) : adj.get(0);
        }
        return ret;
    }

    public List<List<Integer>> allPathsSourceTarget(int[][] graph) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        path.add(0);
        dfs4(graph, 0, path, res);
        return res;
    }

    private void dfs4(int[][] graph, int pos, List<Integer> path, List<List<Integer>> res) {
        if (pos == graph.length) {
            res.add(new ArrayList<>(path));
            return;
        }
        for (int i = 0; i < graph[pos].length; i++) {
            path.add(graph[pos][i]);
            dfs4(graph, graph[pos][i], path, res);
            path.remove(path.size() - 1);
        }
    }

    public int[] nextGreaterElement(int[] nums1, int[] nums2) {
        int[] res = new int[nums1.length];
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums1.length; i++) {
            map.put(nums1[i], i);
        }
        Stack<Integer> stack = new Stack<>();
        for (int i = nums2.length - 1; i >= 0; i--) {
            while (!stack.isEmpty() && stack.peek() <= nums2[i]) {
                stack.pop();
            }
            if (map.containsKey(nums2[i])) {
                res[map.get(nums2[i])] = stack.isEmpty() ? -1 : stack.peek();
            }
            stack.push(nums2[i]);
        }
        return res;
    }

    public int[] nextGreaterElements(int[] nums) {
        int[] res = new int[nums.length];
        Stack<Integer> stack = new Stack<>();
        for (int i = nums.length * 2 - 1; i >= 0; i--) {
            while (!stack.isEmpty() && stack.peek() <= nums[i]) {
                stack.pop();
            }
            res[i % nums.length] = stack.isEmpty() ? -1 : stack.peek();
            stack.push(nums[i % nums.length]);
        }
        return res;

    }

    //    public int[] dailyTemperatures(int[] temperatures) {
//        int len = temperatures.length;
//        int[] res = new int[len];
//        Stack<Integer> stack =new Stack<>();
//        for (int i = len-1; i >=0; i--) {
//            while(!stack.isEmpty()&&temperatures[stack.peek()]<=temperatures[i]){
//                stack.pop();
//            }
//            res[i]= stack.isEmpty()?0:stack.peek()-i;
//            stack.push(i);
//        }
//        return res;
//    }
    public int[] dailyTemperatures(int[] temperatures) {
        int len = temperatures.length;
        int[] res = new int[len];
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < len; i++) {
            while (!stack.isEmpty() && temperatures[stack.peek()] < temperatures[i]) {
                res[stack.peek()] = i - stack.peek();
                stack.pop();
            }
            stack.push(i);
        }
        return res;
    }

    public int calculate(String s) {
        // 存放所有的数字
        Deque<Integer> nums = new ArrayDeque<>();
        // 为了防止第一个数为负数，先往 nums 加个 0
        nums.addLast(0);
        // 将所有的空格去掉
        s = s.replaceAll(" ", "");
        // 存放所有的操作，包括 +/-
        Deque<Character> ops = new ArrayDeque<>();
        int n = s.length();
        char[] cs = s.toCharArray();
        for (int i = 0; i < n; i++) {
            char c = cs[i];
            if (c == '(') {
                ops.addLast(c);
            } else if (c == ')') {
                while (!ops.isEmpty() && ops.peekLast() != '(') {
                    calc(nums, ops);
                }
                ops.removeLast();
            } else if (Character.isDigit(c)) {
                int u = 0;
                int j = i;
                while (j < n && Character.isDigit(cs[j]))
                    u = u * 10 + (int) (cs[j++] - '0');
                nums.addLast(u);
                i = j - 1;
            } else {
                while (!ops.isEmpty() && ops.peekLast() != '(') {
                    calc(nums, ops);
                }
                ops.addLast(c);
            }
        }
        while (!ops.isEmpty()) calc(nums, ops);
        return nums.peekLast();
    }

    void calc(Deque<Integer> nums, Deque<Character> ops) {
        if (nums.isEmpty() || nums.size() < 2) return;
        if (ops.isEmpty()) return;
        int b = nums.pollLast(), a = nums.pollLast();
        char op = ops.pollLast();
        nums.addLast(op == '+' ? a + b : a - b);
    }


    public int calculate2(String s) {
        Deque<Integer> stack = new ArrayDeque<>();
        int num = 0;
        char preSign = '+';
        char[] chars = s.toCharArray();
        for (int i = 0; i < chars.length; i++) {
            if (Character.isDigit(chars[i])) {
                num = num * 10 + chars[i] - '0';
            }
            if (!Character.isDigit(chars[i]) && chars[i] != ' ' || i == chars.length - 1) {
                if (preSign == '+') {
                    stack.push(num);
                } else if (preSign == '-') {
                    stack.push(-num);
                } else if (preSign == '*') {
                    stack.push(stack.pop() * num);
                } else if (preSign == '/') {
                    stack.push(stack.pop() / num);
                }
                preSign = chars[i];
                num = 0;
            }
        }
        int res = 0;
        while (!stack.isEmpty()) {
            res += stack.pop();
        }
        return res;
    }

    public int largestRectangleArea(int[] heights) {
        int len = heights.length;
        if (len == 0) {
            return 0;
        }
        if (len == 1) {
            return heights[0];
        }
        Deque<Integer> stack = new ArrayDeque<>(len);
        stack.addLast(-1);
        int max = 0;
        for (int i = 0; i < len; i++) {
            while (stack.peekLast() != -1 && heights[stack.peekLast()] > heights[i]) {
                int cur = stack.pollLast();
                max = Math.max(max, heights[cur] * (i - stack.peekLast() - 1));
            }
            stack.addLast(i);
        }
        while (stack.peekLast() != -1) {
            int cur = stack.pollLast();
            max = Math.max(max, heights[cur] * (len - stack.peekLast() - 1));
        }
        return max;
    }

//    public int minOperations(int[] target, int[] arr) {
//        int count =0,id=0;
//        Map<Integer,Integer> map = new HashMap<>();
//        for (int i = 0; i < target.length; i++) {
//            map.put(target[i],i);
//        }
//        for (int i = 0; i < arr.length; i++) {
//            if(map.containsKey(arr[i])){
//                if(map.get(arr[i])<id){
//                    continue;
//                }else if(map.get(arr[i])==id){
//                    id++;
//                }else{
//                    count+=(map.get(arr[i])-id);
//                    id = map.get(arr[i])+1;
//                }
//
//            }
//        }
//        count+=(target.length-id);
//        return count;
//    }


    public int minOperations(int[] target, int[] arr) {
        int id = 0;
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < target.length; i++) {
            map.put(target[i], i);
        }
        int[] newArr = new int[arr.length];
        for (int i = 0; i < arr.length; i++) {
            if (map.containsKey(arr[i])) {
                newArr[id++] = map.get(arr[i]);
            }
        }
        int[] new2 = new int[id];
        for (int i = 0; i < id; i++) {
            new2[i] = newArr[i];
        }
        int res = lengthOfLIS(new2);
        return target.length - res;
    }


    public int lengthOfLIS(int[] nums) {
        if (nums.length == 0) return 0;
        int[] tails = new int[nums.length];
        tails[0] = nums[0];
        int index = 0;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] > tails[index]) {
                index++;
                tails[index] = nums[i];
            }
            if (nums[i] < tails[index]) {
                int l = 0, r = index;
                boolean equal = false;
                while (l < r) {
                    int mid = (r - l) / 2 + l;
                    if (tails[mid] == nums[i]) {
                        equal = true;
                        break;
                    } else if (tails[mid] > nums[i]) {
                        r = mid;
                    } else {
                        l = mid + 1;
                    }
                }
                if (!equal) {
                    tails[l] = nums[i];
                }
            }
        }
        return index + 1;
    }

    public int trap(int[] height) {
        int sum = 0;
        int[] max_left = new int[height.length];
        int[] max_right = new int[height.length];

        for (int i = 1; i < height.length; i++) {
            max_left[i] = Math.max(max_left[i - 1], height[i - 1]);
        }
        for (int i = height.length - 2; i >= 0; i--) {
            max_right[i] = Math.max(max_right[i + 1], height[i + 1]);
        }
        for (int i = 0; i < height.length; i++) {
            int min = Math.min(max_left[i], max_right[i]);
            if (min > height[i]) {
                sum += (min - height[i]);
            }
        }
        return sum;
    }

    //    public int[] subSort(int[] array) {
//        int left = Integer.MAX_VALUE, right = -1, max = Integer.MIN_VALUE;
//        for (int i = 1; i < array.length; i++) {
//            if (array[i] < array[i - 1]) {
//                int target = array[i], l = 0, r = i - 1;
//                //find the first bigger than target
//                while (l < r) {
//                    int mid = (r - l) / 2 + l;
//                    if (array[mid] >= target) {
//                        r = mid;
//                    } else {
//                        l = mid + 1;
//                    }
//                }
//                while(l<array.length&&array[l]==target){
//                    l++;
//                }
//                left = Math.min(left,l);
//                max = array[l];
//                for (int j = l+1; j <i ; j++) {
//                    max = Math.max(max,array[j]);
//                }
//                for (int j = i; j < array.length; j++) {
//                    if(array[j]<max){
//                        right = Math.max(j,right);
//                    }
//                }
//            }
//        }
//        return left==Integer.MAX_VALUE? new int[]{-1,right}:new int[]{left,right};
//    }


//    public int[] subSort(int[] array) {
//        Stack<Integer> stack = new Stack<>();
//        int left = Integer.MAX_VALUE, right = -1, max = Integer.MIN_VALUE;
//        for (int i = 0; i < array.length; i++) {
//            while (!stack.isEmpty() && array[stack.peek()] > array[i]) {
//                left = Math.min(left, stack.peek() + 1);
//                right = Math.max(right, i);
//                //first peek, then pop!
//                int pop = stack.pop();
//                max = Math.max(pop, max);
//            }
//            if(array[i]<max){
//                right = i;
//            }
//            stack.push(i);
//        }
//        return left == Integer.MAX_VALUE ? new int[]{-1, right} : new int[]{left, right};
//    }


    public int[] subSort(int[] array) {
        Stack<Integer> stack1 = new Stack<>();
        Stack<Integer> stack2 = new Stack<>();
        int min = array.length;
        int max = 0;
        int endIndex;
        for (int i = 0; i < array.length; i++) {
            endIndex = array.length - 1 - i;
            while (!stack1.isEmpty() && array[i] < array[stack1.peek()]) {
                min = Math.min(min, stack1.pop());
            }
            stack1.push(i);
            while (!stack2.isEmpty() && array[endIndex] > array[stack2.peek()]) {
                max = Math.max(max, stack2.pop());
            }
            stack2.push(endIndex);
        }
        if (min >= max) return new int[]{-1, -1};
        return new int[]{min, max};
    }

    int min = -1, sec = Integer.MAX_VALUE;
    boolean changed = false;

    public int findSecondMinimumValue(TreeNode root) {
        if (root == null) return -1;
        min = root.val;
        recur(root);
        return !changed ? -1 : sec;
    }

    private void recur(TreeNode root) {
        if (root == null) return;
        System.out.println(Integer.MAX_VALUE);
        if (root.val > min) {
            sec = Math.min(root.val, sec);
            changed = true;
        }
        recur(root.left);
        recur(root.right);
    }

    //    public boolean find132pattern(int[] nums) {
//        Stack<Integer> stack = new Stack<>();
//        for (int i = 0; i < nums.length; i++) {
//            while(!stack.isEmpty()&&nums[stack.peek()]>nums[i]){
//                stack.pop();
//                if(!stack.isEmpty()&&nums[stack.peek()]<nums[i]){
//                    return true;
//                }
//            }
//            stack.push(i);
//        }
//        return false;
//    }
    public boolean find132pattern(int[] nums) {
        Stack<Integer> stack = new Stack<>();
        int k = Integer.MIN_VALUE;
        for (int i = nums.length - 1; i >= 0; i--) {
            if (nums[i] < k) return true;
            while (!stack.isEmpty() && nums[stack.peek()] < nums[i]) {
                k = Math.max(k, nums[stack.pop()]);
            }
            stack.push(i);
        }
        return false;
    }

    public int[] finalPrices(int[] prices) {
        int[] res = new int[prices.length];
        Stack<Integer> stack = new Stack<>();
        for (int i = prices.length - 1; i >= 0; i--) {
            while (!stack.isEmpty() && stack.peek() > prices[i]) {
                stack.pop();
            }
            if (!stack.isEmpty() && stack.peek() <= prices[i]) {
                res[i] = prices[i] - stack.peek();
            } else {
                res[i] = prices[i];
            }

            stack.push(prices[i]);
        }
        return res;
    }


    public TreeNode constructMaximumBinaryTree(int[] nums) {
        return constructTree(nums, 0, nums.length - 1);

    }

    private TreeNode constructTree(int[] nums, int start, int end) {
        if (start > end) return null;
        int rootId = findMax(nums, start, end);
        TreeNode root = new TreeNode(nums[rootId]);
        root.left = constructTree(nums, start, rootId - 1);
        root.right = constructTree(nums, rootId + 1, end);
        return root;
    }

    private int findMax(int[] nums, int start, int end) {
        int max = Integer.MIN_VALUE, res = 0;
        for (int i = start; i <= end; i++) {
            if (max < nums[i]) {
                res = i;
                max = nums[i];
            }
        }
        return res;
    }

//    public String removeDuplicateLetters(String s) {
//        boolean[] exist = new boolean[s.length()];
//        Arrays.fill(exist, true);
//        Stack<Integer> stack = new Stack<>();
//        Map<Character, Integer> map = new HashMap<>();
//        for (int i = s.length() - 1; i >= 0; i--) {
//            if (!map.containsKey(s.charAt(i))) {
//                stack.push((int) s.charAt(i));
//                map.put(s.charAt(i), i);
//            } else {
//                if ((int) s.charAt(i) >= stack.peek()) {
////                    exist[map.get(s.charAt(i))]=false;
//                    exist[i] = false;
//                } else {
//                    exist[map.get(s.charAt(i))] = false;
//                    map.put(s.charAt(i), i);
//                    stack.push((int) s.charAt(i));
//                }
//            }
//        }
//        StringBuilder sb = new StringBuilder();
//        for (int i = 0; i < exist.length; i++) {
//            if (exist[i]) {
//                sb.append(s.charAt(i));
//            }
//        }
//        return sb.toString();
//    }

    public String removeDuplicateLetters(String s) {
        Stack<Character> stack = new Stack<>();
        Map<Character, Integer> map = new HashMap<>();
        for (int i = 0; i < s.length(); i++) {
            map.put(s.charAt(i), map.getOrDefault(s.charAt(i), 0) + 1);
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            map.put(s.charAt(i), map.get(s.charAt(i)) - 1);
            if (!stack.contains(s.charAt(i))) {
                while (!stack.isEmpty() && map.get(stack.peek()) > 0 && s.charAt(i) <= stack.peek()) {
                    char cur = stack.pop();
//                    map.put(cur, map.get(cur) - 1);
                }
                stack.push(s.charAt(i));
            }
        }
        while (!stack.isEmpty()) {
            sb.append(stack.pop());
        }
        sb.reverse();
        return sb.toString();
    }

    public String removeKdigits(String num, int k) {
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < num.length(); i++) {
            while (!stack.isEmpty() && k > 0 && stack.peek() > num.charAt(i) - '0') {
                k--;
                stack.pop();
            }
            stack.push(num.charAt(i) - '0');
        }
        while (k > 0) {
            stack.pop();
            k--;
        }
        StringBuilder sb = new StringBuilder();
        while (!stack.isEmpty()) {
            sb.append(stack.pop());
        }
        sb.reverse();
        int id = 0;
        for (int i = 0; i < sb.length(); i++) {
            if (sb.charAt(i) == '0') {
                id++;
            } else {
                break;
            }
        }
        if (id == sb.length()) return "0";
        else {
            return sb.substring(id).toString();
        }
    }

    public static void main(String[] args) {
        Lc10 lc10 = new Lc10();
//        int res = lc10.addRungs(new int[]{3, 4, 6, 7}, 2);
//        System.out.println(res);
//        lc10.maxFrequency(new int[]{1, 2, 4}, 5);
//        TreeNode root = new TreeNode(2, new TreeNode(1), new TreeNode(3, new TreeNode(4), new TreeNode(5)));
        TreeNode root = new TreeNode();
//        List<List<Integer>> res = lc10.BSTSequences(root);
//        List<List<Integer>> res2 = lc10.BSTSequences(root);
//        for (int i = 0; i < res.size(); i++) {
//            List<Integer> tmp = res.get(i);
//            StringBuilder sb = new StringBuilder();
//            sb.append(Arrays.toString(tmp.toArray()));
//            System.out.println(sb.toString());
//        }
//        for (int i = 0; i < res2.size(); i++) {
//            List<Integer> tmp = res2.get(i);
//            StringBuilder sb = new StringBuilder();
//            sb.append(Arrays.toString(tmp.toArray()));
//            System.out.println(sb.toString());
//        }
        int[][] path = new int[][]{{0, 0, 0}, {0, 1, 0}, {0, 0, 0}};
        int[][] path2 = {{0}, {1}};
//        lc10.pathWithObstacles(path2);
        int[] target = new int[]{4, 1, 8, 7};
//        lc10.judgePoint24(target);

//        PriorityQueue<int[]> pq = new PriorityQueue<>((a,b)->(a[1]-b[1]));
//        pq.offer(new int[]{1,3});
//        pq.offer(new int[]{2,4});
////        pq.offer(new int[]{3,-1});
//        System.out.println(pq.peek()[1]);
        int[][] p = {{33889, 98676}, {80071, 89737}, {44118, 52565}, {52992, 84310}, {78492, 88209}, {21695, 67063}, {84622, 95452}, {98048, 98856}, {98411, 99433}, {55333, 56548}, {65375, 88566}, {55011, 62821}, {48548, 48656}, {87396, 94825}, {55273, 81868}, {75629, 91467}};

//        lc10.smallestChair(p, 6);
//        lc10.getLucky("fleyctuuajsr", 5);
//        int[][] s = new int[][]{{0, 0, 1, 1, 1, 0, 1}, {0, 1, 1, 0, 0, 0, 0}, {0, 0, 1, 1, 1, 1, 1}, {0, 1, 0, 0, 1, 0, 1}, {1, 0, 1, 1, 1, 1, 1}};
//
//        int[][] m = {{0, 1, 1, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0, 1}, {0, 1, 0, 1, 0, 0, 1}, {1, 0, 0, 0, 1, 0, 1}, {1, 1, 1, 1, 1, 0, 0}};
        int[][] s = {{1, 1, 1}, {0, 0, 1}, {0, 0, 1}, {0, 1, 0}};
        int[][] m = {{1, 0, 1}, {0, 1, 1}, {0, 1, 0}, {1, 1, 0}};
//        lc10.maxCompatibilitySum(s, m);
//        lc10.calculate("(1+(4+5+2)-3)+(6+8)");
//        int res = lc10.calculate2("2*21");
//        System.out.println(res);

//        lc10.largestRectangleArea(new int[]{0, 3, 3, 3});
//        lc10.minOperations(new int[]{6,4,8,1,3,2}, new int[]{4,7,6,2,3,8,6,1});
//        lc10.minOperations(new int[]{5, 1, 3}, new int[]{9, 4, 2, 3, 4});


        int[] bb = new int[]{1, 2, 4, 7, 10, 11, 7, 12, 6, 7, 16, 18, 19};
//        lc10.subSort(bb);


//        lc10.removeKdigits("10200", 1);
//        String res = lc10.removeDuplicateLetters("abacb");
        String res = lc10.removeDuplicateLetters("bbcaac");
        System.out.println(res);
//[5,1,3]
//[9,4,2,3,4]
//        lc10.lengthOfLIS(new int[]{4,10,4,3,8,9});

    }
}

//class Solution {
//    public int[] subSort(int[] array) {
//        Stack<Integer> stack1 = new Stack<>();
//        Stack<Integer> stack2 = new Stack<>();
//        int min = array.length;
//        int max = 0;
//        int endIndex;
//        for (int i = 0; i < array.length; i++) {
//            endIndex = array.length - 1 - i;
//            while (!stack1.isEmpty() && array[i] < array[stack1.peek()]) {
//                min = Math.min(min, stack1.pop());
//            }
//            stack1.push(i);
//            while (!stack2.isEmpty() && array[endIndex] > array[stack2.peek()]) {
//                max = Math.max(max, stack2.pop());
//            }
//            stack2.push(endIndex);
//        }
//        if (min >= max) return new int[]{-1, -1};
//        return new int[]{min, max};
//    }
//}


//class Solution3 {
//    public int[] restoreArray(int[][] aps) {
//        int m = aps.length, n = m + 1;
//        Map<Integer, Integer> cnts = new HashMap<>();
//        Map<Integer, List<Integer>> map = new HashMap<>();
//        for (int[] ap : aps) {
//            int a = ap[0], b = ap[1];
//            cnts.put(a, cnts.getOrDefault(a, 0) + 1);
//            cnts.put(b, cnts.getOrDefault(b, 0) + 1);
//            List<Integer> alist = map.getOrDefault(a, new ArrayList<>());
//            alist.add(b);
//            map.put(a, alist);
//            List<Integer> blist = map.getOrDefault(b, new ArrayList<>());
//            blist.add(a);
//            map.put(b, blist);
//        }
//        int start = -1;
//        for (int i : cnts.keySet()) {
//            if (cnts.get(i) == 1) {
//                start = i;
//                break;
//            }
//        }
//        int[] ans = new int[n];
//        ans[0] = start;
//        ans[1] = map.get(start).get(0);
//        for (int i = 2; i < n; i++) {
//            int x = ans[i - 1];
//            List<Integer> list = map.get(x);
//            for (int j : list) {
//                if (j != ans[i - 2]) ans[i] = j;
//            }
//        }
//        return ans;
//    }
//}


//class Solution {
//    int N = 1010, M = N * 2;
//    int[] he = new int[N], e = new int[M], ne = new int[M];
//    int idx;
//    void add(int a, int b) {
//        e[idx] = b;
//        ne[idx] = he[a];
//        he[a] = idx++;
//    }
//    boolean[] vis = new boolean[N];
//    public List<Integer> distanceK(TreeNode root, TreeNode t, int k) {
//        List<Integer> ans = new ArrayList<>();
//        Arrays.fill(he, -1);
//        dfs(root);
//        Deque<Integer> d = new ArrayDeque<>();
//        d.addLast(t.val);
//        vis[t.val] = true;
//        while (!d.isEmpty() && k >= 0) {
//            int size = d.size();
//            while (size-- > 0) {
//                int poll = d.pollFirst();
//                if (k == 0) {
//                    ans.add(poll);
//                    continue;
//                }
//                for (int i = he[poll]; i != -1 ; i = ne[i]) {
//                    int j = e[i];
//                    if (!vis[j]) {
//                        d.addLast(j);
//                        vis[j] = true;
//                    }
//                }
//            }
//            k--;
//        }
//        return ans;
//    }
//    void dfs(TreeNode root) {
//        if (root == null) return;
//        if (root.left != null) {
//            add(root.val, root.left.val);
//            add(root.left.val, root.val);
//            dfs(root.left);
//        }
//        if (root.right != null) {
//            add(root.val, root.right.val);
//            add(root.right.val, root.val);
//            dfs(root.right);
//        }
//    }
//}
