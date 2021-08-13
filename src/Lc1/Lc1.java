package Lc1;

import java.util.*;

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode() {
    }

    TreeNode(int val) {
        this.val = val;
    }

    TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}
public class Lc1 {

    int k;
    public int kthSmallest(TreeNode root, int k_) {
        k=k_;
        return helper(root);
    }

    private int helper(TreeNode root) {
        int left = -1, right = -1;
        if (root.left != null) {
            left = kthSmallest(root.left, k);
        }
        if (left != -1) {
            return left;
        }
        k--;
        if (k == 0) {
            return root.val;
        }
        if (root.right != null) {
            right = kthSmallest(root.right, k);
        }
        if (right != -1) {
            return right;
        }
        return -1;
    }

    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        helper2(res,root,0);
        return res;
    }

    private void helper2(List<List<Integer>> res, TreeNode root, int depth) {
        if(res.size()<=depth){
            res.add(new ArrayList<>());
        }
        res.get(depth).add(root.val);
        if(root.left!=null){
            helper2(res,root.left,depth+1);
        }
        if(root.right!=null){
            helper2(res,root.right,depth+1);
        }
    }





    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int len = queue.size();
            for (int i = 0; i < len; i++) {
                TreeNode node = queue.poll();
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
                if (i == len - 1) {
                    res.add(node.val);
                }
            }
        }
        return res;
    }

    public boolean checkSubarraySum2(int[] nums, int k) {
        Set<Integer> set = new HashSet<>();
        int n = nums.length;
        int[] sum = new int[n + 1];
        for (int i = 1; i <= n; i++) {
            sum[i] = sum[i - 1] + nums[i - 1];
        }
        for (int i = 2; i <= n; i++) {
            set.add(sum[i - 2] % k);
            if (set.contains(sum[i] % k)) return true;
        }
        return false;
    }

    public boolean checkSubarraySum(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<>();
        int len = nums.length;
        int sum = 0;
        for (int i = 0; i < len; i++) {
            sum += nums[i];
            if (map.containsKey(sum % k)) {
                if (i - map.get(sum % k) >= 2) {
                    return true;
                }
            } else {
                map.put(sum % k, i);
            }
        }
        return false;
    }

    public int openLock(String[] deadends, String target) {
        if ("0000".equals(target)) {
            return 0;
        }
        Set<String> deadSet = new HashSet<>(Arrays.asList(deadends));
        if (deadSet.contains("0000")) {
            return -1;
        }
        int step = 0;
        Queue<String> queue = new LinkedList<>();
        Set<String> seen = new HashSet<>(Arrays.asList("0000"));
        queue.offer("0000");
        while (!queue.isEmpty()) {
            int size = queue.size();
            step++;
            for (int i = 0; i < size; ++i) {
                String cur = queue.poll();
                for (String next : get(cur)) {
                    if (!seen.contains(next) && !deadSet.contains(next)) {
                        if (next.equals(target)) {
                            return step;
                        }
                        seen.add(next);
                        queue.offer(next);
                    }
                }
            }
        }
        return -1;

    }

    private List<String> get(String cur) {
        List<String> res = new ArrayList<>();
        char[] array = cur.toCharArray();
        for (int i = 0; i < 4; i++) {
            char num = array[i];
            array[i] = numPre(num);
            res.add(new String(array));
            array[i] = numSuc(num);
            res.add(new String(array));
            array[i] = num;
        }
        return res;
    }

    private char numPre(char c) {
        return c == '0' ? '9' : (char) (c - 1);
    }

    private char numSuc(char c) {
        return c == '9' ? '0' : (char) (c + 1);
    }

    public int findMaxLength(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, -1);
        int sum = 0;
        int res = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == 1) {
                sum += 1;
            } else {
                sum -= 1;
            }
            if (map.containsKey(sum)) {
                if (i - map.get(sum) > res) {
                    res = i - map.get(sum);
                }
            } else {
                map.put(sum, i);
            }
        }
        return res;
    }


    class Node {
        String str;
        int x, y;

        Node(String _str, int _x, int _y) {
            str = _str;
            x = _x;
            y = _y;
        }
    }

    int n = 2, m = 3;
    String s, e;
    int x, y;

    public int slidingPuzzle(int[][] board) {
        s = "";
        e = "123450";
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                s += board[i][j];
                if (board[i][j] == 0) {
                    x = i;
                    y = j;
                }
            }
        }
        int ans = bfs2();
        return ans;
    }

    int[][] dirs = new int[][]{{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    public int bfs() {
        Deque<Node> d = new ArrayDeque<>();
        Map<String, Integer> map = new HashMap<>();
        Node root = new Node(s, x, y);
        d.addLast(root);
        map.put(s, 0);
        while (!d.isEmpty()) {
            Node poll = d.pollFirst();
            int step = map.get(poll.str);
            if (poll.str.equals(e)) return step;
            int dx = poll.x, dy = poll.y;
            for (int[] di : dirs) {
                int nx = dx + di[0], ny = dy + di[1];
                if (nx < 0 || nx >= n || ny < 0 || ny >= m) continue;
                String nStr = update(poll.str, dx, dy, nx, ny);
                if (map.containsKey(nStr)) continue;
                Node next = new Node(nStr, nx, ny);
                d.addLast(next);
                map.put(nStr, step + 1);
            }
        }
        return -1;
    }

    public int bfs2() {
        Deque<Node> d = new ArrayDeque<>();
        Set<String> map = new HashSet<>();
        Node root = new Node(s, x, y);
        d.addLast(root);
        map.add(s);
        int step = 0;
        while (!d.isEmpty()) {
            int size = d.size();
            step++;
            for (int i = 0; i < size; i++) {
                Node poll = d.pollFirst();
//                int step = map.get(poll.str);
                if (poll.str.equals(e)) return step - 1;
                int dx = poll.x, dy = poll.y;
                for (int[] di : dirs) {
                    int nx = dx + di[0], ny = dy + di[1];
                    if (nx < 0 || nx >= n || ny < 0 || ny >= m) continue;
                    String nStr = update(poll.str, dx, dy, nx, ny);
                    if (map.contains(nStr)) continue;
                    Node next = new Node(nStr, nx, ny);
                    d.addLast(next);
                    map.add(nStr);
                }
            }

        }
        return -1;
    }

    String update(String cur, int i, int j, int p, int q) {
        char[] cs = cur.toCharArray();
        char tmp = cs[i * m + j];
        cs[i * m + j] = cs[p * m + q];
        cs[p * m + q] = tmp;
        return String.valueOf(cs);
    }


    //    public int ladderLength2(String beginWord, String endWord, List<String> wordList) {
//        Set<String> set = new HashSet<>(wordList);
//        Queue<String> queue = new LinkedList<>();
//        queue.offer(beginWord);
//        int res = 0;
//        while(!queue.isEmpty()){
//            int size = queue.size();
//            res++;
//            for(int i=0;i<size;i++){
//                String cur = queue.poll();
//                char[] curArray = cur.toCharArray();
//                for(int i=0;i<cur.length();i++){
//                    char tmp =curArray[i];
//                    for(int j=0;j<26;j++){
//                        tmp
//                    }
//                }
//            }
//        }
//        return 0;
//    }
    public int ladderLength2(String beginWord, String endWord, List<String> wordList) {
        Set<String> wordSet = new HashSet<>(wordList);
        if (wordSet.size() == 0 || !wordSet.contains(endWord)) {
            return 0;
        }
        Queue<String> queue = new LinkedList<>();
        queue.offer(beginWord);
        Set<String> visited = new HashSet<>();
        visited.add(beginWord);
        int step = 0;
        while (!queue.isEmpty()) {
            int size = queue.size();
            step++;
            for (int i = 0; i < size; i++) {
                String cur = queue.poll();
                if (updateWord(cur, wordSet, endWord, queue, visited)) {
                    return step;
                }
            }
        }
        return 0;
    }

    private boolean updateWord(String cur, Set<String> wordSet, String endWord, Queue<String> queue, Set<String> visited) {
        char[] array = cur.toCharArray();
        for (int i = 0; i < array.length; i++) {
            char curChar = array[i];
            for (char j = 'a'; j <= 'z'; j++) {
                if (j == curChar) {
                    continue;
                }
                array[i] = j;
                String nextWord = String.valueOf(array);
                if (wordSet.contains(nextWord)) {
                    if (nextWord.equals(endWord)) {
                        return true;
                    }
                    if (!visited.contains(nextWord)) {
                        queue.offer(nextWord);
                        visited.add(nextWord);
                    }
                }
            }
            array[i] = curChar;
        }
        return false;
    }

    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        // 第 1 步：先将 wordList 放到哈希表里，便于判断某个单词是否在 wordList 里
        Set<String> wordSet = new HashSet<>(wordList);
        if (wordSet.size() == 0 || !wordSet.contains(endWord)) {
            return 0;
        }

        // 第 2 步：已经访问过的 word 添加到 visited 哈希表里
        Set<String> visited = new HashSet<>();
        // 分别用左边和右边扩散的哈希表代替单向 BFS 里的队列，它们在双向 BFS 的过程中交替使用
        Set<String> beginVisited = new HashSet<>();
        beginVisited.add(beginWord);
        Set<String> endVisited = new HashSet<>();
        endVisited.add(endWord);

        // 第 3 步：执行双向 BFS，左右交替扩散的步数之和为所求
        int step = 1;
        while (!beginVisited.isEmpty() && !endVisited.isEmpty()) {
            // 优先选择小的哈希表进行扩散，考虑到的情况更少
            if (beginVisited.size() > endVisited.size()) {
                Set<String> temp = beginVisited;
                beginVisited = endVisited;
                endVisited = temp;
            }

            // 逻辑到这里，保证 beginVisited 是相对较小的集合，nextLevelVisited 在扩散完成以后，会成为新的 beginVisited
            Set<String> nextLevelVisited = new HashSet<>();
            for (String word : beginVisited) {
                if (changeWordEveryOneLetter(word, endVisited, visited, wordSet, nextLevelVisited)) {
                    return step + 1;
                }
            }

            // 原来的 beginVisited 废弃，从 nextLevelVisited 开始新的双向 BFS
            beginVisited = nextLevelVisited;
            step++;
        }
        return 0;
    }


    /**
     * 尝试对 word 修改每一个字符，看看是不是能落在 endVisited 中，扩展得到的新的 word 添加到 nextLevelVisited 里
     */
    private boolean changeWordEveryOneLetter(String word, Set<String> endVisited,
                                             Set<String> visited,
                                             Set<String> wordSet,
                                             Set<String> nextLevelVisited) {
        char[] charArray = word.toCharArray();
        for (int i = 0; i < word.length(); i++) {
            char originChar = charArray[i];
            for (char c = 'a'; c <= 'z'; c++) {
                if (originChar == c) {
                    continue;
                }
                charArray[i] = c;
                String nextWord = String.valueOf(charArray);
                if (wordSet.contains(nextWord)) {
                    if (endVisited.contains(nextWord)) {
                        return true;
                    }
                    if (!visited.contains(nextWord)) {
                        nextLevelVisited.add(nextWord);
                        visited.add(nextWord);
                    }
                }
            }
            // 恢复，下次再用
            charArray[i] = originChar;
        }
        return false;
    }

    public static int ladderLength3(String beginWord, String endWord, List<String> wordList) {
        Set<String> wordSet = new HashSet<>(wordList);
        if (wordSet.size() == 0 || !wordSet.contains(endWord)) {
            return 0;
        }
        Queue<String> beginQueue = new LinkedList<>();
        beginQueue.offer(beginWord);
        Queue<String> endQueue = new LinkedList<>();
        endQueue.offer(endWord);
        Set<String> visited = new HashSet<>();
        int step = 0;
        while (!beginQueue.isEmpty() && !endQueue.isEmpty()) {
            if (endQueue.size() < beginQueue.size()) {
                Queue<String> tmp = beginQueue;
                beginQueue = endQueue;
                endQueue = tmp;
            }
            int size = beginQueue.size();
            step++;
            for (int i = 0; i < size; i++) {
                String cur = beginQueue.poll();
                if (updateWord2(cur, wordSet, endQueue, beginQueue, visited)) {
                    return step + 1;
                }
            }

        }
        return 0;
    }

    private static boolean updateWord2(String cur, Set<String> wordSet, Queue<String> endQueue, Queue<String> beginQueue, Set<String> visited) {
        char[] array = cur.toCharArray();
        for (int i = 0; i < array.length; i++) {
            char oldChar = array[i];
            for (char j = 'a'; j <= 'z'; j++) {
                if (oldChar == j) {
                    continue;
                }
                array[i] = j;
                String next = String.valueOf(array);
                if (wordSet.contains(next)) {
                    if (endQueue.contains(next)) {
                        return true;
                    }
                    if (!visited.contains(next)) {
                        visited.add(next);
                        beginQueue.offer(next);
                    }
                }
            }
            array[i] = oldChar;
        }
        return false;
    }

    public int maxProductDifference(int[] nums) {
        Arrays.sort(nums);
        int len = nums.length;
        return nums[len - 1] * nums[len - 2] - nums[0] * nums[1];
    }


    public static void main(String[] args) {
//        System.out.println("hello");
        String[] r = new String[]{"hot", "dot", "dog", "lot", "log", "cog"};
//        String[] r = new String[]{"hot", "dog"};

        List<String> res = new ArrayList<String>(Arrays.asList(r));
        int res2 = ladderLength3("hot", "dog", res);
        System.out.println(res2);
    }


}



