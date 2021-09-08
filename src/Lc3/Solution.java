package Lc3;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.Queue;
import java.util.*;

class Solution {
    public String longestWord(String[] words) {
        Trie trie = new Trie();
        int index = 0;
        for (String word : words) {
            trie.insert(word, ++index); //indexed by 1
        }
        trie.words = words;
        return trie.dfs();
    }

    int[][] memo; // 记忆数组

//    public int minDifficulty2(int[] jobDifficulty, int d) {
//        // 如果d大于数组元素个数，无法分组，返回-1
//        int INF = 0x3f3f3f3f;
//        if (d > jobDifficulty.length) return -1;
//        int[][] dp =new int[jobDifficulty.length+1][d+1];
//        for (int i = 0; i < jobDifficulty.length; i++) {
//            Arrays.fill(dp[i],INF);
//        }
//        dp[0][0]=0;
//        for (int i = 1; i <=jobDifficulty.length ; i++) {
//            for (int j = 1; j <=d ; j++) {
//                int md = 0;
//                for (int k = i-1; k >=j-1 ; k--) {
//                    md=Math.max(md,jobDifficulty[k]);
//                    dp[i][j]=Math.min(dp[i][j],dp[k][j-1]+md);
//                }
//            }
//        }
//        return dp[jobDifficulty.length][d];
//    }
    public int minDifficulty(int[] jobDifficulty, int d) {
        int n = jobDifficulty.length;
        if (d > n) return -1;
        int[][] F = new int[d+1][n+1];
        for (int i = 1; i <= n; i++) F[1][i] = Math.max(F[1][i-1], jobDifficulty[i-1]);
        for (int i = 2; i <= d; i++) {
            for (int j = i; j <= n; j++) {
                F[i][j] = Integer.MAX_VALUE;
                int currMax = 0;
//                for (int k = j; k >= i; k--) {
//                    currMax = Math.max(currMax, jobDifficulty[k-1]);
//                    F[i][j] = Math.min(F[i][j], F[i-1][k-1] + currMax);
//                }
                for (int k = i; k <= j; k++) {
                    currMax = Math.max(currMax, jobDifficulty[k-1]);
                    F[i][j] = Math.min(F[i][j], F[i-1][k-1] + currMax);
                }
            }
        }
        return F[d][n];
    }

    public static void main(String[] args) {
        Solution s = new Solution();
        int[] s1 = {6,5,4,3,2,1};
        s.minDifficulty(s1,2);
    }

    // jobDifficulty：原始数组
// d：剩余尚未分组个数
// job：当前开始job下标
// 返回值：将下标job开始到数组结尾区间分成d组，得到的最小难度值
    int help(int[] jobDifficulty, int d, int job) {
        // 如果记忆数组中存在该值，直接返回
        if (memo[d][job] > 0) return memo[d][job];
        // 当前区间内最大值
        int maxDifficult = 0;
        // 返回结果
        int res = Integer.MAX_VALUE;
        // 当前区间最大结束坐标
        int end = jobDifficulty.length - d;
        // 循环每一个结束坐标
        for (int i = job; i <= end; i++) {
            // 更新当前区间内最大值
            maxDifficult = Math.max(maxDifficult, jobDifficulty[i]);
            // 如果所剩分组个数大于1，继续递归分组
            if (d > 1) {
                // 当前区间最大值加上子问题的结果，为当前解，
                // 利用当前解更新最优解
                res = Math.min(res, maxDifficult + help(jobDifficulty, d - 1, i + 1));
            }
        }
        // 如果尚未分组个数为1，返回当前区间内最大值
        if (d == 1) res = maxDifficult;
        // 将当前最优解保存至记忆数组
        memo[d][job] = res;
        return res;
    }
    public int compress(char[] chars) {
        StringBuilder sb = new StringBuilder();
        int pre =0,cur =0,len=chars.length;
        while(cur<len){
            while(cur<len&&chars[cur]==chars[pre]){
                cur++;
            }
            sb.append(chars[pre]);
            if(pre!=cur-1){
                sb.append(cur-pre);
            }
            pre=cur;
        }
        for(int i=0;i<sb.length();i++){
            chars[i]=sb.charAt(i);
        }
        return sb.length();
    }
    // 去看看这样的应该怎么处理，用int array？？？？
    public boolean escapeGhosts(int[][] ghosts, int[] target) {
        Queue<String> queue = new LinkedList<>();
        String tar = target[0]+":"+target[1];
        String end = 0+":"+0;
        queue.offer(tar);
        Set<String> set =new HashSet<>();
        List<String> glist = new ArrayList<>();
        int[][] directions = {{0, 1}, {-1, 0}, {0, -1}, {1, 0}};
        for(int[] each:ghosts){
            glist.add(each[0]+":"+each[1]);
        }
        while(!queue.isEmpty()){
            if(queue.contains(end)){
                return true;
            }
            String cur =queue.poll();
            if(set.contains(cur))continue;
            set.add(cur);
            if(glist.contains(cur)) return false;
            String[] s=cur.split(":");
            int x = Integer.parseInt(s[0]);
            int y = Integer.parseInt(s[1]);
            for(int[]d:directions){
                int x1 = x+d[0],y1=y+d[1];
                String next = x1+":"+y1;
                queue.offer(next);
            }
        }
        return false;
    }
//    public int minDifficulty(int[] jobDifficulty, int d) {
//        // 如果d大于数组元素个数，无法分组，返回-1
//        if (d > jobDifficulty.length) return -1;
//        // 初始化记忆数组
//        memo = new int[d + 1][jobDifficulty.length];
//        // 递归求解
//        return help(jobDifficulty, d, 0);
//    }
//
//    public static void main(String[] args) {
//        Solution s = new Solution();
//        char[] s1 = {'a'};
//        s.compress(s1);
//    }

}

class Node {
    char c;
    HashMap<Character, Node> children = new HashMap();
    int end;

    public Node(char c) {
        this.c = c;
    }
}

class Trie {
    Node root;
    String[] words;

    public Trie() {
        root = new Node('0');
    }

    public void insert(String word, int index) {
        Node cur = root;
        for (char c : word.toCharArray()) {
            cur.children.putIfAbsent(c, new Node(c));
            cur = cur.children.get(c);
        }
        cur.end = index;
    }

    public String dfs() {
        String ans = "";
        Stack<Node> stack = new Stack();
        stack.push(root);
        while (!stack.empty()) {
            Node node = stack.pop();
            if (node.end > 0 || node == root) {
                if (node != root) {
                    String word = words[node.end - 1];
                    if (word.length() > ans.length() ||
                            word.length() == ans.length() && word.compareTo(ans) < 0) {
                        ans = word;
                    }
                }
                for (Node nei : node.children.values()) {
                    stack.push(nei);
                }
            }
        }
        return ans;
    }
}
