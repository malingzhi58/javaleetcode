package Lc2;
import Lc1.Lc1;

import java.util.*;

public class Lc2 {
    class Node {
        String str;
        int val;

        Node(String _str, int _val) {
            str = _str;
            val = _val;
        }
    }

    String s, e;
    int INF = 0x3f3f3f3f;
    Set<String> set = new HashSet<>();

    public int ladderLength(String _s, String _e, List<String> ws) {
        s = _s;
        e = _e;
        for (String w : ws) set.add(w);
        if (!set.contains(e)) return 0;
        int ans = astar();
        return ans == -1 ? 0 : ans + 1;
    }

    int astar() {
        PriorityQueue<Node> q = new PriorityQueue<>((a, b) -> a.val - b.val);
        Map<String, Integer> dist = new HashMap<>();
        dist.put(s, 0);
        q.add(new Node(s, f(s)));

        while (!q.isEmpty()) {
            Node poll = q.poll();
            String str = poll.str;
            int distance = dist.get(str);
            if (str.equals(e)) {
                break;
            }
            int n = str.length();
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < 26; j++) {
                    String sub = str.substring(0, i) + String.valueOf((char) ('a' + j)) + str.substring(i + 1);
                    if (!set.contains(sub)) continue;
                    if (!dist.containsKey(sub) || dist.get(sub) > distance + 1) {
                        dist.put(sub, distance + 1);
                        q.add(new Node(sub, dist.get(sub) + f(sub)));
                    }
                }
            }
        }
        return dist.containsKey(e) ? dist.get(e) : -1;
    }

    int f(String str) {
        if (str.length() != e.length()) return INF;
        int n = str.length();
        int ans = 0;
        for (int i = 0; i < n; i++) {
            ans += str.charAt(i) == e.charAt(i) ? 0 : 1;
        }
        return ans;
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
        Queue<String> beginQueue = new LinkedList<>();
        Queue<String> endQueue = new LinkedList<>();
//        Set<String> seen = new HashSet<>(Arrays.asList("0000",target));
        Set<String> seen = new HashSet<>();
        beginQueue.offer("0000");
        endQueue.offer(target);
        while (!beginQueue.isEmpty() && !endQueue.isEmpty()) {
            if (endQueue.size() < beginQueue.size()) {
                Queue<String> tmp = beginQueue;
                beginQueue = endQueue;
                endQueue = tmp;
            }
            step++;
            int size = beginQueue.size();
            for (int i = 0; i < size; i++) {
                String cur = beginQueue.poll();
                for (String next : get(cur)) {
                    if (endQueue.contains(next)) {
                        return step;
                    }
                    if (!seen.contains(next) && !deadSet.contains(next)) {
                        seen.add(next);
                        beginQueue.offer(next);
//                        System.out.println(next);
                    }
                }
            }
        }
        return 0;
    }

    public int openLock2(String[] deadends, String target) {
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







    public static void main(String[] args) {
        String[] r = new String[]{"hot", "dot", "dog", "lot", "log", "cog"};
        List<String> res = new ArrayList<String>(Arrays.asList(r));
        Lc2 lc2 = new Lc2();
        int res2 = lc2.openLock(new String[]{"0201","0101","0102","1212","2002"},"0202");
//        int res2 = ladderLength("hot", "dog", res);
        System.out.println(res2);
//        TreeNode treeNode =
    }
}
