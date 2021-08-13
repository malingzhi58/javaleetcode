import java.util.*;

public class Lc4 {
    class Node {
        String str;
        int val, step;

        Node(String _str, int _val, int _step) {
            str = _str;
            val = _val;
            step = _step;
        }
    }

    String s, t;
    Set<String> set = new HashSet<>();

    public int openLock(String[] ds, String _t) {
        s = "0000";
        t = _t;
        if (s.equals(t)) return 0;
        set.addAll(Arrays.asList(ds));
        if (set.contains(s)) return -1;
        PriorityQueue<Node> pq = new PriorityQueue<>();
        Map<String, Node> map = new HashMap<>();
        Node root = new Node(s, f(s), 0);
        pq.add(root);
        map.put(s, root);
        while (!pq.isEmpty()) {
            Node poll = pq.poll();
            int step = poll.step;
            if (poll.str.equals(t)) return step;
            for (String each : get(poll.str)) {
                if (set.contains(each)) continue;
                // 如果 each 还没搜索过，或者 each 的「最短距离」被更新，则入队
                if (!map.containsKey(each) || map.get(each).step > step + 1) {
                    Node node = new Node(each, step + 1 + f(each), step + 1);
                    map.put(each, node);
                    pq.add(node);
                }
            }
        }
        return -1;
    }

    private int f(String s) {
        int res = 0;
        for (int i = 0; i < s.length(); i++) {
            int a = s.charAt(i) - '0', b = t.charAt(i) - '0';
            int ma = Math.max(a, b), mi = Math.min(a, b);
            res += Math.min(ma - mi, mi + 10 - ma);
        }
        return res;
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


    // 本来想使用如下逻辑将「所有可能用到的状态」打表，实现 O(1) 查询某个状态有多少个字符，但是被卡了
    // static int N = 26, M = (1 << N);
    // static int[] cnt = new int[M];
    // static {
    //     for (int i = 0; i < M; i++) {
    //         for (int j = 0; j < 26; j++) {
    //             if (((i >> j) & 1) == 1) cnt[i]++;
    //         }
    //     }
    // }

    static Map<Integer, Integer> map = new HashMap<>();

    int get(int cur) {
        if (map.containsKey(cur)) {
            return map.get(cur);
        }
        int ans = 0;
        for (int i = cur; i > 0; i -= lowbit(i)) ans++;
        map.put(cur, ans);
        return ans;
    }

    int lowbit(int x) {
        return x & -x;
    }

    int n;
    int ans = Integer.MIN_VALUE;
    int[] hash;

    public int maxLength(List<String> _ws) {
        n = _ws.size();
        HashSet<Integer> set = new HashSet<>();
        for (String s : _ws) {
            int val = 0;
            for (char c : s.toCharArray()) {
                int t = (int) (c - 'a');
                if (((val >> t) & 1) != 0) {
                    val = -1;
                    break;
                }
                val |= (1 << t);
            }
            if (val != -1) set.add(val);
        }

        n = set.size();
        if (n == 0) return 0;
        hash = new int[n];

        int idx = 0;
        int total = 0;
        for (Integer i : set) {
            hash[idx++] = i;
            total |= i;
        }
        dfs(0, 0, total);
        return ans;
    }

    void dfs(int u, int cur, int total) {
        if (get(cur | total) <= ans) return;
        if (u == n) {
            ans = Math.max(ans, get(cur));
            return;
        }
        // 在原有基础上，选择该数字（如果可以）
        if ((hash[u] & cur) == 0) {
            dfs(u + 1, hash[u] | cur, total - (total & hash[u]));
        }
        // 不选择该数字
        dfs(u + 1, cur, total);
    }
}
