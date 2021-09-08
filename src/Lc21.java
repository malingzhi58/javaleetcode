import java.io.*;
import java.nio.ByteBuffer;
import java.time.Instant;
import java.util.*;
import java.text.*;
import java.math.*;
import java.util.regex.*;

public class Lc21 {
    public int minimumMoves(int[] arr) {
        int len = arr.length;
        int[][] dp = new int[len][len];
        for (int i = 0; i < len; i++) {
            dp[i][i] = 1;
        }
        for (int j = 0; j < len; j++) {
            for (int i = j - 1; i >= 0; i--) {
                if (i == j - 1) {
                    dp[i][j] = arr[i] == arr[j] ? 1 : 2;
                    continue;
                }
                int min = Integer.MAX_VALUE;
                if (arr[i] == arr[j]) {
                    min = Math.min(min, dp[i + 1][j - 1]);
                }
                for (int k = i; k < j; k++) {
                    min = Math.min(min, dp[i][k] + dp[k + 1][j]);
                }
                dp[i][j] = min;
            }
        }
        return dp[0][len - 1];

    }

    public int countPairs(int[] deliciousness) {
        Map<Integer, Integer> map = new HashMap<>();
        long sum = 0;
        int MOD = (int) 1e9 + 7;
        for (int each : deliciousness) {
            map.put(each, map.getOrDefault(each, 0) + 1);
        }
        Set<Integer> seen = new HashSet<>();
        for (int each : map.keySet()) {
            for (int i = 22; i >= 0; i--) {
                int target = (int) Math.pow(2, i) - each;
                if (target < each) break;
                if (map.containsKey(target) && (target != each)) {
                    sum = (sum + (map.get(target) % MOD) * (map.get(each) % MOD)) % MOD;
                }
                if (map.containsKey(target) && target == each && map.get(each) > 1) {
                    sum = (sum + (map.get(target) % MOD) * ((map.get(target) - 1) % MOD) / 2) % MOD;
                }
            }
        }
        return (int) (sum % MOD);
    }

    public String validateIPv4(String IP) {
        String[] nums = IP.split("\\.", -1);
        for (String x : nums) {
            // Validate integer in range (0, 255):
            // 1. length of chunk is between 1 and 3
            if (x.length() == 0 || x.length() > 3) return "Neither";
            // 2. no extra leading zeros
            if (x.charAt(0) == '0' && x.length() != 1) return "Neither";
            // 3. only digits are allowed
            for (char ch : x.toCharArray()) {
                if (!Character.isDigit(ch)) return "Neither";
            }
            // 4. less than 255
            if (Integer.parseInt(x) > 255) return "Neither";
        }
        return "IPv4";
    }

    public String validateIPv6(String IP) {
        String[] nums = IP.split(":", -1);
        String hexdigits = "0123456789abcdefABCDEF";
        for (String x : nums) {
            // Validate hexadecimal in range (0, 2**16):
            // 1. at least one and not more than 4 hexdigits in one chunk
            if (x.length() == 0 || x.length() > 4) return "Neither";
            // 2. only hexdigits are allowed: 0-9, a-f, A-F
            for (Character ch : x.toCharArray()) {
                if (hexdigits.indexOf(ch) == -1) return "Neither";
            }
        }
        return "IPv6";
    }

    public String validIPAddress(String IP) {
        if (IP.chars().filter(ch -> ch == '.').count() == 3) {
            return validateIPv4(IP);
        } else if (IP.chars().filter(ch -> ch == ':').count() == 7) {
            return validateIPv6(IP);
        } else return "Neither";
    }

    //    public int makeConnected(int n, int[][] connections) {
//        int len = connections.length;
//        if(len<n-1) return -1;
//        int res = 0;
//        Map<Integer,Integer> map = new HashMap<>();
//        for (int i = 0; i < n; i++) {
//            map.put(i,0);
//        }
//        for(int[] each:connections){
//            map.put(each[0],map.get(each[0])+1);
//            map.put(each[1],map.get(each[1])+1);
//        }
//        for(int each:map.keySet()){
//            if(map.get(each)==0){
//                res++;
//            }
//        }
//        return res;
//    }
    public int makeConnected(int n, int[][] connections) {
        if (connections.length < n - 1) {
            return -1;
        }
        UnionFind2 uf = new UnionFind2(n);
        for (int[] conn : connections) {
            uf.unite(conn[0], conn[1]);
        }
        return uf.setsize - 1;
    }

    class UnionFind2 {
        int[] parent;
        int[] size;
        int setsize;
        int n;

        public UnionFind2(int n) {
            this.n = n;
            parent = new int[n];
            for (int i = 0; i < n; i++) {
                parent[i] = i;
            }
            size = new int[n];
            Arrays.fill(size, 1);
            setsize = n;
        }

        public boolean unite(int x, int y) {
            x = find(x);
            y = find(y);
            if (x == y) return false;
            if (size[x] < size[y]) {
                int tmp = x;
                x = y;
                y = tmp;
            }
            parent[y] = x;
            size[x] += size[y];
            setsize--;
            return true;
        }

        private int find(int x) {
            return parent[x] == x ? x : (parent[x] = find(parent[x]));
        }
    }

    public List<List<String>> accountsMerge(List<List<String>> accounts) {
        Map<String, String> emialToName = new HashMap<>();
        Map<String, Integer> emialToIdx = new HashMap<>();
        int idx = 0;
        for (List<String> each : accounts) {
            String name = each.get(0);
            for (int i = 1; i < each.size(); i++) {
                String now = each.get(i);
                if (!emialToIdx.containsKey(now)) {
                    emialToIdx.put(each.get(i), idx++);
                    emialToName.put(each.get(i), name);
                }
            }
        }
        UnionFind3 unionFind3 = new UnionFind3(idx);
        for (List<String> each : accounts) {
            String first = each.get(1);
            int firstId = emialToIdx.get(first);
            for (int i = 2; i < each.size(); i++) {
                String nex = each.get(i);
                int nexid = emialToIdx.get(nex);
                unionFind3.union(firstId, nexid);
            }
        }
        Map<Integer, List<String>> emails = new HashMap<>();
        for (String each : emialToIdx.keySet()) {
            int id = unionFind3.get(emialToIdx.get(each));
            if (!emails.containsKey(id)) {
                emails.put(id, new ArrayList<>());
            }
            List<String> list = emails.get(id);
            list.add(each);
        }

        List<List<String>> res = new ArrayList<>();
        for (List<String> each : emails.values()) {
            String name = emialToName.get(each.get(0));
            Collections.sort(each);
            each.add(0, name);
            res.add(each);
        }

        return res;
    }

    class UnionFind3 {
        int setsize;
        int[] parent;
        int[] size;
        int n;

        public UnionFind3(int n) {
            this.n = n;
            parent = new int[n];
            for (int i = 0; i < n; i++) {
                parent[i] = i;
            }
            size = new int[n];
            Arrays.fill(size, 1);
        }

        public int find(int n) {
            return parent[n] == n ? n : (parent[n] = find(parent[n]));
        }

        public void union(int firstId, int nexid) {
            firstId = find(firstId);
            nexid = find(nexid);
            if (firstId == nexid) return;
            if (size[firstId] < size[nexid]) {
                int tmp = nexid;
                nexid = firstId;
                firstId = tmp;
            }
            parent[nexid] = firstId;
            size[firstId] += size[nexid];
            return;
        }

        public int get(int n) {
            return parent[n];
        }
    }

    //    public List<List<String>> accountsMerge2(List<List<String>> accounts) {
//        Map<String, Integer> emailToIndex = new HashMap<String, Integer>();
//        Map<String, String> emailToName = new HashMap<String, String>();
//        int emailsCount = 0;
//        for (List<String> account : accounts) {
//            String name = account.get(0);
//            int size = account.size();
//            for (int i = 1; i < size; i++) {
//                String email = account.get(i);
//                if (!emailToIndex.containsKey(email)) {
//                    emailToIndex.put(email, emailsCount++);
//                    emailToName.put(email, name);
//                }
//            }
//        }
//        UnionFind uf = new UnionFind(emailsCount);
//        for (List<String> account : accounts) {
//            String firstEmail = account.get(1);
//            int firstIndex = emailToIndex.get(firstEmail);
//            int size = account.size();
//            for (int i = 2; i < size; i++) {
//                String nextEmail = account.get(i);
//                int nextIndex = emailToIndex.get(nextEmail);
//                uf.union(firstIndex, nextIndex);
//            }
//        }
//        Map<Integer, List<String>> indexToEmails = new HashMap<Integer, List<String>>();
//        for (String email : emailToIndex.keySet()) {
//            int index = uf.find(emailToIndex.get(email));
//            List<String> account = indexToEmails.getOrDefault(index, new ArrayList<String>());
//            account.add(email);
//            indexToEmails.put(index, account);
//        }
//        List<List<String>> merged = new ArrayList<List<String>>();
//        for (List<String> emails : indexToEmails.values()) {
//            Collections.sort(emails);
//            String name = emailToName.get(emails.get(0));
//            List<String> account = new ArrayList<String>();
//            account.add(name);
//            account.addAll(emails);
//            merged.add(account);
//        }
//        return merged;
//    }
    public int findCircleNum(int[][] isConnected) {
        UnionFind unionFind = new UnionFind(isConnected.length);
        for (int i = 0; i < isConnected.length; i++) {
            for (int j = 0; j < isConnected[i].length; j++) {
                if (i == j) continue;
                if (isConnected[i][j] == 1) {
                    unionFind.unite(i, j);
                }
            }
        }
        return unionFind.setCount;
    }

    public int countComponents(int n, int[][] edges) {
        UnionFind unionFind = new UnionFind(n);
        for (int[] each : edges) {
            unionFind.unite(each[0], each[1]);
        }
        return unionFind.setCount;
    }

    class UnionFind {
        int[] parent;
        int[] size;
        int n;
        int setCount;

        public UnionFind(int n) {
            this.n = n;
            this.setCount = n;
            this.parent = new int[n];
            this.size = new int[n];
            Arrays.fill(size, 1);
            for (int i = 0; i < n; ++i) {
                parent[i] = i;
            }
        }

        public int findset(int x) {
            return parent[x] == x ? x : (parent[x] = findset(parent[x]));
        }

        public boolean unite(int x, int y) {
            x = findset(x);
            y = findset(y);
            if (x == y) {
                return false;
            }
            if (size[x] < size[y]) {
                int temp = x;
                x = y;
                y = temp;
            }
            parent[y] = x;
            size[x] += size[y];
            --setCount;
            return true;
        }

        public boolean connected(int x, int y) {
            x = findset(x);
            y = findset(y);
            return x == y;
        }
    }

    public int maxResult(int[] nums, int k) {
        int len = nums.length;
        int[] dp = new int[len];
        int INF = 0x3f3f3f3f;
        Arrays.fill(dp, -INF);
        dp[0] = nums[0];
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> (b[1] - a[1]));
        pq.offer(new int[]{0, dp[0]});
        for (int i = 1; i < len; i++) {
            while (pq.peek()[0] < i - k) {
                pq.poll();
            }
            dp[i] = pq.peek()[1] + nums[i];
            pq.offer(new int[]{i, dp[i]});
        }
        return dp[len - 1];
    }


    /**
     * A class which constructs a view of the Hash Table's state given the input DML Events.
     */
    public static class HashTable {
        Map<String, String> map = new HashMap<>();
        String waterMark = "0";

        public HashTable(ArrayList<String> rawEvents) {
            // TODO
            for (int i = 0; i < rawEvents.size(); i++) {
                String cur = rawEvents.get(i);
                String[] curArr = cur.split("\\|");
                String key = curArr[2];
                if (curArr[1].equals("INSERT")) {
                    if (!map.containsKey(key)) {
                        map.put(key, curArr[3]);
                    }
                    waterMark = curArr[0];

                }
                if (curArr[1].equals("UPSERT")) {
                    map.put(key, curArr[3]);
                    waterMark = curArr[0];
                }
                if (curArr[1].equals("DELETE")) {
                    if (map.containsKey(key)) {
                        map.remove(key);
                    }
                    waterMark = curArr[0];
                }

            }
        }

        /**
         * Retrieve the state of the HashTable after applying all input events
         *
         * @return a Map representing the Keys and Values of the current state
         */
        public Map<String, String> getTable() {
            return map;  // TODO
        }

        /**
         * Retrieve the high-watermark of the HashTable as the millisecond epoch timestamp
         * of the latest event read or Instant.EPOCH in the case where no event occurred.
         *
         * @return an Instant representing the high watermark
         */
        public Instant getHighWatermark() {
//            Date date = null;
//            try {
//                date = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'").parse(waterMark);
////                SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'");
////                2019-07-18T13:03:04.003Z
//            } catch (ParseException e) {
//                e.printStackTrace();
//            }
//            SimpleDateFormat sf = new SimpleDateFormat("yyyy-MM-dd");
            SimpleDateFormat sf = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'");
            Date date = new Date(Long.parseLong(waterMark));
//            System.out.println(sf.format(date));
            return date.toInstant();  // TODO
        }
    }

    /**
     * A class which wraps a raw binary WAL and reconstructs DML Events.
     */
    public static class WAL {
        /**
         * Construct the WAL
         *
         * @param input the raw binary WAL
         */
        ArrayList<String> res = new ArrayList<>();

        public WAL(byte[] input) {
            // TODO
            int id = 0;
            while (id < input.length) {
                byte[] time = Arrays.copyOfRange(input, id, id + 8);
                ByteBuffer bb = ByteBuffer.wrap(time);
                long l = bb.getLong();
                id+=8;
//                System.out.println(l);
                byte mid =input[id];
                int type = mid & 0xFF;
                id++;
//                System.out.println(type);
                String msgName="";
                if(type==0){
                    msgName="INSERT";
                    short s = (short) ((input[id] & 0xFF) << 8 | (input[id+1] & 0xFF));
                    id+=2;
                    byte[] keybyte = Arrays.copyOfRange(input, id, id + s);
                    String key = new String(keybyte);
                    id+=s;
                    short s2 = (short) ((input[id] & 0xFF) << 8 | (input[id+1] & 0xFF));
                    id+=2;
                    byte[] valbyte = Arrays.copyOfRange(input, id, id + s2);
                    String val = new String(valbyte);
                    String result =l+"|"+msgName+"|"+key+"|"+val;
                    res.add(result);
                }

            }

        }

        /**
         * Retrieve all events contained within the WAL as their string values in time order
         * DML Event String Format: "<epoch_milli>|<message_name>|<key>|<value>"
         *
         * @return a time-ordered sequence of DML Event strings
         */
        public ArrayList<String> getEvents() {
            // TODO
            return res;
        }
    }

    public static void main(String[] args) {
        Lc21 lc21 = new Lc21();
        int[] s1 = {1, 3, 5, 7, 9};
        int[] s2 = {1, 1, 1, 3, 3, 3, 7};
//        int r1 = lc21.countPairs(s2);
//        System.out.println(r1);
//        String cmdline = "commd /Ab11601/d/-a22";
//        cmdline = cmdline.trim();
////        Pattern p = Pattern.compile("\\w+ (\\d+) [\\w ]+ (\\d{2}:\\d{2})");
//        Pattern p = Pattern.compile("\\/([-]?[a-zA-Z]*)");
//        Matcher m = p.matcher(cmdline);
//        while (m.find()) {
//            String cur = m.group(1);
//            if (cur.startsWith("-")) {
//
//            }
//        }
//        String[] split = cmdline.split("\\/([a-zA-Z]*)");
//        System.out.println(Arrays.toString(split));
//        System.out.println(split.length);
        int[][] s3 = {{0, 1}, {0, 2}, {1, 2}};
//        int r3 = lc21.makeConnected(4, s3);
//        System.out.println(r3);

        String[][] s4 = {{"David", "David0@m.co", "David1@m.co"}, {"David", "David3@m.co", "David4@m.co"}, {"David", "David4@m.co", "David5@m.co"}, {"David", "David2@m.co", "David3@m.co"}, {"David", "David1@m.co", "David2@m.co"}};
//        lc21.accountsMerge2(Arrays.asList(Arrays.asList("David","David0@m.co","David1@m.co"),Arrays.asList("David","David3@m.co","David4@m.co"),Arrays.asList("David","David4@m.co","David5@m.co"),Arrays.asList("David","David2@m.co","David3@m.co"),Arrays.asList("David","David1@m.co","David2@m.co")));
        int[][] s5 = {{1, 1, 0}, {1, 1, 0}, {0, 0, 1}};
//        lc21.findCircleNum(s5);

        int[] s6 = {1, -1, -2, 4, -7, 3};
//        lc21.maxResult(s6, 2);
        ArrayList<String> s7 = new ArrayList<>(Arrays.asList("1563454984001|INSERT|test|123", "1563454984002|INSERT|test_2|234", "1563454984003|INSERT|test_3|345"));
//        HashTable hashTable =new HashTable(s7);
//        hashTable.getHighWatermark();
//        0000016c052dcf4100000e746573745f6b65795f30393831320010746573745f76616c75655f3132383736
        String s8 = "1563454984001INSERTtest_key_09812test_value_12876";
        byte[] s88 = s8.getBytes();
        System.out.println(s88);
        //        byte[] s10 ={'0','c'}
//        byte[] s9 ={0,0,0,0,0,1,6,c,0,5,2,'d',c,f,4,1,0,0,0,0,0,e,7,4,6,5,7,3,7,4,5,f,6,b,6,5,7,9,5,f,3,0,3,9,3,8,3,1,3,2,0,0,1,0,7,4,6,5,7,3,7,4,5,f,7,6,6,1,6,c,7,5,6,5,5,f,3,1,3,2,3,8,3,7,3,6};
        byte[] s11 = {'0', '0', '0', '0', '0', '1', '6', 'c', '0', '5', '2', 'd', 'c', 'f', '4', '1', '0', '0', '0', '0', '0', 'e', '7', '4', '6', '5', '7', '3', '7', '4', '5', 'f', '6', 'b', '6', '5', '7', '9', '5', 'f', '3', '0', '3', '9', '3', '8', '3', '1', '3', '2', '0', '0', '1', '0', '7', '4', '6', '5', '7', '3', '7', '4', '5', 'f', '7', '6', '6', '1', '6', 'c', '7', '5', '6', '5', '5', 'f', '3', '1', '3', '2', '3', '8', '3', '7', '3', '6'};
        WAL wal = new WAL(s11);

//        0,0,0,0,0,1,6,c,0,5,2,d,c,f,4,1,0,0,0,0,0,e,7,4,6,5,7,3,7,4,5,f,6,b,6,5,7,9,5,f,3,0,3,9,3,8,3,1,3,2,0,0,1,0,7,4,6,5,7,3,7,4,5,f,7,6,6,1,6,c,7,5,6,5,5,f,3,1,3,2,3,8,3,7,3,6
        String tmp = "0000016c052dcf4100000e746573745f6b65795f30393831320010746573745f76616c75655f3132383736";
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < tmp.length(); i++) {
            sb.append("'");
            sb.append(tmp.charAt(i));
            sb.append("'");
            sb.append(",");
        }
        System.out.println(sb);

    }
}


class UnionFind {
    int[] parent;

    public UnionFind(int n) {
        parent = new int[n];
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
    }

    public void union(int index1, int index2) {
        parent[find(index2)] = find(index1);
    }

    public int find(int index) {
        if (parent[index] != index) {
            parent[index] = find(parent[index]);
        }
        return parent[index];
    }
}
//
//class Solution {
//    public int makeConnected(int n, int[][] connections) {
//        if (connections.length < n - 1) {
//            return -1;
//        }
//
//        UnionFind uf = new UnionFind(n);
//        for (int[] conn : connections) {
//            uf.unite(conn[0], conn[1]);
//        }
//
//        return uf.setCount - 1;
//    }
//}
//
//// 并查集模板
//class UnionFind {
//    int[] parent;
//    int[] size;
//    int n;
//    // 当前连通分量数目
//    int setCount;
//
//    public UnionFind(int n) {
//        this.n = n;
//        this.setCount = n;
//        this.parent = new int[n];
//        this.size = new int[n];
//        Arrays.fill(size, 1);
//        for (int i = 0; i < n; ++i) {
//            parent[i] = i;
//        }
//    }
//
//    public int findset(int x) {
//        return parent[x] == x ? x : (parent[x] = findset(parent[x]));
//    }
//
//    public boolean unite(int x, int y) {
//        x = findset(x);
//        y = findset(y);
//        if (x == y) {
//            return false;
//        }
//        if (size[x] < size[y]) {
//            int temp = x;
//            x = y;
//            y = temp;
//        }
//        parent[y] = x;
//        size[x] += size[y];
//        --setCount;
//        return true;
//    }
//
//    public boolean connected(int x, int y) {
//        x = findset(x);
//        y = findset(y);
//        return x == y;
//    }
//}


class Solution8 {
    static class ParsedCommand {
        ArrayList<String> argv;
        Set<String> switches;

        ParsedCommand() {
            argv = new ArrayList<String>();
            switches = new TreeSet<String>();
        }
    }

    ;

    /*
     * Complete the function below.
     */

    static ParsedCommand parseCommand(String cmdline) {
        ParsedCommand res = new ParsedCommand();
        cmdline = cmdline.trim();
        String[] split = cmdline.split("\\/([-]?[a-zA-Z]*)");
        List<String> args = new ArrayList<>();
        for (int i = 0; i < split.length; i++) {
            String[] tmp = split[i].split(" ");
            for (int j = 0; j < tmp.length; j++) {
                if (tmp[j].length() > 0) {
                    args.add(tmp[j]);
                }
            }
        }
        if (args.size() == 0) return null;
        res.argv.addAll(args);
        Pattern p = Pattern.compile("\\/([-]?[a-zA-Z]*)");
        Matcher m = p.matcher(cmdline);
        while (m.find()) {
            String cur = m.group(1);
            if (cur.startsWith("-")) {
                for (int i = 1; i < cur.length(); i++) {
                    res.switches.remove(String.valueOf(cur.charAt(i)).toUpperCase());
                }
            } else {
                for (int i = 0; i < cur.length(); i++) {
                    res.switches.add(String.valueOf(cur.charAt(i)).toUpperCase());
                }
            }
        }

        return res;
    }

    private static void writeCommand(BufferedWriter bw, ParsedCommand pc) throws IOException {
        if (pc == null) {
            bw.write("Invalid Command Line");
            bw.newLine();
            return;
        }
        if (pc.argv == null || pc.argv.size() == 0) {
            bw.write("Invalid ParsedCommand - empty argv");
            bw.newLine();
            return;
        }
        for (int i = 0; i < pc.argv.size(); i++) {
            bw.write("argv[" + String.valueOf(i) + "]=\"" + pc.argv.get(i) + "\"");
            bw.newLine();
        }
        for (String sw : pc.switches) {
            bw.write("/" + sw);
            bw.newLine();
        }
    }

    public static void main(String[] args) throws IOException {
        Scanner in = new Scanner(System.in);
        final String fileName = System.getenv("OUTPUT_PATH");
        BufferedWriter bw = null;
        if (fileName != null) {
            bw = new BufferedWriter(new FileWriter(fileName));
        } else {
            bw = new BufferedWriter(new OutputStreamWriter(System.out));
        }

        String cmdline;
        try {
//            cmdline = in.nextLine();
            cmdline = "/Ab11601 rulez";
        } catch (Exception e) {
            cmdline = null;
        }

        writeCommand(bw, parseCommand(cmdline));
        bw.close();
    }
}