import javafx.util.Pair;
import org.omg.CORBA.INTERNAL;

import java.lang.reflect.Array;
import java.util.*;

public class Lc20 {

    public int minSessions(int[] tasks, int sessionTime) {
        int len = tasks.length, n = 1 << len;
        int[] dp = new int[n];
        int INF = 0x3f3f3f3f;
        Arrays.fill(dp, INF);
        for (int i = 1; i < n; i++) {
            int state = i, id = 0, time = 0;
            while (state > 0) {
                if ((state & 1) == 1) {
                    time += tasks[id];
                }
                if (time > sessionTime) break;
                state >>= 1;
                id++;
            }
            if (time <= sessionTime) {
                dp[i] = 1;
            }
        }
        for (int i = 1; i < n; i++) {
            if (dp[i] == 1) continue;
            for (int j = i - 1; j > 0; j--) {
                int complement = i ^ j;
                if (dp[complement] != INF && dp[j] != INF) {
                    dp[i] = Math.min(dp[i], dp[j] + dp[complement]);
                }
            }
        }
        return dp[n - 1];
    }

    public int sumOddLengthSubarrays(int[] arr) {
        int sum = 0, len = arr.length;
        int[] preSum = new int[len + 1];
        int tmp = 0;
        for (int i = 0; i < len; i++) {
            tmp += arr[i];
            preSum[i + 1] = tmp;
        }
        for (int i = 1; i <= len; i += 2) {
            for (int j = i; j <= len; j++) {
                sum += (preSum[j] - preSum[j - i]);
            }
        }
        return sum;
    }

    int r1 = 0;

    public int numberWays(List<List<Integer>> hats) {
        dfs(hats, 0, 0);
        return r1;
    }

    private void dfs(List<List<Integer>> hats, int start, int mask) {
        if (start == hats.size()) {
            r1++;
            return;
        }
        List<Integer> list = hats.get(start);
        for (int i = 0; i < list.size(); i++) {
            if ((mask >> (list.get(i) - 1) & 1) != 1) {
                dfs(hats, start + 1, mask | (1 << (list.get(i) - 1)));
            }
        }
    }

    public int numberWays2(List<List<Integer>> hats) {
        int len = hats.size();
        int[][] dp = new int[41][(1 << len)];
        dp[0][0] = 1;
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int i = 0; i < hats.size(); i++) {
            List<Integer> people = hats.get(i);
            if (people == null) continue;
            for (int j = 0; j < people.size(); j++) {
                map.computeIfAbsent(people.get(j), k -> new ArrayList<>()).add(i);
            }
        }
        for (int i = 1; i < 41; i++) {
            for (int j = 0; j < (1 << len); j++) {
                dp[i][j] = dp[i - 1][j];
                List<Integer> people = map.get(i);
                for (int k = 0; k < people.size(); k++) {
                    int pos = people.get(k);
                    if ((j & (1 << pos)) > 0) {
                        dp[i][j] += dp[i - 1][j ^ (1 << pos)];
                    }
                }
            }
        }
        return dp[40][(1 << len) - 1];
    }

//    int result;
//    List<Integer> sessions = new ArrayList();
//    public int minSessions(int[] tasks, int sessionTime) {
//        result = tasks.length;
//        dfs(0, tasks, sessionTime);
//        return result;
//    }
//    public void dfs(int index, int[]  tasks, int sessionTime){
//        if(result <= sessions.size())
//            return ;
//
//        if(index == tasks.length){
//            result = sessions.size();
//            return;
//        }
//        for(int i =0 ; i < sessions.size();i++){
//            if(sessions.get(i) + tasks[index] <= sessionTime){
//                sessions.set(i, sessions.get(i) + tasks[index]);
//                dfs(index +1, tasks, sessionTime);
//                sessions.set(i, sessions.get(i) - tasks[index]);
//            }
//        }
//        sessions.add(tasks[index]);
//        dfs(index+1, tasks, sessionTime);
//        sessions.remove(sessions.size()-1);
//    }

    public int numberOfUniqueGoodSubsequences2(String binary) {
        Set<String> startOne = new HashSet<>();
        Set<String> zero = new HashSet<>();
        for (int i = 0; i < binary.length(); i++) {
            Set<String> newOne = new HashSet<>(startOne);
            for (String each : startOne) {
                newOne.add(each + binary.charAt(i));
            }
            startOne = newOne;
            if (zero.isEmpty() && binary.charAt(i) == '0') {
                zero.add("0");
            }
            if (startOne.isEmpty() && binary.charAt(i) == '1') {
                startOne.add("1");
            }
        }
        return startOne.size() + zero.size();
    }

    public int numberOfUniqueGoodSubsequences(String binary) {
        long[][] dp = new long[binary.length()][2];
        int MOD = (int) 1e9 + 7;
        if (binary.charAt(binary.length() - 1) == '1') {
            dp[binary.length() - 1][1] = 1;
        } else {
            dp[binary.length() - 1][0] = 1;
        }
        for (int i = binary.length() - 2; i >= 0; i--) {
            if (binary.charAt(i) == '0') {
                dp[i][0] = (dp[i + 1][0] + dp[i + 1][1] + 1) % MOD;
                dp[i][1] = dp[i + 1][1];
            } else {
                dp[i][0] = dp[i + 1][0];
                dp[i][1] = (dp[i + 1][1] + dp[i + 1][0] + 1) % MOD;
            }
        }

        return dp[0][0] != 0 ? (int) (dp[0][1] + 1) % MOD : (int) dp[0][1];
    }

    public int distinctSubseqII(String s) {
        int[] dp = new int[26];
        int MOD = (int) 1e9 + 7;
        for (int i = 0; i < s.length(); i++) {
            int cur = s.charAt(i) - 'a';
            for (int j = 0; j < 26; j++) {
                if (j == cur) {
                    dp[cur] = (dp[cur] + 1) % MOD;
                } else {
                    dp[cur] = (dp[cur] + dp[j]) % MOD;
                }
            }
        }
        int sum = 0;
        for (int i = 0; i < 26; i++) {
            sum = (sum + dp[i]) % MOD;
        }
//        return Arrays.stream(dp).sum()%MOD;
        return sum;
    }

    public int[] corpFlightBookings(int[][] bookings, int n) {
        int[] res = new int[n];
        for (int[] each : bookings) {
            for (int i = each[0]; i <= each[1]; i++) {
                res[i - 1] += each[2];
            }
        }
        return res;

    }

    public int findTargetSumWays(int[] nums, int S) {
        int sum = 0;
        for (int num : nums) sum += num;
        if (S > sum || S < -sum) return 0;
        int len = nums.length;
        int[][] dp = new int[len][2 * sum + 1];
        dp[0][sum - nums[0]]++;
        dp[0][sum + nums[0]]++;
        for (int i = 1; i < len; i++) {
            for (int j = 0; j < 2 * sum + 1; j++) {
                if (dp[i - 1][j] != 0) {
                    dp[i][j + nums[i]] += dp[i - 1][j];
                    dp[i][j - nums[i]] += dp[i - 1][j];
                }
            }
        }
        return dp[len - 1][sum + S];
    }

    public int findGCD(int[] nums) {

        return gcdByEuclidsAlgorithm(Arrays.stream(nums).max().getAsInt(), Arrays.stream(nums).min().getAsInt());
    }

    int gcdByEuclidsAlgorithm(int n1, int n2) {
        if (n2 == 0) {
            return n1;
        }
        return gcdByEuclidsAlgorithm(n2, n1 % n2);
    }

    public int eraseOverlapIntervals(int[][] intervals) {
//        Arrays.sort(intervals,(a,b)->{
//            if(a[0]!=b[0]) return a[0]-b[0];
//            else return a[1]-b[1];
//        });
        Arrays.sort(intervals, (a, b) -> {
            if (a[1] != b[1]) return a[1] - b[1];
            else return a[0] - b[0];
        });
        int pre = intervals[0][1], res = 0;
        for (int i = 1; i < intervals.length; i++) {
            if (intervals[i][0] >= pre) {
                res++;
                pre = intervals[i][1];
            }
        }
        return intervals.length - res;
    }

    public int lengthOfLIS(int[] nums) {
        TreeMap<Integer, Integer> map = new TreeMap<>();
        for (int i = 0; i < nums.length; i++) {
            int toput = 1;
            for (int each : map.keySet()) {
                if (each >= nums[i]) break;
                if (toput < map.get(each) + 1) {
                    toput = map.get(each) + 1;
                }
            }
            map.put(nums[i], toput);
        }
        int max = 0;
        for (int each : map.keySet()) {
            max = Math.max(max, map.get(each));
        }
        return max;
    }

    public int findMinArrowShots(int[][] points) {
        Arrays.sort(points, (a, b) -> {
            if (a[1] != b[1]) {
                if (a[1] > b[1]) return 1;
                else return -1;
            } else {
                return a[0] - b[0];
            }
        });
        int sum = 1, pre = points[0][1];
        for (int i = 1; i < points.length; i++) {
            if (points[i][0] > pre) {
                sum++;
                pre = points[i][1];
            }
        }
        return sum;
    }

    public int minEatingSpeed(int[] piles, int h) {
        int max = Arrays.stream(piles).max().getAsInt();
        int left = 1, right = max;
        while (left < right) {
            int mid = (right - left) / 2 + left;
            int hour = compute(piles, mid);
            if (hour > h) {
                left = mid + 1;

            } else {
                right = mid;
            }
        }
        return left;
    }


    private int compute(int[] piles, int mid) {
        int sum = 0;
        for (int i = 0; i < piles.length; i++) {
            sum += ((piles[i] + mid - 1) / mid);
        }
        return sum;
    }

    String strres = "";

    public String findDifferentBinaryString(String[] nums) {
        Set<String> set = new HashSet<>();
        for (String each : nums) {
            set.add(each);
        }
        StringBuilder sb = new StringBuilder();
        boolean res = dfs2(sb, nums[0].length(), set);
        return strres;
    }

    private boolean dfs2(StringBuilder sb, int length, Set<String> set) {
        if (sb.length() == length) {
            if (!set.contains(sb.toString())) {
                strres = sb.toString();
                return true;
            } else {
                return false;
            }
        }
        for (int i = 0; i < 1; i++) {
            sb.append(i);
            if (dfs2(sb, length, set)) {
                return true;
            }
            ;
            sb.deleteCharAt(sb.length() - 1);
        }
        return false;
    }

    public int numWays(String s) {
        int sum = 0;
        for (int i = 1; i < s.length() - 1; i++) {
            for (int j = i + 1; j < s.length(); j++) {
                String a = s.substring(0, i);
                String b = s.substring(i, j);
                String c = s.substring(j);
                int na = countOne(a);
                int nb = countOne(b);
                int nc = countOne(c);
//                    System.out.println(a+":"+b+":"+c);

                if (na == nb && nb == nc) {
//                    System.out.println(a+":"+b+":"+c);
                    sum++;
                }
            }
        }
        return sum;
    }

    private int countOne(String a) {
        int sum = 0;
        for (int i = 0; i < a.length(); i++) {
            if (a.charAt(i) == '1') sum++;
        }
        return sum;
    }


    public int countBattleships(char[][] board) {
        Queue<int[]> queue = new LinkedList<>();
        int[][] directions = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
        int count = 0;
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                if (board[i][j] == 'X') {
                    queue.offer(new int[]{i, j});
                    count++;
                    while (!queue.isEmpty()) {
                        int[] cur = queue.poll();

                        int x = cur[0], y = cur[1];
                        if (x < 0 || x >= board.length || y < 0 || y >= board[0].length || board[x][y] == '.') continue;
                        board[x][y] = '.';
                        for (int k = 0; k < 4; k++) {
                            int x1 = x + directions[k][0], y1 = y + directions[k][1];
                            queue.offer(new int[]{x1, y1});
                        }
                    }
                }
            }
        }
        return count;
    }

    int factor35(int l, int r) {
        int fivemax = (int) ((int) Math.log(r) * Math.log(5));
        int threemax = (int) ((int) Math.log(r) * Math.log(3));
        long max = (long) (Math.pow(5, fivemax + 1) * Math.pow(3, threemax + 1));
        int count = 0;
        for (int i = l; i <= r; i++) {
            if (max % i == 0) {
                count++;
            }
        }
        return count;
    }

    public int lastStoneWeight(int[] stones) {
        PriorityQueue<Integer> pq = new PriorityQueue<>((a, b) -> b - a);
        for (int each : stones) {
            pq.offer(each);
        }
        while (!pq.isEmpty() && pq.size() > 1) {
            int one = pq.poll(), two = pq.poll();
            if (one != two) {
                pq.offer(Math.abs(one - two));
            }
        }
        return pq.isEmpty() ? 0 : pq.peek();
    }

    int removeOneDigit(String s, String t) {
        StringBuilder s1 = new StringBuilder(s);
        StringBuilder s2 = new StringBuilder(t);
        int res = 0;
        for (int i = 0; i < s.length(); i++) {
            if (Character.isDigit(s1.charAt(i))) {
                char tmp = s1.charAt(i);
                StringBuilder s11 = s1.deleteCharAt(i);
                if (s11.toString().compareTo(t) < 0) {
                    res++;
                }
                s1.insert(i, tmp);
            }
        }
        for (int i = 0; i < t.length(); i++) {
            if (Character.isDigit(s2.charAt(i))) {
                char tmp = s2.charAt(i);
                StringBuilder s11 = s2.deleteCharAt(i);
                if (s.compareTo(s11.toString()) < 0) {
                    res++;
                }
                s2.insert(i, tmp);
            }
        }
        return res;
    }

    //    int[][] meanAndChessboard(int[][] matrix, int[][] queries) {
//        PriorityQueue<int[]> black = new PriorityQueue<>((a,b)->{
//            if(matrix[a[0]][a[1]]!=matrix[b[0]][b[1]]){
//                return matrix[a[0]][a[1]]-matrix[b[0]][b[1]];
//            }else if(a[0]!=b[0]){
//                return a[0]-b[0];
//            }else{
//                return a[1]-b[1];
//            }
//        });
//        PriorityQueue<int[]> white = new PriorityQueue<>((a,b)->{
//            if(matrix[a[0]][a[1]]!=matrix[b[0]][b[1]]){
//                return matrix[a[0]][a[1]]-matrix[b[0]][b[1]];
//            }else if(a[0]!=b[0]){
//                return a[0]-b[0];
//            }else{
//                return a[1]-b[1];
//            }
//        });
//        int row = matrix.length,col = matrix[0].length;
//        for(int i=0;i<row;i+=2){
//            for(int j =0;j<col;j+=2){
//                white.offer(new int[]{i,j});
//            }
//            if(i+1==row)break;
//            for(int j =1;j<col;j+=2){
//                white.offer(new int[]{i+1,j});
//            }
//        }
//        //black
//        for(int i=0;i<row;i+=2){
//
//            for(int j =1;j<col;j+=2){
//                black.offer(new int[]{i,j});
//            }
//            if(i+1==row)break;
//            for(int j =0;j<col;j+=2){
//                black.offer(new int[]{i+1,j});
//            }
//        }
//        for(int[]each:queries){
//            Queue<int[]> tmp = new LinkedList<>();
//            Queue<int[]> tmp2 = new LinkedList<>();
//            int bid = each[0];
//            while(bid-->0){
//                tmp.offer(black.poll());
//            }
//            int[] b =black.poll();
//            tmp.offer(b);
//            int wid = each[1];
//            while(wid-->0){
//                tmp2.offer(white.poll());
//            }
//            int[] w =white.poll();
//            tmp2.offer(w);
//            int ave = (matrix[b[0]][b[1]]+matrix[w[0]][w[1]]);
//            if(ave%2==0){
//                matrix[b[0]][b[1]]=ave/2;
//                matrix[w[0]][w[1]]=ave/2;
//            }else{
//                if()
//                matrix[b[0]][b[1]]=ave/2;
//                matrix[w[0]][w[1]]=ave/2;
//            }
//
//        }
//        return matrix;
//    }
    int findPairsSummingToK(int[] a, int m, int k) {
        int res = 0;
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < a.length; i++) {
            //delte
            if (i > m - 1) {
                if (map.get(a[i - m]) == 1) {
                    map.remove(a[i - m]);
                } else {
                    map.put(a[i - m], map.get(a[i - m]) - 1);
                }
            }
            if (map.containsKey(k - a[i])) res++;
            map.put(a[i], map.getOrDefault(a[i], 0) + 1);
        }
        return res;
    }

    static int[] prefixSum(String[] inputs) {
        int len = inputs.length;
        int[] res = new int[len];
        for (int i = 0; i < len; i++) {
            String cur = inputs[i];
            int sum = 0;
            for (int j = 0; j < cur.length(); ) {

                String cmp = cur.substring(j);
                int id1 = 0, id2 = 0;
                while (id1 < cur.length() && id2 < cmp.length()) {
                    if (cur.charAt(id1) == cmp.charAt(id2)) {
                        id1++;
                        id2++;
                        sum++;
                    } else {
                        break;
                    }
                }
                int tmp = cur.indexOf(cur.charAt(0), j + 1);
                if (tmp < 0) {
                    break;
                } else {
                    j = tmp;
                }
            }
            res[i] = sum;
        }
        return res;
    }

    //    static int[] prefixSum(String[] inputs) {
//        int len = inputs.length;
//        int[] res = new int[len];
//        Map<String,Integer> map = new HashMap<>();
//        for (int i = 0; i < len; i++) {
//            String cur = inputs[i];
//            if(map.containsKey(cur)){
//                res[i] = map.get(cur);
//                break;
//            }
//            int sum = 0;
//            for (int j = 0; j < cur.length(); ) {
//                // String cmp = cur.substring(j);
//                int id1 = 0, id2 = j;
//                while (id1 < cur.length() && id2 < cur.length()) {
//                    if (cur.charAt(id1) == cur.charAt(id2)) {
//                        id1++;
//                        id2++;
//                        sum++;
//                    } else {
//                        break;
//                    }
//                }
//                int tmp = cur.indexOf(cur.charAt(0), j + 1);
//                if (tmp < 0) {
//                    break;
//                } else {
//                    j = tmp;
//                }
//            }
//            res[i] = sum;
//            map.put(cur,sum);
//        }
//        return res;
//    }
    static String isPangram(String[] strings) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < strings.length; i++) {
            String cur = strings[i];
            boolean find = false;
            for (char j = 'a'; j <= 'z'; j++) {
                if (cur.indexOf(j) == -1) {
                    sb.append('0');
                    find = true;
                    break;
                }
            }
            if (!find)
                sb.append('1');
        }
        return sb.toString();
    }

    public String longestPalindrome(String s) {
        int len = s.length();
        boolean[][] dp = new boolean[len][len];
        String res = s.substring(0,1);
        for (int i = len - 1; i >= 0; i--) {
            for (int j = i; j < len; j++) {
                if (i == j) {
                    dp[i][j] = true;
                } else {
                    char ci = s.charAt(i), cj = s.charAt(j);
                    if (ci == cj) {
                        if(i==j-1){
                            dp[i][j]=true;
                        }else {
                            dp[i][j] = dp[i + 1][j - 1];
                        }
                        if(dp[i][j])
                            res=j-i+1>res.length()?s.substring(i,j+1):res;
                    }
                }
            }
        }
        return res;
    }

    int res =0;
    Map<Integer,Integer> num = new HashMap<>();
    public int numDecodings(String s) {
        int res =dfs(s,0);
        return res;
    }
//    void dfs(String s,int pos){
//        if(pos==s.length()){
//            res++;
//            return;
//        }
//        if(s.charAt(pos)!='0'){
//            dfs(s,pos+1);
//        }
//        if(pos<s.length()-1&&Integer.parseInt(s.substring(pos,pos+2))>=20&&Integer.parseInt(s.substring(pos,pos+2))<=26){
//            dfs(s,pos+2);
//        }
//    }
    int dfs(String s,int pos){
        if(pos==s.length()){
            return 1;
        }
        if(num.containsKey(pos))return num.get(pos);
        int res = 0;
        if(s.charAt(pos)!='0'){
             res+=dfs(s,pos+1);
        }
        if(pos<s.length()-1&&Integer.parseInt(s.substring(pos,pos+2))>=10&&Integer.parseInt(s.substring(pos,pos+2))<=26){
            res+=dfs(s,pos+2);
        }
        num.put(pos,res);
        return res;

    }


    public static void main(String[] args) {
        Lc20 lc20 = new Lc20();
//        List<List<Integer>> s1 = Arrays.asList(Arrays.asList(3,4),Arrays.asList(4,5), Arrays.asList(5));
//        int r1=lc20.numberWays(s1);
//        System.out.println(r1);

        int[] s2 = {3, 4, 6, 8};
//        int r2= lc20.maxScore(s2);
//        System.out.println(r2);
//        int r3 = lc20.numberOfUniqueGoodSubsequences("111001101100000001001110110101110001100");
//        int r3 = lc20.numberOfUniqueGoodSubsequences("0101011101111000101111110101111100010100000011001111110110111100000000100100100100111110110111001111011111100011110111100010010011101101001100000010100110011110001");
//        System.out.println(r3);
        int[] s4 = {100};
//        int r4 = lc20.findTargetSumWays(s4,200);
//        System.out.println(r4);

        int[][] s5 = {{-52, 31}, {-73, -26}, {82, 97}, {-65, -11}, {-62, -49}, {95, 99}, {58, 95}, {-31, 49}, {66, 98}, {-63, 2}, {30, 47}, {-40, -26}};
//        lc20.eraseOverlapIntervals(s5);

        int[][] s6 = {{-2147483646, -2147483645}, {2147483646, 2147483647}};
//        lc20.findMinArrowShots(s6);
//        System.out.println(Integer.MAX_VALUE);
//        System.out.println(Integer.MIN_VALUE);
//        String version1 = "1.1";
//        String[] s1=version1.split("\\.");
//        System.out.println(Arrays.toString(s1));
//        System.out.println(Integer.valueOf("001"));

        int[] s7 = {3, 6, 7, 11};
//        lc20.minEatingSpeed(s7,8);
//        lc20.findDifferentBinaryString()

//        lc20.numWays("10101");

//        System.out.println(Math.pow(2,4));
//        int r21 = lc20.factor35(200, 405);
//        System.out.println(r21);

//        System.out.println("a1".compareTo("a2c"));
//        System.out.println("a2d".compareTo("a2ce"));
//        StringBuilder sb =new StringBuilder();
//        sb.insert(0,'1');
//        sb.charAt(0);

//        int r22=lc20.removeOneDigit("ab12c","1zz456");
//        System.out.println(r22);
//        boolean r23= "s1f1".contains("sf");
//        System.out.println(r23);

        String[] s24 = {"ababaa"};
//        prefixSum(s24);
//        lc20.longestPalindrome("babad");
        lc20.longestPalindrome("aacabdkacaa");

    }
}

class AlmostTetris {
    public static void main(String[] args) {
        int n = 4;
        int m = 4;
        int[] figures = {4, 2, 1, 3};
        AlmostTetris almostTetris = new AlmostTetris();
        almostTetris.almostTetris(n, m, figures);
    }

    int[][][] figureDimension = {{{0, 0}}, {{0, 0}, {0, 1}, {0, 2}},
            {{0, 0}, {0, 1}, {1, 0}, {1, 1}},
            {{0, 0}, {1, 0}, {2, 0}, {1, 1}}, {{0, 1}, {1, 0}, {1, 1}, {1, 2}}};

    public int[][] almostTetris(int n, int m, int[] figures) {
        int[][] matrix = new int[n][m];
        int code = 1;
        for (int figure : figures) {
            boolean figurePlaced = false;
            for (int i = 0; i < matrix.length; i++) {
                for (int j = 0; j < matrix[i].length; j++) {
                    if (isPossibleAtThisPoint(matrix, figureDimension[figure - 1], i, j, code)) {
                        figurePlaced = true;
                        code++;
                        break;
                    }
                }
                if (figurePlaced) {
                    break;
                }
            }
        }
        return matrix;
    }

    private boolean isPossibleAtThisPoint(int[][] matrix, int[][] fd, int x, int y, int code) {
        for (int i = 0; i < fd.length; i++) {
            int next_x = fd[i][0] + x;
            int next_y = fd[i][1] + y;
            if (next_x >= 0 && next_x < matrix.length && next_y >= 0 && next_y < matrix[0].length) {
                if (matrix[next_x][next_y] != 0) {
                    return false;
                }
            } else {
                return false;
            }
        }
        for (int i = 0; i < fd.length; i++) {
            int next_x = fd[i][0] + x;
            int next_y = fd[i][1] + y;
            matrix[next_x][next_y] = code;
        }
        return true;
    }
}
//class Solution {
//    public List<List<String>> groupAnagrams(String[] strs) {
//        Map<String, List<String>> map = new HashMap<String, List<String>>();
//        for (String str : strs) {
//            char[] array = str.toCharArray();
//            Arrays.sort(array);
//            String key = new String(array);
//            List<String> list = map.getOrDefault(key, new ArrayList<String>());
//            list.add(str);
//            map.put(key, list);
//        }
//        return new ArrayList<List<String>>(map.values());
//    }
//}


//class Solution {
//    public int minSessions(int[] tasks, int sessionTime) {
//        int n = tasks.length, m = 1 << n;
//        final int INF = 20;
//        int[] dp = new int[m];
//        Arrays.fill(dp, INF);
//
//        // 预处理每个状态，合法状态预设为 1
//        for (int i = 1; i < m; i++) {
//            int state = i, idx = 0;
//            int spend = 0;
//            while (state > 0) {
//                int bit = state & 1;
//                if (bit == 1) {
//                    spend += tasks[idx];
//                }
//                state >>= 1;
//                idx++;
//            }
//            if (spend <= sessionTime) {
//                dp[i] = 1;
//            }
//        }
//
//        // 对每个状态枚举子集，跳过已经有最优解的状态
//        for (int i = 1; i < m; i++) {
//            if (dp[i] == 1) {
//                continue;
//            }
//            for (int j = i; j > 0; j = (j - 1) & i) {
//                // i 状态的最优解可能由当前子集 j 与子集 j 的补集得来
//                dp[i] = Math.min(dp[i], dp[j] + dp[i ^ j]);
//            }
//        }
//
//        return dp[m - 1];
//    }
//}
class DFS {

    static int mod = (int) Math.pow(10, 9) + 7;

    public static int func(List<Integer> hats, int i, boolean[] visited, HashMap<Integer, HashSet<Integer>> map, int[][] dp) {
        int mask = 0;
        for (int j = 0; j < visited.length; j++) mask += visited[j] ? (int) Math.pow(2, j) : 0;
        if (mask == Math.pow(2, visited.length) - 1) return 1;
        if (i == hats.size()) return 0;
        if (dp[i][mask] != -1) return dp[i][mask];
        long ans = 0;
        for (int j = 0; j < visited.length; j++) {
            if (visited[j]) continue;
            HashSet<Integer> set = map.get(j);
            if (set.contains(hats.get(i))) {
                visited[j] = true;
                ans = (ans + func(hats, i + 1, visited, map, dp)) % mod;
                visited[j] = false;
            }
        }
        ans = (ans + func(hats, i + 1, visited, map, dp)) % mod;
        return dp[i][mask] = (int) ans;
    }

    public int numberWays(List<List<Integer>> hats) {
        HashSet<Integer> set = new HashSet<>();
        HashMap<Integer, HashSet<Integer>> map = new HashMap<>();
        for (int i = 0; i < hats.size(); i++) {
            map.put(i, new HashSet<>());
            for (int j = 0; j < hats.get(i).size(); j++) {
                set.add(hats.get(i).get(j));
                map.get(i).add(hats.get(i).get(j));
            }
        }
        List<Integer> h = new ArrayList<>(set);
        Collections.sort(h);
        int[][] dp = new int[h.size() + 1][(int) Math.pow(2, hats.size())];
        for (int i = 0; i <= h.size(); i++) {
            for (int j = 0; j < (int) Math.pow(2, hats.size()); j++) dp[i][j] = -1;
        }
        return func(h, 0, new boolean[hats.size()], map, dp);
    }
}
