//import java.lang.reflect.Array;

import java.util.*;

public class Lc12 {
    //    public int firstMissingPositive(int[] nums) {
//        Arrays.sort(nums);
//        int left =0,right =nums.length-1;
//        while(left<right){
//            int mid =(right-left)/2 +left;
//            if(nums[mid]>0){
//                right=mid;
//            }else{
//                left=mid+1;
//            }
//        }
//        if(nums[left]>1||nums[left]<0)
//            return 1;
//        else{
//            for (int i = 1; i <=nums.length&&left< nums.length ; ) {
//                while(left>0&&nums[left]==nums[left-1]){
//                    left++;
//                }
//                if(nums[left]!=i){
//                    return i;
//                }
//                left++;
//                i++;
//            }
//        }
//        return nums[nums.length-1]+1;
//    }
    public List<Integer> findDuplicates(int[] nums) {
        int len = nums.length;
        List<Integer> res = new ArrayList<>();

        for (int i = 0; i < len; i++) {
            if (nums[nums[i] - 1] != nums[i]) {
                swap(nums, nums[i] - 1, i);
                i--;
            }
        }
        for (int i = 1; i <= len; i++) {
            if (nums[i - 1] != i) {
                res.add(i);
            }
        }
        return res;
    }

    public List<Integer> findDisappearedNumbers(int[] nums) {
        int len = nums.length;
        List<Integer> res = new ArrayList<>();

        for (int i = 0; i < len; i++) {
            if (nums[i] >= i && nums[nums[i] - 1] != nums[i]) {
                swap(nums, nums[i] - 1, i);
                i--;
            }
        }
        for (int i = 1; i <= len; i++) {
            if (nums[i - 1] != i) {
                res.add(i);
            }
        }
        return res;
    }

    public int findRepeatNumber(int[] nums) {
        int len = nums.length;
        for (int i = 0; i < len; i++) {
            if (nums[nums[i]] != nums[i]) {
                swap(nums, nums[i], i);
                i--;
            } else if (nums[nums[i]] != i) {
                return nums[i];
            }
        }
        return -1;
    }

    public int firstMissingPositive(int[] nums) {
        int len = nums.length;
        for (int i = 0; i < len; i++) {
            if (nums[i] > 0 && nums[i] <= len && nums[nums[i] - 1] != nums[i]) {
                swap(nums, nums[i] - 1, i);
                i--;
            }
        }
        for (int i = 0; i < len; i++) {
            if (nums[i] != i + 1) {
                return i + 1;
            }
        }
        return len + 1;
    }

    private void swap(int[] nums, int i, int i1) {
        int tmp = nums[i];
        nums[i] = nums[i1];
        nums[i1] = tmp;
    }

    public int maxProfit2(int[] prices) {
        int len = prices.length;
        int[][] dp = new int[len][4];
        dp[0][0] = -prices[0];
        dp[0][2] = -prices[0];
        for (int i = 1; i < len; i++) {
            dp[i][0] = Math.max(dp[i - 1][0], -prices[i]);
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] + prices[i]);
            dp[i][2] = Math.max(dp[i - 1][2], dp[i - 1][1] - prices[i]);
            dp[i][3] = Math.max(dp[i - 1][3], dp[i - 1][2] + prices[i]);
        }
        return dp[len - 1][3];
    }

    public int maxProfit3(int[] prices) {
        int len = prices.length;
        int[][][] dp = new int[len][2][3];
        dp[0][0][1] = -prices[0];
        dp[0][0][1] = -prices[0];
        for (int i = 1; i < len; i++) {
            for (int j = 1; j <= 2; j++) {
                dp[i][0][j] = Math.max(dp[i - 1][0][j], dp[i - 1][0][j - 1] - prices[i]);
                dp[i][1][j] = Math.max(dp[i - 1][1][j], dp[i - 1][0][j] + prices[i]);
            }

//            dp[i][0][2] = Math.max(dp[i - 1][0][2], dp[i - 1][1][1] - prices[i]);
//            dp[i][1][2] = Math.max(dp[i - 1][1][2], dp[i - 1][0][2] + prices[i]);
        }
        return dp[len - 1][1][2];
    }

    public int maxProfit4(int[] prices) {
        int len = prices.length;
        int[][][] dp = new int[len][2][3];
        dp[0][0][1] = -prices[0];
        dp[0][0][2] = -prices[0];
        for (int i = 1; i < len; i++) {
            //         dp[i][0][1] = Math.max(dp[i - 1][0][1], dp[i - 1][0][0] - prices[i]);
            //         dp[i][1][1] = Math.max(dp[i - 1][1][1], dp[i - 1][0][1] + prices[i]);
            //         dp[i][0][2] = Math.max(dp[i - 1][0][2], dp[i - 1][1][1] - prices[i]);
            //         dp[i][1][2] = Math.max(dp[i - 1][1][2], dp[i - 1][0][2] + prices[i]);
            for (int j = 1; j <= 2; j++) {
                dp[i][0][j] = Math.max(dp[i - 1][0][j], dp[i - 1][1][j - 1] - prices[i]);
                dp[i][1][j] = Math.max(dp[i - 1][1][j], dp[i - 1][0][j] + prices[i]);
            }

//            dp[i][0][2] = Math.max(dp[i - 1][0][2], dp[i - 1][1][1] - prices[i]);
//            dp[i][1][2] = Math.max(dp[i - 1][1][2], dp[i - 1][0][2] + prices[i]);
        }
        return dp[len - 1][1][2];
    }

    //    dp[i][j][k]
//    i means the prices[i]
//    when j= 0, means buying,  j =1, means selling
//    when j = 0, k means buying the kth stock
//    when j = 1, k means selling the kth stock
    public int maxProfit(int k, int[] prices) {
        int len = prices.length;
        if (len == 0 || k == 0) return 0;
        int[][][] dp = new int[len][2][k + 1];
        for (int i = 1; i <= k; i++) {
            dp[0][0][i] = -prices[0];
        }
        for (int i = 1; i < len; i++) {
            for (int j = 1; j <= k; j++) {
                dp[i][0][j] = Math.max(dp[i - 1][0][j], dp[i - 1][1][j - 1] - prices[i]);
                dp[i][1][j] = Math.max(dp[i - 1][1][j], dp[i - 1][0][j] + prices[i]);
            }
        }
        return dp[len - 1][1][k];
    }

    public int[] kWeakestRows(int[][] mat, int k) {
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> {
            if (a[0] != b[0]) return a[0] - b[0];
            else return a[1] - b[1];
        });
        for (int i = 0; i < mat.length; i++) {
            int sum = 0;
            for (int j = 0; j < mat[0].length; j++) {
                if (mat[i][j] == 1) sum++;
            }
            pq.offer(new int[]{sum, i});
        }
        int[] res = new int[k];
        for (int i = 0; i < k; i++) {
            res[i] = pq.poll()[1];
        }
        return res;
    }

    public int droppedRequest(List<Integer> requestTime) {
        int ans = 0;
        for (int i = 0; i < requestTime.size(); i++) {
            if (i > 2 && requestTime.get(i) == requestTime.get(i - 3)) {
                ans++;
            } else if (i > 19 && (requestTime.get(i) - requestTime.get(i - 20)) < 10) {
                ans++;
            } else if (i > 59 && (requestTime.get(i) - requestTime.get(i - 60)) < 60) {
                ans++;
            }
        }
        return ans;
    }

    public int orangesRotting(int[][] grid) {
        Queue<int[]> queue = new LinkedList<>();
        int row = grid.length, col = grid[0].length;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (grid[i][j] == 2) {
                    queue.offer(new int[]{i, j});
                    grid[i][j] = 1;
                }
            }
        }
        int depth = -1;
        while (!queue.isEmpty()) {
            int size = queue.size();
            boolean valid = false;
            for (int i = 0; i < size; i++) {
                int[] cur = queue.poll();
                if (cur[0] < 0 || cur[0] >= row || cur[1] < 0 || cur[1] >= col || grid[cur[0]][cur[1]] != 1) {
                    continue;
                }
                grid[cur[0]][cur[1]] = 2;
                valid = true;
                queue.offer(new int[]{cur[0] + 1, cur[1]});
                queue.offer(new int[]{cur[0] - 1, cur[1]});
                queue.offer(new int[]{cur[0], cur[1] + 1});
                queue.offer(new int[]{cur[0], cur[1] - 1});
            }
            if (valid) depth++;
        }
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (grid[i][j] == 1) {
                    return -1;
                }
            }
        }
        return depth;
    }

    public boolean isThree(int n) {
        int count = 0;
        for (int i = 1; i <= n; i++) {
            if (n % i == 0) {
                count++;
            }
            if (count > 3) return false;
        }
        return count == 3;
    }

    //    public long numberOfWeeks(int[] milestones) {
//        long res = 0;
//        int len = milestones.length;
//        if (len == 0) return res;
//        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> {
//            if (a[0] != b[0]) return b[0] - a[0];
//            else return a[1] - b[1];
//        });
//
//        for (int i = 0; i < len; i++) {
//            pq.offer(new int[]{milestones[i], i});
//        }
//        int pre = -1;
//        while (!pq.isEmpty()) {
//            int[] cur, next;
//            cur = pq.poll();
//            res++;
//            if (pre == -1 || cur[1] != pre) {
//                if (cur[0] != 1) {
//                    pq.offer(new int[]{cur[0] - 1, cur[1]});
//                }
//                pre = cur[1];
//            } else {
//                if (!pq.isEmpty()) {
//                    next = pq.poll();
//                } else {
//                    res--;
//                    return res;
//                }
//                pre = next[1];
//                if (next[0] != 1) {
//                    pq.offer(new int[]{next[0] - 1, next[1]});
//                }
//                pq.offer(new int[]{cur[0], cur[1]});
//            }
//        }
//        return res;
//    }
    public long numberOfWeeks(int[] milestones) {
        long res = 0;
        int len = milestones.length;
        if (len == 0) return res;
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> {
            if (a[0] != b[0]) return b[0] - a[0];
            else return a[1] - b[1];
        });

        for (int i = 0; i < len; i++) {
            pq.offer(new int[]{milestones[i], i});
        }
        int pre = -1;
        while (!pq.isEmpty()) {
            if (pq.size() > 1) {
                int[] max = pq.poll();
                int[] nex = pq.poll();
                if (max[0] > nex[0]) {
                    if (pre != max[1]) {
                        res += (nex[0] * 2 + 1);
                    } else {
                        res += (nex[0] * 2);
                    }
                    if (max[0] != nex[0] + 1) {
                        pq.offer(new int[]{max[0] - nex[0] - 1, max[1]});
                        pre = max[1];
                    } else {
                        pre = -1;
                    }
                } else if (max[0] == nex[0]) {
                    res += (nex[0] * 2);
                    pre = -1;
                } else {
                    if (pre != nex[1]) {
                        res += (max[0] * 2 + 1);
                    } else {
                        res += (max[0] * 2);
                    }
                    if (nex[0] != max[0] + 1) {
                        pq.offer(new int[]{nex[0] - max[0] - 1, nex[1]});
                        pre = nex[1];
                    } else {
                        pre = -1;
                    }
                }

            } else {
                int[] cur = pq.poll();
                if (pre != cur[1]) {
                    res++;
                    return res;
                }
            }
        }
        return res;
    }

    public long minimumPerimeter(long neededApples) {
        long width = 2;
        long pre = 12;
        long all = 12;
        while (all < neededApples) {
            width += 2;
            pre = 4 * width + (2 * width - 1) * 4 + pre;
            System.out.println(pre);

            all += pre;
        }
        return width * 4;
    }

    public int countSpecialSubsequences(int[] nums) {
//        long MOD = 1000000007;
        long MOD = (long) (1e9 + 7);
        int row = nums.length;
        long[][] dp = new long[row + 1][4];
        dp[0][0] = 1;
        for (int i = 1; i <= row; i++) {
            dp[i] = dp[i - 1].clone();
            if (nums[i - 1] == 0) {
                dp[i][1] = (int) ((int) (dp[i - 1][1] * 2 + dp[i - 1][0]) % MOD);
            }
            if (nums[i - 1] == 1) {
                dp[i][2] = (int) ((dp[i - 1][2] * 2 + dp[i - 1][1]) % MOD);
            }
            if (nums[i - 1] == 2) {
                dp[i][3] = (int) ((dp[i - 1][3] * 2 + dp[i - 1][2]) % MOD);
            }
        }
        return (int) dp[row][3];
    }

//    public int countSpecialSubsequences(int[] nums) {
//        int row = nums.length;
//        int[][] dp = new int[4][row+1];
////        for (int i = 0; i <= row; i++) {
////            dp[i][0] = 1;
////        }
//        dp[0][0]=1;
//        for (int i = 1; i <= row; i++) {
//            for (int j = 0; j < 4; j++) {
//                dp[j][i]=dp[j][i-1];
//            }
//            dp[i] = dp[i - 1].clone();
//            if (nums[i - 1] == 0) {
//                dp[i][1] = dp[i - 1][1] > 0 ? dp[i - 1][1] + 1 + dp[i - 1][0] : dp[i - 1][0];
//            }
//            if (nums[i - 1] == 1) {
//                dp[i][2] = dp[i - 1][2] > 0 ? dp[i - 1][2] + 1 + dp[i - 1][1] : dp[i - 1][1];
//            }
//            if (nums[i - 1] == 2) {
//                dp[i][2] = dp[i - 1][2] > 0 ? dp[i - 1][2] + 1 + dp[i - 1][2] : dp[i - 1][2];
//            }
//        }
//        return dp[row][3];
//    }


    public int maxProfit5(int k, int[] prices) {
        int len = prices.length;
        if (len == 0 || k == 0) return 0;
        k = Math.min(k, len / 2);
        int[][][] dp = new int[len][2][k + 1];
        //dp i j k ,j = 0 for not having stock, j = 1 for owning stock
        for (int i = 1; i <= k; i++) {
            dp[0][1][i] = -prices[0];
        }
        for (int i = 1; i < len; i++) {
            for (int j = 1; j <= k; j++) {
                dp[i][0][j] = Math.max(dp[i - 1][0][j], dp[i - 1][1][j] + prices[i]);
                dp[i][1][j] = Math.max(dp[i - 1][1][j], dp[i - 1][0][j - 1] - prices[i]);
            }
        }
        return dp[len - 1][1][k];
    }

    //    public int maxProfit(int[] prices) {
//        int len = prices.length;
//        int[][] dp = new int[len][2];
//        dp[0][1]=-prices[0];
//        for (int i = 1; i <len ; i++) {
//            dp[i][0]= Math.max(dp[i-1][0],dp[i-1][1]+prices[i]);
//            dp[i][1]= Math.max(dp[i-1][1],-prices[i]);
//        }
//        return dp[len-1][0];
//    }
    public int maxProfit(int[] prices) {
        int len = prices.length;
        int[][] dp = new int[len][3];
        dp[0][2] = -prices[0];
        for (int i = 1; i < len; i++) {
            dp[i][0] = dp[i - 1][2] + prices[i];
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0]);
            dp[i][2] = Math.max(dp[i - 1][2], dp[i - 1][1] - prices[i]);
        }
        return Math.max(dp[len - 1][0], dp[len - 1][1]);
    }

    public int maxProfit(int[] prices, int fee) {
        int len = prices.length;
        int[][] dp = new int[len][2];
        dp[0][1] = -prices[0] - fee;
        for (int i = 1; i < len; i++) {
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] - prices[i] - fee);
        }
        return dp[len - 1][0];
    }

    public int lengthOfLongestSubstringKDistinct(String s, int k) {
        // write your code here
        int left = 0, right = 0, len = s.length(), max = 0;
        HashMap<Character, Integer> map = new HashMap<>();
        while (right < len) {
            map.put(s.charAt(right), map.getOrDefault(s.charAt(right), 0) + 1);
            right++;
            while (map.size() > k && left < len) {
                int n = map.get(s.charAt(left));
                if (n == 1) {
                    map.remove(s.charAt(left));
                } else {
                    map.put(s.charAt(left), n - 1);
                }
                left++;
            }
            max = Math.max(max, right - left);
        }
        return max;
    }

    public int subarraysWithKDistinct(int[] nums, int k) {
        return subarrayWithAtMostK(nums, k) - subarrayWithAtMostK(nums, k - 1);
    }

    private int subarrayWithAtMostK(int[] nums, int k) {
        int res = 0, left = 0, right = 0, len = nums.length;
        int count = 0;
        int[] arr = new int[126];
        while (right < len) {
            if (arr[nums[right]] == 0) {
                count++;
            }
            arr[nums[right]]++;
            right++;
            while (count > k) {
                arr[nums[left]]--;
                if (arr[nums[left]] == 0) {
                    count--;
                }
                left++;
            }
            res += right - left;
        }
        return res;
    }

    //    public int subarraysWithKDistinct(int[] nums, int k) {
//        return subarrayWithAtMostK2(nums, k) - subarrayWithAtMostK2(nums, k - 1);
//    }
//
//    private int subarrayWithAtMostK2(int[] nums, int k) {
//        int res = 0, left = 0, right = 0, len = nums.length;
//        Map<Integer, Integer> map = new HashMap<>();
//        while (right < len) {
//            map.put(nums[right], map.getOrDefault(nums[right], 0) + 1);
//            right++;
//            while (map.size() > k) {
//                if (map.get(nums[left]) == 1) {
//                    map.remove(nums[left]);
//                } else {
//                    map.put(nums[left], map.get(nums[left]) - 1);
//                }
//                left++;
//            }
//            res += right - left;
//        }
//        return res;
//    }
    public String minWindow(String s, String t) {
        if (s == null || s.length() == 0 || t == null || t.length() == 0) {
            return "";
        }
        int[] array = new int[128];
        int left = 0, right = 0, pre = -1, min = Integer.MAX_VALUE, len = s.length(), count = 0;
        for (int i = 0; i < t.length(); i++) {
            if (array[t.charAt(i) - 'A'] == 0) count++;
            array[t.charAt(i) - 'A']++;
        }
        while (right < len) {
            array[s.charAt(right) - 'A']--;
            if (array[s.charAt(right) - 'A'] == 0) {
                count--;
            }
            right++;
            if (count == 0) {
                while (count == 0) {
                    array[s.charAt(left) - 'A']++;
                    if (array[s.charAt(left) - 'A'] > 0) {
                        if (right - left < min) {
                            min = right - left;
                            pre = left;
                        }
                        count++;
                    }
                    left++;
                }
            }
        }
        return pre == -1 ? "" : s.substring(pre, pre + min);
    }

    public int characterReplacement(String s, int k) {
        if (s == null) {
            return 0;
        }
        int[] map = new int[26];
        char[] chars = s.toCharArray();
        int left = 0;
        int right = 0;
        int historyCharMax = 0, res = 0;
        for (right = 0; right < chars.length; right++) {
            int index = chars[right] - 'A';
            map[index]++;
            historyCharMax = Math.max(historyCharMax, map[index]);
            if (right - left + 1 > historyCharMax + k) {
                map[chars[left] - 'A']--;
                left++;
            }
            res = Math.max(res, right - left + 1);
        }
        return res;
    }

    public int networkDelayTime2(int[][] times, int n, int k) {
//        boolean[] visit = new boolean[n];
        int[] minTime = new int[n];
        Arrays.fill(minTime, -1);
        Map<Integer, List<int[]>> map = new HashMap<>();
        for (int[] each : times) {
            if (!map.containsKey(each[0])) {
                map.put(each[0], map.getOrDefault(each[0], new ArrayList<>(Arrays.asList(new int[]{each[1], each[2]}))));
            } else {
                map.get(each[0]).add(new int[]{each[1], each[2]});
            }
        }
        if (!map.containsKey(k)) return -1;
        Queue<int[]> queue = new LinkedList<>();
        queue.offer(new int[]{k, 0});
//        int time = 0;
//        visit[k-1]=true;
        minTime[k - 1] = 0;
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                int[] cur = queue.poll();
                int nowTime = cur[1];
                int source = cur[0];
//                visit[source-1]=true;
//                time = Math.max(time, nowTime);
                if (map.containsKey(source)) {
                    List<int[]> curTarget = map.get(source);
                    for (int j = 0; j < curTarget.size(); j++) {
                        int[] target = curTarget.get(j);
//                        if (visit[target[0] - 1]) continue;
//                        visit[target[0] - 1] = true;
                        if (minTime[target[0] - 1] == -1) {
                            minTime[target[0] - 1] = target[1] + nowTime;
                        } else {
                            minTime[target[0] - 1] = Math.min(minTime[target[0] - 1], target[1] + nowTime);
                        }
                        queue.offer(new int[]{target[0], target[1] + nowTime});
//                        time = Math.max(time, target[1] + nowTime);

                    }

                    map.remove(source);
                }
            }
        }
        int maxTime = 0;
        for (int i = 0; i < n; i++) {
            if (minTime[i] == -1) return -1;
            else maxTime = Math.max(maxTime, minTime[i]);
        }
        return maxTime;
    }

//    int N = 110;
//    int[][] w;
//    int INF = 0x3f3f3f3f;
//    int n, k;
//    public int networkDelayTime(int[][] ts, int _n, int _k) {
//        n = _n; k = _k;
//        // 初始化邻接矩阵
//        w= new int[n+1][n+1];
//        for (int i = 1; i <= n; i++) {
//            for (int j = 1; j <= n; j++) {
//                w[i][j] = w[j][i] = i == j ? 0 : INF;
//            }
//        }
//        // 邻接矩阵存图
//        for (int[] t : ts) {
//            int u = t[0], v = t[1], c = t[2];
//            w[u][v] = c;
//        }
//        // Floyd
//        floyd();
//        // 遍历答案
//        int ans = 0;
//        for (int i = 1; i <= n; i++) {
//            ans = Math.max(ans, w[k][i]);
//        }
//        return ans >= INF / 2 ? -1 : ans;
//    }
//    void floyd() {
//        for (int p = 1; p <= n; p++) {
//            for (int i = 1; i <= n; i++) {
//                for (int j = 1; j <= n; j++) {
//                    w[i][j] = Math.min(w[i][j], w[i][p] + w[p][j]);
//                }
//            }
//        }
//    }


    //dijkstra
//    public int networkDelayTime(int[][] ts, int n, int k) {
//        int[][] weight = new int[n + 1][n + 1];
//        int INF = 0x3f3f3f3f;
//        for (int i = 1; i <= n; i++) {
//            for (int j = 1; j <= n; j++) {
//                weight[i][j] = i != j ? INF : 0;
//            }
//        }
//        for (int[] each : ts) {
//            weight[each[0]][each[1]] = each[2];
//        }
//
//        int max = 0;
//        for (int i = 1; i <= n; i++) {
//            int dis = dijkstra(k, weight,i,n);
//            max = Math.max(max, dis);
//        }
//        return max;
//    }
//
    private int dijkstra(int k, int[][] weight, int target, int len) {
        boolean[] vis = new boolean[len + 1];
        int[] distance = new int[len + 1];
        int INF = 0x3f3f3f3f;
        Arrays.fill(distance, INF);
        distance[k] = 0;
        for (int i = 1; i <= len; i++) {
            int t = -1;
            for (int j = 1; j <= len; j++) {
                if (!vis[j] && (t == -1 || distance[j] < distance[t])) t = j;
            }
            vis[t] = true;
            for (int j = 1; j <= len; j++) {
                distance[j] = Math.min(distance[j], distance[t] + weight[t][j]);
            }
        }
        return distance[target];
    }


//    public int networkDelayTime(int[][] ts, int n, int k) {
//        int[][] weight = new int[n + 1][n + 1];
//        int INF = 0x3f3f3f3f;
//        for (int i = 1; i <= n; i++) {
//            for (int j = 1; j <= n; j++) {
//                weight[i][j] = i != j ? INF : 0;
//            }
//        }
//        for (int[] each : ts) {
//            weight[each[0]][each[1]] = each[2];
//        }
//        floyd(weight,n);
//        int max = 0;
//        for (int i = 1; i <=n ; i++) {
//            max = Math.max(max,weight[k][i]);
//        }
//        return max > INF/2  ? -1 : max;
//    }

    private void floyd(int[][] weight, int n) {
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                for (int k = 1; k <= n; k++) {
                    weight[j][k] = Math.min(weight[j][k], weight[j][i] + weight[i][k]);
                }
            }
        }
    }


//    int N = 510, M = N * 4;
////    int[] he = new int[N], e = new int[M], ne = new int[M];
//    int idx;
//    int[] head = new int[N],end = new int[M],next = new int[M];
////    void add(int a, int b) {
////        e[idx] = b;
////        ne[idx] = he[a];
////        he[a] = idx++;
////    }
//    void add(int a,int b){
//        end[idx] = b;
//        next[idx] = head[a];
//        head[a] = idx++;
//    }
//    boolean[] vis = new boolean[N];
//    public List<Integer> distanceK(TreeNode root, TreeNode t, int k) {
//        List<Integer> ans = new ArrayList<>();
//        Arrays.fill(head, -1);
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
//                for (int i = head[poll]; i != -1 ; i = next[i]) {
//                    int j = end[i];
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


//    int N = 110, M = 6010, n = 0, k = 0, idx = 0;
//    int[] head = new int[N], end = new int[M], next = new int[M], w = new int[N];
//    int INF = 0x3f3f3f3f;
//
//    void add(int a, int b, int weight) {
//        end[idx] = b;
//        next[idx] = head[a];
//        w[idx] = weight;
//        head[a] = idx++;
//    }
//
//    public int networkDelayTime(int[][] ts, int _n, int _k) {
//        n = _n;
//        k = _k;
//        Arrays.fill(head, -1);
//        for (int[] each : ts) {
//            add(each[0], each[1], each[2]);
//        }
//        int time = 0;
//        for (int i = 1; i <= n; i++) {
//            if (i == k) continue;
//            int curRes = dijkstra2(i);
//            time = Math.max(time, curRes);
//        }
//        return time > INF / 2 ? -1 : time;
//    }
//
//    private int dijkstra2(int i) {
//        boolean[] vis = new boolean[N];
//        int[] dis = new int[N];
//        Arrays.fill(dis, INF);
//        dis[k] = 0;
//        PriorityQueue<int[]> queue = new PriorityQueue<>((a, b) -> a[1] - b[1]);
//        queue.offer(new int[]{k, 0});
//        while (!queue.isEmpty()) {
//            int[] cur = queue.poll();
//            int nowNode = cur[0];
//            int nowDis = cur[1];
//            if (vis[nowNode]) continue;
//            vis[nowNode] = true;
//            for (int j = head[nowNode]; j != -1; j = next[j]) {
//                int nextone = end[j];
//                if (dis[nextone] > nowDis + w[j]) {
//                    dis[nextone] = nowDis + w[j];
//                    queue.offer(new int[]{nextone, nowDis + w[j]});
//                }
//            }
//        }
//        return dis[i];
//    }

    int N = 1200, M = 10001, idx = 0;
    int[] head = new int[N], end = new int[M], next = new int[M], count = new int[N];
    int INF = 0x3f3f3f3f;

    void add(int a, int b) {
        end[idx] = b;
        next[idx] = head[a];
        head[a] = idx++;
        count[a]++;
    }

    public int findJudge(int n, int[][] trust) {
        Arrays.fill(head, -1);
        for (int[] each : trust) {
            add(each[1], each[0]);
        }
        for (int i = 1; i <= n; i++) {
            if (count[i] == n - 1) {
                boolean valid = true;
                for (int j = 1; j <= n; j++) {
                    if (j == i) continue;
                    for (int k = head[j]; k != -1; k = next[k]) {
                        int cur = end[k];
                        if (cur == i) {
                            valid = false;
                            break;
                        }
                    }
                }
                if (valid) {
                    return i;
                }
            }
        }
        return -1;
    }

    public int findCenter(int[][] edges) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int[] each : edges) {
            map.put(each[0], map.getOrDefault(each[0], 0) + 1);
            map.put(each[1], map.getOrDefault(each[1], 0) + 1);
        }
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            if (entry.getValue() == map.size()) {
                return entry.getKey();
            }
        }

        return -1;
    }

    public int countRestrictedPaths(int n, int[][] es) {
        int MOD = (int) 1e9 + 7;
        int INF = 0x3f3f3f3f;
        Map<Integer, Map<Integer, Integer>> map = new HashMap<>();
//        for(int[] each:es){
//            map.computeIfAbsent(each[0],k->new HashMap<Integer,Integer>()).put(each[1],each[2]);
//            map.computeIfAbsent(each[1],k->new HashMap<Integer,Integer>()).put(each[0],each[2]);
//        }
        for (int[] e : es) {
            int a = e[0], b = e[1], w = e[2];
            Map<Integer, Integer> am = map.getOrDefault(a, new HashMap<Integer, Integer>());
            am.put(b, w);
            map.put(a, am);
            Map<Integer, Integer> bm = map.getOrDefault(b, new HashMap<Integer, Integer>());
            bm.put(a, w);
            map.put(b, bm);
        }
        int[] dis = new int[n + 1];
        Arrays.fill(dis, INF);
        boolean[] vis = new boolean[n + 1];
        dis[n] = 0;
        PriorityQueue<int[]> queue = new PriorityQueue<>((a, b) -> a[1] - b[1]);
        queue.offer(new int[]{n, 0});
        while (!queue.isEmpty()) {
            int[] tmp = queue.poll();
            int idx = tmp[0], step = tmp[1];
            if (vis[idx]) continue;
            vis[idx] = true;
//            Map<Integer,Integer> curMap = map.get(idx);
//            if(curMap==null) continue;
//            for(Map.Entry<Integer,Integer>entry:curMap.entrySet()){
//                if(dis[entry.getKey()]>dis[idx]+entry.getValue()) {
//                    dis[entry.getKey()] = dis[idx]+entry.getValue();
//                    queue.offer(new int[]{entry.getKey(),dis[entry.getKey()]});
//                }
//            }
            Map<Integer, Integer> mm = map.get(idx);
            if (mm == null) continue;
            for (int i : mm.keySet()) {
                dis[i] = Math.min(dis[i], dis[idx] + mm.get(i));
                queue.add(new int[]{i, dis[i]});
            }
        }
        int[][] dp = new int[n][2];
        for (int i = 0; i < n; i++) {
            dp[i] = new int[]{i + 1, dis[i + 1]};
        }
        Arrays.sort(dp, (a, b) -> a[1] - b[1]);
        int[] f = new int[n + 1];
        f[n] = 1;
        for (int i = 0; i < n; i++) {
//            int idx = dp[i][0], step = dp[i][1];
//            Map<Integer,Integer> curMap = map.get(idx);
//            if(curMap==null) continue;
//            for(Map.Entry<Integer,Integer>entry: curMap.entrySet()){
//                if(dis[entry.getKey()]<step){
//                    f[entry.getKey()]+=f[idx];
//                    f[entry.getKey()] %= MOD;
//                }
//            }
            int idx = dp[i][0], cur = dp[i][1];
            Map<Integer, Integer> mm = map.get(idx);
            if (mm == null) continue;
            for (int next : mm.keySet()) {
                if (cur > dis[next]) {
                    f[idx] += f[next];
                    f[idx] %= MOD;
                }
            }
            if (idx == 1) break;
        }
        return f[1];
    }

    //    public int findUnsortedSubarray(int[] nums) {
//        int left =0,len =nums.length,right = 0;
//        for (int i = 1; i < len; i++) {
//            if(nums[i]<nums[i-1]){
//                left = i-1;
//                break;
//            }
//        }
//        for (int i = len-2; i >=0 ; i--) {
//            if(nums[i]>nums[i+1]){
//                right = i+1;
//                break;
//            }
//        }
//        if(left==right) return 0;
//        int min = Integer.MAX_VALUE,max = Integer.MIN_VALUE;
//        for (int i = left; i <=right ; i++) {
//            min = Math.min(min,nums[i]);
//            max= Math.max(max,nums[i]);
//        }
//        while(left-1>=0&&nums[left-1]>min){
//            left--;
//        }
//        while(right+1<len&&nums[right+1]<max){
//            right++;
//        }
//        return right-left+1;
//    }

    public int findUnsortedSubarray(int[] nums) {
        Stack<Integer> s1 = new Stack<>();
        Stack<Integer> s2 = new Stack<>();
        int len = nums.length,left = len,right =0,idx=0;
        while(idx<len){
            while(!s1.isEmpty()&&nums[idx]<nums[s1.peek()]){
                left = Math.min(left,s1.pop());
            }
            s1.push(idx++);
        }
        idx=len-1;
        while(idx>=0){
            while(!s2.isEmpty()&&nums[idx]>nums[s2.peek()]){
                right = Math.max(right,s2.pop());
            }
            s2.push(idx--);
        }
        if(left==len&&right==0) return 0;
        return right-left+1;
    }

    public int kthFactor(int n, int k) {
        List<Integer> list = new ArrayList<>();
        for (int i = 1; i <=n ; i++) {
            if(n%i==0){
                list.add(i);
            }
        }
        if(list.size()<k){
            return -1;
        }else{
            return list.get(k-1);
        }
    }
    public static void main(String[] args) {
        System.out.println();
        Lc12 lc12 = new Lc12();
        int[] s1 = {1, 1000};
//        lc12.firstMissingPositive(s1);
        int[] s2 = {4, 3, 2, 7, 8, 2, 3, 1};
//        lc12.findDisappearedNumbers(s2);
        int[] s3 = {1, 2, 3, 4, 5};
        int[] s4 = {3, 2, 6, 5, 0, 3};
//        int r1 = lc12.maxProfit(2, s4);
//        System.out.println(r1);
//        System.out.println(lc12.droppedRequest(Arrays.asList(1, 1, 1, 1, 2)));//1
//        System.out.println(lc12.droppedRequest(Arrays.asList(1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 7, 11, 11, 11, 11)));//7
//        System.out.println(lc12.droppedRequest(Arrays.asList(1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7)));//2

        int[] s5 = {10, 5, 1, 2};
//        lc12.numberOfWeeks(s5);
//        lc12.minimumPerimeter(13);
        int[] s6 = {0, 1, 2, 2};
        int[] s7 = {2, 0, 0, 2, 0, 1, 2};

//        lc12.countSpecialSubsequences(s7);
//        lc12.minWindow("ADOBECODEBANC","ABC");
//        lc12.minWindow("cabwefgewcwaefgcf","cae");
        int[] s8 = {0, 2, 5, 6, 8, 12, 15};
//        int r1 = Arrays.binarySearch(s8, 9);
//        System.out.println(-r1 - 1);

        int[][] s9 = {{2, 1, 1}, {2, 3, 1}, {3, 4, 1}};
//        lc12.networkDelayTime(s9, 4, 2);
        int[][] s10 = {{1, 3}, {2, 3}, {3, 1}};
//        lc12.findJudge(3, s10);

//        lc12.distanceK(new TreeNode(1, new TreeNode(2), new TreeNode(3)), null, 2);

        int[] s11 = {2, 6, 4, 8, 10, 9, 15};
        int[] s12 = {2, 3, 3, 2, 4};
        lc12.findUnsortedSubarray(s12);

    }
}


//class Solution {
//    int N = 110, M = 6010;
//    int[][] w = new int[N][N];
//    int INF = 0x3f3f3f3f;
//    int n, k;
//
//    public int networkDelayTime(int[][] ts, int _n, int _k) {
//        n = _n;
//        k = _k;
//        // 初始化邻接矩阵
//        for (int i = 1; i <= n; i++) {
//            for (int j = 1; j <= n; j++) {
//                w[i][j] = w[j][i] = i == j ? 0 : INF;
//            }
//        }
//        // 邻接矩阵存图
//        for (int[] t : ts) {
//            int u = t[0], v = t[1], c = t[2];
//            w[u][v] = c;
//        }
//        // Floyd
//        floyd();
//        // 遍历答案
//        int ans = 0;
//        for (int i = 1; i <= n; i++) {
//            ans = Math.max(ans, w[k][i]);
//        }
//        return ans >= INF / 2 ? -1 : ans;
//    }
//
//    void floyd() {
//        for (int p = 1; p <= n; p++) {
//            for (int i = 1; i <= n; i++) {
//                for (int j = 1; j <= n; j++) {
//                    w[i][j] = Math.min(w[i][j], w[i][p] + w[p][j]);
//                }
//            }
//        }
//    }
//}
//
//class Solution {
//    int N = 110, M = 6010;
//    int[][] w = new int[N][N];
//    int[] dist = new int[N];
//    boolean[] vis = new boolean[N];
//    int INF = 0x3f3f3f3f;
//    int n, k;
//
//    public int networkDelayTime(int[][] ts, int _n, int _k) {
//        n = _n;
//        k = _k;
//        // 初始化邻接矩阵
//        for (int i = 1; i <= n; i++) {
//            for (int j = 1; j <= n; j++) {
//                w[i][j] = w[j][i] = i == j ? 0 : INF;
//            }
//        }
//        // 邻接矩阵存图
//        for (int[] t : ts) {
//            int u = t[0], v = t[1], c = t[2];
//            w[u][v] = c;
//        }
//        // 朴素 Dijkstra
//        int ans = 0;
//        for (int end = 1; end <= n; end++) {
//            if (end == k) continue;
//            ans = Math.max(ans, dijkstra(end));
//        }
//        return ans > INF / 2 ? -1 : ans;
//    }
//
//    int dijkstra(int end) {
//        Arrays.fill(vis, false);
//        Arrays.fill(dist, INF);
//        dist[k] = 0;
//        for (int p = 1; p <= n; p++) {
//            int t = -1;
//            for (int i = 1; i <= n; i++) {
//                if (!vis[i] && (t == -1 || dist[i] < dist[t])) t = i;
//            }
//            vis[t] = true;
//            for (int i = 1; i <= n; i++) {
//                dist[i] = Math.min(dist[i], dist[t] + w[t][i]);
//            }
//        }
//        return dist[end];
//    }
//}


//class Solution {
//    int N = 110, M = 6010;
//    int[] he = new int[N], e = new int[M], ne = new int[M], w = new int[M];
//    int[] dist = new int[N];
//    boolean[] vis = new boolean[N];
//    int INF = 0x3f3f3f3f;
//    int n, k, idx;
//    void add(int a, int b, int c) {
//        e[idx] = b;
//        ne[idx] = he[a];
//        he[a] = idx;
//        w[idx] = c;
//        idx++;
//    }
//    public int networkDelayTime(int[][] ts, int _n, int _k) {
//        n = _n; k = _k;
//        // 初始化邻接表（链表头）
//        Arrays.fill(he, -1);
//        // 邻接表存图
//        for (int[] t : ts) {
//            int u = t[0], v = t[1], c = t[2];
//            add(u, v, c);
//        }
//        // 堆优化朴素 Dijkstra
//        int ans = 0;
//        for (int end = 1; end <= n; end++) {
//            if (end == k) continue;
//            ans = Math.max(ans, dijkstra(end));
//        }
//        return ans > INF / 2 ? -1 : ans;
//    }
//    int dijkstra(int end) {
//        Arrays.fill(vis, false);
//        Arrays.fill(dist, INF);
//        dist[k] = 0;
//        PriorityQueue<int[]> q = new PriorityQueue<>((a,b)->a[1]-b[1]); // id dist
//        q.add(new int[]{k, 0});
//        while (!q.isEmpty()) {
//            int[] poll = q.poll();
//            int id = poll[0], step = poll[1];
//            if (vis[id]) continue;
//            vis[id] = true;
//            for (int i = he[id]; i != -1; i = ne[i]) {
//                int j = e[i];
//                if (dist[j] > step + w[i]) {
//                    dist[j] = step + w[i];
//                    q.add(new int[]{j, dist[j]});
//                }
//            }
//        }
//        return dist[end];
//    }
//}
//
//


//class Solution {
//    int N = 110, M = 6010;
//    int[] he = new int[N], e = new int[M], ne = new int[M], w = new int[M];
//    int[] dist = new int[N];
//    boolean[] vis = new boolean[N];
//    int INF = 0x3f3f3f3f;
//    int n, k, idx;
//
//    void add(int a, int b, int c) {
//        e[idx] = b;
//        ne[idx] = he[a];
//        he[a] = idx;
//        w[idx] = c;
//        idx++;
//    }
//
//    public int networkDelayTime(int[][] ts, int _n, int _k) {
//        n = _n;
//        k = _k;
//        // 初始化邻接表（链表头）
//        Arrays.fill(he, -1);
//        // 邻接表存图
//        for (int[] t : ts) {
//            int u = t[0], v = t[1], c = t[2];
//            add(u, v, c);
//        }
//        // 堆优化朴素 Dijkstra
//        int ans = 0;
//        for (int end = 1; end <= n; end++) {
//            if (end == k) continue;
//            ans = Math.max(ans, dijkstra(end));
//        }
//        return ans > INF / 2 ? -1 : ans;
//    }
//
//    // d->end
//    int dijkstra(int end) {
//        Arrays.fill(vis, false);
//        Arrays.fill(dist, INF);
//        dist[k] = 0;
//        PriorityQueue<int[]> q = new PriorityQueue<>((a, b) -> a[1] - b[1]); // id dist
//        q.add(new int[]{k, 0});
//        while (!q.isEmpty()) {
//            int[] poll = q.poll();
//            int id = poll[0], step = poll[1];
//            if (vis[id]) continue;
//            vis[id] = true;
//            for (int i = he[id]; i != -1; i = ne[i]) {
//                int j = e[i];
//                if (dist[j] > step + w[i]) {
//                    dist[j] = step + w[i];
//                    q.add(new int[]{j, dist[j]});
//                }
//            }
//        }
//        return dist[end];
//    }
//}

//
//class Solution {
//    public int countRestrictedPaths(int n, int[][] edges) {
//        int cnt = 0;
//        Map<Integer, List<int[]>> map = new HashMap<>();
//        // 初始化邻接表
//        for (int[] t : edges) {
//            int x = t[0];
//            int y = t[1];
//            map.computeIfAbsent(x, k -> new ArrayList<>()).add(new int[]{y, t[2]});
//            map.computeIfAbsent(y, k -> new ArrayList<>()).add(new int[]{x, t[2]});
//        }
//
//        // 保存到n点的 最短距离 和 受限路径数
//        int[] distance = findShortPath(map, n, n);
//        Long[] mem = new Long[n + 1];
//
//        cnt = (int) findLimitedPathCount(map, 1, n, distance, mem);
//        return cnt;
//    }
//
//    private long findLimitedPathCount(Map<Integer, List<int[]>> map, int i, int n, int[] distance, Long[] mem) {
//        if (mem[i] != null) return mem[i];
//        if (i == n) return 1;
//        long cnt = 0;
//        List<int[]> list = map.getOrDefault(i, Collections.emptyList());
//        for (int[] arr : list) {
//            int next = arr[0];
//            //如果相邻节点距离比当前距离小，说明是受限路径
//            if (distance[next] < distance[i]) {
//                cnt += findLimitedPathCount(map, next, n, distance, mem);
//                cnt %= MOD;
//            }
//        }
//        mem[i] = cnt;
//        return cnt;
//    }
//
//
//    public int[] findShortPath(Map<Integer, List<int[]>> map, int n, int start) {
//        // 初始化distance数组和visit数组，并用最大值填充作为非连接状态INF
//        int[] distance = new int[n + 1];
//        Arrays.fill(distance, Integer.MAX_VALUE);
//        boolean[] visit = new boolean[n + 1];
//
//        // 初始化，索引0和起点的distance为0
//        distance[start] = 0;
//        distance[0] = 0;
//
//        // 堆优化，将距离作为排序标准。单独用传入距离是因为PriorityQueue的上浮规则决定
//        PriorityQueue<int[]> queue = new PriorityQueue<>((o1, o2) -> o1[1] - o2[1]);
//        // 把起点放进去，距离为0
//        queue.offer(new int[]{start, 0});
//
//        while (!queue.isEmpty()) {
//            // 当队列不空，拿出一个源出来
//            Integer poll = queue.poll()[0];
//            if (visit[poll]) continue;
//            // 标记访问
//            visit[poll] = true;
//            // 遍历它的相邻节点
//            List<int[]> list = map.getOrDefault(poll, Collections.emptyList());
//            for (int[] arr : list) {
//                int next = arr[0];
//                if (visit[next]) continue;
//                // 更新到这个相邻节点的最短距离，与 poll出来的节点增加的距离 比较
//                distance[next] = Math.min(distance[next], distance[poll] + arr[1]);
//                //堆中新增节点，这里需要手动传入 next节点堆距离值。否则如果next在队列中，将永远无法上浮。
//                queue.offer(new int[]{next, distance[next]});
//            }
//        }
//        return distance;
//    }
//
//    final int MOD = 1000000007;
//
//}
//
//
//class Solution {
//    int mod = 1000000007;
//
//    public int countRestrictedPaths(int n, int[][] es) {
//        // 预处理所有的边权。 a b w -> a : { b : w } + b : { a : w }
//        Map<Integer, Map<Integer, Integer>> map = new HashMap<>();
//        for (int[] e : es) {
//            int a = e[0], b = e[1], w = e[2];
//            Map<Integer, Integer> am = map.getOrDefault(a, new HashMap<Integer, Integer>());
//            am.put(b, w);
//            map.put(a, am);
//            Map<Integer, Integer> bm = map.getOrDefault(b, new HashMap<Integer, Integer>());
//            bm.put(a, w);
//            map.put(b, bm);
//        }
//
//        // 堆优化 Dijkstra：求 每个点 到 第n个点 的最短路
//        int[] dist = new int[n + 1];
//        boolean[] st = new boolean[n + 1];
//        Arrays.fill(dist, Integer.MAX_VALUE);
//        dist[n] = 0;
//        Queue<int[]> q = new PriorityQueue<int[]>((a, b) -> a[1] - b[1]); // 点编号，点距离。根据点距离从小到大
//        q.add(new int[]{n, 0});
//        while (!q.isEmpty()) {
//            int[] e = q.poll();
//            int idx = e[0], cur = e[1];
//            if (st[idx]) continue;
//            st[idx] = true;
//            Map<Integer, Integer> mm = map.get(idx);
//            if (mm == null) continue;
//            for (int i : mm.keySet()) {
//                dist[i] = Math.min(dist[i], dist[idx] + mm.get(i));
//                q.add(new int[]{i, dist[i]});
//            }
//        }
//
//        // dp 过程
//        int[][] arr = new int[n][2];
//        for (int i = 0; i < n; i++) arr[i] = new int[]{i + 1, dist[i + 1]}; // 点编号，点距离
//        Arrays.sort(arr, (a, b) -> a[1] - b[1]); // 根据点距离从小到大排序
//
//        // 定义 f(i) 为从第 i 个点到结尾的受限路径数量
//        // 从 f[n] 递推到 f[1]
//        int[] f = new int[n + 1];
//        f[n] = 1;
//        for (int i = 0; i < n; i++) {
//            int idx = arr[i][0], cur = arr[i][1];
//            Map<Integer, Integer> mm = map.get(idx);
//            if (mm == null) continue;
//            for (int next : mm.keySet()) {
//                if (cur > dist[next]) {
//                    f[idx] += f[next];
//                    f[idx] %= mod;
//                }
//            }
//            // 第 1 个节点不一定是距离第 n 个节点最远的点，但我们只需要 f[1]，可以直接跳出循环
//            if (idx == 1) break;
//        }
//        return f[1];
//    }
//}


