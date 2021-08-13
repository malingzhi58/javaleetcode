

import java.util.Arrays;
import java.util.*;


public class Lc7 {
    public int findLongestChain(int[][] pairs) {
        Arrays.sort(pairs, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                if (o1[0] != o2[0]) return o1[0] - o2[0];
                else return o1[1] - o2[1];
            }
        });
        int[][] dp = new int[pairs.length][2];
        int index = 0;
        dp[0][0] = pairs[0][0];
        dp[0][1] = pairs[0][1];
        for (int i = 1; i < pairs.length; i++) {
            if (pairs[i][0] > dp[index][1]) {
                index++;
                dp[index][0] = pairs[i][0];
                dp[index][1] = pairs[i][1];
            } else if (pairs[i][1] < dp[index][1]) {
                dp[index][0] = pairs[i][0];
                dp[index][1] = pairs[i][1];
            }
        }
        return index + 1;
    }

    public int wiggleMaxLength(int[] nums) {
        int[] dp = new int[nums.length];
        Arrays.fill(dp, 1);
        int pre = 0, preIndex = 0, cur = 0;
        for (int i = 1; i < nums.length; i++) {
            cur = nums[i] - nums[preIndex];
            if ((pre == 0 && cur != 0) || (cur * pre < 0)) {
                dp[i] = Math.max(dp[i], dp[i - 1] + 1);
            } else {
                dp[i] = dp[i - 1];
            }
            pre = cur;
            preIndex = i;
        }
        return dp[nums.length - 1];
    }

    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        if (nums == null || nums.length == 0) return res;
        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 2; i++) {
//            if ( nums[i] == nums[i + 1]) continue;
//            if (i > 0 && nums[i] == nums[i - 1]) continue;
            int right = nums.length - 1;
            int j = i + 1;
            for (; j < right; ) {
                int curSum = nums[i] + nums[right] + nums[j];
                if (j > i + 1 && nums[j] == nums[j - 1]) {
                    j++;
                    continue;
                }
                if (curSum == 0) {
                    res.add(new ArrayList<>(Arrays.asList(nums[i], nums[j], nums[right])));
                } else if (curSum > 0) {
                    right--;
                } else {
                    j++;
                }
            }
        }
        return res;
    }

    public boolean canPartition3(int[] nums) {
        Arrays.sort(nums);
        int left = 0, right = nums.length - 1, leftSum = nums[left], rightSum = nums[right];
        if (right == 0) return false;
        while (left + 1 < right) {
            if (leftSum == rightSum && right - left == 1) return true;
            if (leftSum < rightSum) {
                left++;
                leftSum += nums[left];
            } else if (leftSum > rightSum) {
                right--;
                rightSum += nums[right];
            } else {
                left++;
                leftSum += nums[left];
            }
        }
        return leftSum == rightSum;
    }

    public boolean canPartition2(int[] nums) {
        int left = 0, right = nums.length - 1, leftSum = 0, rightSum = 0;
        Arrays.sort(nums);
        while (left < right) {
            if (leftSum == rightSum && right - left == 1) return true;

            if (leftSum < rightSum) {
                leftSum += nums[left];
                left++;
            } else if (leftSum > rightSum) {
                rightSum += nums[right];
                right--;
            } else {
                leftSum += nums[left];
                rightSum += nums[right];
                left++;
                right--;
            }
        }
        return leftSum == rightSum;
    }

    public boolean canPartition4(int[] nums) {
        Arrays.sort(nums);
        Map<Integer, Integer> map = new HashMap<>();
        int sum = 0;
        map.put(0, -1);
        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
//            set.add(sum);
            map.put(sum, i);
        }
        if (sum - sum / 2 != sum / 2) {
            return false;
        }
        int target = sum / 2;
        sum = 0;
        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
            if (map.containsKey(sum - target)) {
                return true;
            }
        }
        return false;
    }

    /**
     * 我现在困惑在，这道题没分清是01背包，导致先后用滑动窗口，presum做，但都不是符合题意的做法
     */
    public boolean canPartition(int[] nums) {
        int len = nums.length, sum = 0;
        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
        }
        if (sum - sum / 2 != sum / 2) {
            return false;
        }
        int target = sum / 2;
        boolean[][] dp = new boolean[len][target + 1];
        //避免 [100] 这种case
        if (nums[0] <= target) {
            dp[0][nums[0]] = true;
        }
        for (int i = 1; i < len; i++) {
            for (int j = 0; j <= target; j++) {
                dp[i][j] = dp[i - 1][j];
                if (nums[i] == j) {
                    dp[i][j] = true;
                }
                if (nums[i] <= j) {
                    dp[i][j] = dp[i][j] || dp[i - 1][j - nums[i]];
                }
            }
        }
        return dp[len - 1][target];
    }

    public boolean canPartition5(int[] nums) {
        int len = nums.length, sum = 0;
        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
        }
        if (sum - sum / 2 != sum / 2) {
            return false;
        }
        int target = sum / 2;
        boolean[] dp = new boolean[target + 1];
        if (nums[0] <= target) {
            dp[nums[0]] = true;
        }
        for (int i = 0; i < len; i++) {
            for (int j = target; j >= 0; j--) {
                if (j >= nums[i])
                    dp[j] = dp[j] || dp[j - nums[i]];
            }
        }
        return dp[target];
    }

    public int findTargetSumWays(int[] nums, int target) {
        int sum = 0;
        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
        }
        if (target > sum || target < -sum) return 0;
        int[][] dp = new int[nums.length][sum * 2 + 1];
        dp[0][nums[0] + sum] = 1;
        dp[0][-nums[0] + sum] = 1;
        for (int i = 1; i < nums.length; i++) {
            for (int j = 0; j <= sum * 2; j++) {
                if (j - nums[i] >= 0) {
                    dp[i][j] += dp[i - 1][j - nums[i]];
                }
                if (nums[i] + j <= 2 * sum)
                    dp[i][j] += dp[i - 1][j + nums[i]];

            }
        }
        return dp[nums.length - 1][target + sum];
    }


    public int lengthOfLongestSubstring(String s) {
        if (s.length() == 0) return 0;
        HashMap<Character, Integer> map = new HashMap<Character, Integer>();
        int max = 0;
        int left = 0;
        for (int i = 0; i < s.length(); i++) {
            if (map.containsKey(s.charAt(i))) {
                left = Math.max(left, map.get(s.charAt(i)) + 1);
                //"abba" 一定要max，否则max 会虚高
                // +1也不能放外边，因为这时候避免的情况是map.get 到+1的位置 在 left的后边
                // left=map.get(s.charAt(i))+1;
            }
            map.put(s.charAt(i), i);
            max = Math.max(max, i - left + 1);
        }
        return max;
    }

    public int findTargetSumWays2(int[] nums, int target) {
        int sum = 0;
        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
        }
        if (target > sum || target < -sum) return 0;
        int[][] dp = new int[2][sum * 2 + 1];
        dp[0][sum + nums[0]] += 1;
        dp[0][sum - nums[0]] += 1;
        for (int i = 1; i < nums.length; i++) {
            dp[1] = new int[sum * 2 + 1];
            for (int j = 0; j <= sum * 2; j++) {
                if (j - nums[i] >= 0)
                    dp[1][j] += dp[0][j - nums[i]];
                if (nums[i] + j <= 2 * sum)
                    dp[1][j] += dp[0][j + nums[i]];
            }
            dp[0] = dp[1].clone();
        }
        return dp[1][sum + target];
    }

    public int findMaxForm(String[] strs, int m, int n) {
        int res = 0, count0 = 0, count1 = 0;
        for (int i = 0; i < strs.length; i++) {
            count0 = 0;
            count1 = 0;
            String cur = strs[i];
            boolean exceed = false;
            for (int j = 0; j < cur.length(); j++) {
                if (cur.charAt(j) == '0') count0++;
                if (cur.charAt(j) == '1') count1++;
                if (count0 > m || count1 > n) {
                    exceed = true;
                    break;
                }
            }
            if (!exceed) {
                res++;
            }
        }
        return res;
    }

    // m for 0, n for 1
    public int findMaxForm2(String[] strs, int m, int n) {
        int len = strs.length;
        int[][][] dp = new int[len][m + 1][n + 1];
        int[] str1 = countNum(strs[0]);
        if (str1[0] <= m && str1[1] <= n) {
            dp[0][str1[0]][str1[1]] += 1;
        }
        for (int i = 1; i < len; i++) {
            int[] cur = countNum(strs[i]);
            if (cur[0] > m || cur[1] > n) {
                dp[i] = dp[i - 1].clone();
                continue;
            }
            dp[i][cur[0]][cur[1]] += 1;
            for (int j = m; j >= 0; j--) {
                for (int k = n; k >= 0; k--) {
                    dp[i][j][k] = Math.max(dp[i - 1][j][k], dp[i][j][k]);
                    if (j >= cur[0] && k >= cur[1] && dp[i - 1][j - cur[0]][k - cur[1]] > 0) {
                        dp[i][j][k] = dp[i - 1][j - cur[0]][k - cur[1]] + 1;
                    }
                }
            }
        }
        int max = 0;
        for (int i = 0; i <= m; i++) {
            for (int j = 0; j <= n; j++) {
                max = Math.max(dp[len - 1][i][j], max);
            }
        }
        return max;
    }

    private int[] countNum(String str) {
        int count0 = 0, count1 = 0;
        for (int j = 0; j < str.length(); j++) {
            if (str.charAt(j) == '0') count0++;
            if (str.charAt(j) == '1') count1++;
        }
        return new int[]{count0, count1};
    }

    public int majorityElement(int[] nums) {
        int left = 0, right = 0, len = nums.length, max = 0;
        Arrays.sort(nums);
        while (right < len) {
            if (nums[left] == nums[right]) {
                right++;
            } else {
                left = right;

            }
            max = Math.max(right - left, max);
            if (max > len / 2) {
                return nums[left];
            }
        }
        return -1;
    }

    // 完全背包，写两个，一个是初始化int[coins.length+1]，另一个是int[coins.length] 不太方便！
//    这个dp的意思是dp i j 用第0-i个coin，得到j的组合数，因为是组合，所以在上次的基础上++
    public int change2(int amount, int[] coins) {
        int[][] dp = new int[coins.length + 1][amount + 1];
        dp[0][0] = 1;
        for (int i = 1; i <= coins.length; i++) {
            int val = coins[i - 1];
            for (int j = 0; j < amount + 1; j++) {
                dp[i][j] = dp[i - 1][j];
                for (int k = 1; val * k <= j; k++) {
                    dp[i][j] += dp[i - 1][j - val * k];
                }
            }
        }
        return dp[coins.length][amount];
    }

    //    public int change(int amount, int[] coins) {
//        int[][] dp = new int[coins.length][amount + 1];
//        dp[0][0]=1;
//        dp[0][coins[0]] = 1;
//        for (int i = 1; i <= coins.length; i++) {
//            int val = coins[i];
//            for (int j = 0; j < amount + 1; j++) {
//                dp[i][j] = dp[i - 1][j];
//                for (int k = 1; val * k <= j; k++) {
//                    dp[i][j] += dp[i - 1][j - val * k];
//                }
//            }
//        }
//        return dp[coins.length-1][amount];
//    }
    public int combinationSum4(int[] nums, int target) {
        int[] dp = new int[target + 1];
        Arrays.sort(nums);
        dp[0] = 1;
        for (int i = 1; i <= target; i++) {
            for (int j = 0; j < nums.length; j++) {
                if (i < nums[j]) {
                    break;
                }
                dp[i] += dp[i - nums[j]];
            }
        }
        return dp[target];
    }

    public int maxProfit(int[] prices) {
        int min = Integer.MAX_VALUE, max = Integer.MIN_VALUE, res = 0;
        for (int i = 0; i < prices.length; i++) {
            if (prices[i] < min) {
                min = prices[i];
            }
            if (prices[i] > min) {
                res = Math.max(res, prices[i] - min);
            }
        }
        return res;
    }

    public int mySqrt(int x) {
        int left = 1, right = x;
        while (left < right) {
            int mid = (right - left) / 2 + left;
            int power = mid * mid;
            if (power == x) {
                return mid;
            } else if (power > x) {
                right = mid - 1;
            } else {
                left = mid;
            }
        }
        return left;
    }

    //    public int mySqrt(int x) {
//        int low = 0, high = x;
//        while (low < high) {
//            int mid = low + (high - low) / 2;
//            int r = mid * mid;
//            if (r > x) high = mid;
//            else if (r < x) low = mid + 1;
//            else return (int) mid;
//        }
//        return low == x ? x : (int) low - 1;
//    }
    public char nextGreatestLetter(char[] letters, char target) {
        int left = 0, right = letters.length - 1;
        while (left < right) {
            int mid = (right - left) / 2 + left;
            if (letters[mid] <= target) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        if (left == letters.length - 1 && letters[left] < target) return letters[0];
        return letters[left];
    }

    public int singleNonDuplicate(int[] nums) {
        if (nums.length == 1) return nums[0];
        int left = 0, right = nums.length - 1;
        while (left < right) {
            int mid = (right - left) / 2 + left;
            int ml = nums[mid - 1], mr = nums[mid + 1], m = nums[mid], lcount = 0;
            if (m != ml && m != mr) {
                return m;
            } else if (m == ml) {
                if (right - left == 2) {
                    return mr;
                }
                lcount = mid - 1 - left;
                if ((lcount & 1) == 1) {
                    right = mid - 2;
                } else {
                    left = mid + 1;
                }
            } else if (m == mr) {
                if (right - left == 2) {
                    return ml;
                }
                lcount = mid - left;
                if ((lcount & 1) == 1) {
                    right = mid - 1;
                } else {
                    left = mid + 2;
                }
            }

        }
        return nums[left];
    }

    public int firstBadVersion(int n) {
        int left = 1, right = n;
        while (left < right) {
            int mid = (right - left) / 2 + left;
            if (!isBadVersion(mid)) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return left;
    }

    private boolean isBadVersion(int mid) {
        return false;
    }

    public int findMin(int[] nums) {
        int left = 0, right = nums.length - 1;
        if (nums[right] > nums[left]) return nums[left];
        while (left < right) {
            int mid = (right - left) / 2 + left;
            if (nums[mid] == nums[left]) {
                right--;
            } else if (nums[mid] > nums[left]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return nums[left];
    }

//    public int kthSmallest(TreeNode root, int k) {
//        return preOrder(root,k);
//    }
//
//    private int preOrder(TreeNode root, int k) {
//        if(root==null){
//
//        }
//    }

    int res = 0, k;

    public int kthSmallest2(TreeNode root, int _k) {
        this.k = _k;
        preOrder2(root);
        return res;
    }

    private void preOrder2(TreeNode root) {
        if (root == null) return;
        preOrder2(root.left);
        k--;
        if (k == 0) {
            res = root.val;
        }
        preOrder2(root.right);
    }

    public boolean canPlaceFlowers(int[] flowerbed, int n) {
        int count = 0;
        for (int i = 0; i < flowerbed.length; ) {
            if (i == flowerbed.length - 1 && i - 1 >= 0 && flowerbed[i] == 0 && flowerbed[i - 1] == 0) {
                count++;
                i += 2;
            }
            if (i == 0 && flowerbed[i] == 0 && flowerbed[i + 1] == 0) {
                count++;
                i += 2;
            } else if (i > 0 && i + 1 < flowerbed.length && flowerbed[i] == 0 && flowerbed[i - 1] == 0 && flowerbed[i + 1] == 0) {
                count++;
                i += 2;
            } else {
                i++;
            }
        }
        return count >= n;
    }

    public String reverseWords(String s) {
        String[] array = s.split(" ");
        int le = 0, ri = array.length - 1;
        while (le < ri) {
            String tmp = array[ri];
            array[ri] = array[le];
            array[le] = tmp;
            le++;
            ri--;
        }
        StringBuffer sb = new StringBuffer();
        for (int i = 0; i < array.length; i++) {
            sb.append(array[i].trim());
            sb.append(" ");
        }
        return sb.toString().trim();
    }

    public int minInsertions(String s) {
        int count = 0;
        Stack<Character> left = new Stack<>();
        Stack<Character> right = new Stack<>();
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                if (!right.isEmpty()) {
                    while (!right.isEmpty()) {
                        right.pop();
                        if (!right.isEmpty()) {
                            right.pop();
                        } else {
                            count++;
                        }
                        count++;// for left  parenthesis
                    }
                } else {
                    left.push('(');
                }
            } else
            //for right parenthesis
            {
//                right.push(')');
                if (!left.isEmpty()) {
//                    if(right.isEmpty())
                    if (i + 1 < s.length() && s.charAt(i + 1) == ')') {
                        i++;
                        left.pop();
                    } else {
                        left.pop();
                        count++;
                    }
                } else {
                    right.push(')');
                }
            }
        }
        while (!right.isEmpty() || !left.isEmpty()) {
            if (!left.isEmpty()) {
                left.pop();
            } else {
                count++;
            }
            if (!right.isEmpty()) {
                right.pop();
            } else {
                count++;
            }
            if (!right.isEmpty()) {
                right.pop();
            } else {
                count++;
            }
        }
        return count;
    }

    List<List<Integer>> res2;
    int mark = 0, depth = 0;

    public List<List<Integer>> verticalOrder(TreeNode root) {
        res2 = new ArrayList<>();
        dfs(root, 0);
        return res2;
    }

    private void dfs(TreeNode root, int index) {
        if (root == null) return;
        dfs(root.left, index - 1);
        mark++;
        if (mark == 1) {
            depth = -index;
        }
        if (res2.size() <= index + depth) {
            res2.add(new ArrayList<>());
        }
        res2.get(index + depth).add(root.val);
        dfs(root.right, index + 1);
    }

    public List<List<Integer>> verticalOrder2(TreeNode root) {
        Map<Integer, List<Integer>> res = new TreeMap<>();
        Map<TreeNode, Integer> depthRecord = new HashMap<>();
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        depthRecord.put(root, 0);
        while (!queue.isEmpty()) {
            TreeNode cur = queue.poll();
            int curDepth = depthRecord.get(cur);
            if (cur.left != null) {
                queue.offer(cur.left);
                depthRecord.put(cur.left, curDepth - 1);
            }

        }
        return new ArrayList<>(res.values());
    }

    int sum = 0;

    public int rangeSumBST(TreeNode root, int low, int high) {
        dfs2(root, low, high);
        return sum;
    }

    private void dfs2(TreeNode root, int low, int high) {
        if (root == null) return;
        if (root.val >= low) {
            dfs2(root.left, low, high);
        }
        if (root.val <= high && root.val >= low) {
            sum += root.val;
        }
        if (root.val <= high) {
            dfs2(root.right, low, high);
        }
    }

    public ListNode reverseKGroup(ListNode head, int k) {
        if (head == null) return null;
        ListNode dum = new ListNode(-1);
        dum.next = head;
        ListNode cur = head, last = dum;
        Stack<ListNode> stack = new Stack<>();
        while (cur != null) {
            stack.push(cur);
            cur = cur.next;
            if (stack.size() == k) {
                while (!stack.isEmpty()) {
                    last.next = stack.pop();
                    last = last.next;
                }
                last.next = cur;
            }
        }
        return dum.next;
    }

    public List<Integer> findDuplicates(int[] nums) {
        List<Integer> list = new ArrayList<>();
        if (nums == null || nums.length == 0) return list;
        Arrays.sort(nums);
        int pre = nums[0];
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] == pre) {
                list.add(nums[i]);
            }
            pre = nums[i];
        }
        return list;
    }

    int res3 = Integer.MAX_VALUE;

    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        Set<String> wordSet = new HashSet<>(wordList);
        Set<String> curSet = new HashSet<>();
        if (!wordSet.contains(endWord)) return 0;
        dfs3(beginWord, endWord, wordSet, curSet, 0);
        return res3 == Integer.MAX_VALUE ? 0 : res3;
    }

    private void dfs3(String beginWord, String endWord, Set<String> wordSet, Set<String> set, int dep) {
        if (beginWord.equals(endWord)) {
            res3 = Math.min(res3, dep);
            return;
        }
        char[] charArray = beginWord.toCharArray();
        for (int i = 0; i < charArray.length; i++) {
            char curChar = charArray[i];
            for (char j = 'a'; j <= 'z'; j++) {
                if (curChar == j) continue;
                charArray[i] = j;
                String tmp = new String(charArray);
                if (set.contains(tmp)) {
                    continue;
                }
                if (wordSet.contains(tmp)) {
                    set.add(tmp);
                    dfs3(tmp, endWord, wordSet, set, dep + 1);
                    set.remove(tmp);
                }
            }
            charArray[i]=curChar;
        }
    }

    public int hIndex(int[] citations) {
        Arrays.sort(citations);
        for (int i = citations.length-1; i >=0 ; i--) {
            if(citations[i]==citations.length-i-1||citations[i]==citations.length-i){
                return citations[i];
            }
            if(citations[i]<=citations.length-i-1||citations[i]<=citations.length-i){
                return i+1;
            }
        }
        return 0;
    }

    public static void main(String[] args) {
        Lc7 lc7 = new Lc7();
//        lc7.mySqrt(8);
//        lc7.singleNonDuplicate(new int[]{1, 1, 2});
//        lc7.reverseWords("a good   example");
        lc7.ladderLength("hit", "cog", Arrays.asList("hot","dot","dog","lot","log","cog"));
//        TimeMap obj = new TimeMap();
//        obj.set("foo", "high", 10);
////        String param_2 = obj.get("foo", 1);
////        System.out.println(param_2);
////        String param_3 = obj.get("foo", 3);
////        System.out.println(param_2);
//
//        obj.set("foo", "low", 20);
//        String param_4 = obj.get("foo", 5);
//        System.out.println(param_4);

//        int res = lc7.findTargetSumWays2(new int[]{1, 1, 1, 1, 1}, 3);
//        int res = lc7.findMaxForm2(new String[]{"10","0001","111001","1","0"}, 5, 3);
//        int res = lc7.change(5, new int[]{1, 2, 5});
//
//        System.out.println(res);
//        lc7.lengthOfLongestSubstring("abcabcbb");
//        lc7.findTargetSumWays(new int[]{1}, 2);
//        lc7.threeSum(new int[]{-1, 0, 1, 2, -1, -4});
//         boolean res= lc7.canPartition(new int[]{1,3,4,4});
//        System.out.println(res);
    }
}

//class MyCalendar {
//    List<int[]> list;
//    Comparator<int[]> comparator;
//
//    public MyCalendar() {
//        list = new ArrayList<>();
//        comparator = new Comparator<int[]>() {
//            @Override
//            public int compare(int[] o1, int[] o2) {
//                return o1[0] - o2[0];
//            }
//        };
//    }
//
//    public boolean book(int start, int end) {
//
//        int index = Collections.binarySearch(list, new int[]{start, end}, comparator);
//        if (index > 0) {
////            int[] cur = list.get(index);
//            return false;
//        } else if (index < 0) {
//            int tobe = -(index + 1);
//            if (tobe > 0) {
//                int[] pre = list.get();
//            } else {
//                if (list.size() > 0) {
//                    int[] next = list.get(0);
//                    if (next[0] > end) {
//                        list.add(new int[]{start, end});
//                        return true;
//                    } else if (next[0] == end) {
//                        next[0] = start;
//                        return true;
//                    } else {
//                        return false;
//                    }
//                }
//                return true;
//            }
//        }
//    }
//}
class ListNode {
    int val;
    ListNode next;

    ListNode() {
    }

    ListNode(int val) {
        this.val = val;
    }

    ListNode(int val, ListNode next) {
        this.val = val;
        this.next = next;
    }
}

class MyCalendar {
    TreeMap<Integer, Integer> calendar;

    MyCalendar() {
        calendar = new TreeMap();
    }

    public boolean book(int start, int end) {
        Integer prev = calendar.floorKey(start),
                next = calendar.ceilingKey(start);
        if ((prev == null || calendar.get(prev) <= start) &&
                (next == null || end <= next)) {
            calendar.put(start, end);
            return true;
        }
        return false;
    }
}


class TimeNode {
    String value;
    int timestamp;

    public TimeNode(String value, int timestamp) {
        this.value = value;
        this.timestamp = timestamp;
    }
}

class TimeMap {
    Map<String, List<TimeNode>> map;

    /**
     * Initialize your data structure here.
     */
    public TimeMap() {
        map = new HashMap<>();
    }

    public void set(String key, String value, int timestamp) {
        if (map.containsKey(key)) {
            map.get(key).add(new TimeNode(value, timestamp));
        } else {
            List<TimeNode> list = new ArrayList<>();
            list.add(new TimeNode(value, timestamp));
            map.put(key, list);
        }
    }

    public String get(String key, int timestamp) {
        if (!map.containsKey(key)) {
            return "";
        } else {
            List<TimeNode> tmp = map.get(key);
            int left = 0, right = tmp.size() - 1;
            while (right > left) {
//                int mid = left + (right - left) / 2;
                int mid = (right + left + 1) / 2;
                TimeNode cur = tmp.get(mid);
                if (cur.timestamp == timestamp) {
                    return cur.value;
                } else if (cur.timestamp > timestamp) {
                    right = mid - 1;
                } else {
                    left = mid;
                }
            }
            return tmp.get(left).value;
        }
    }
}

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

    @Override
    public String toString() {
        return "TreeNode{" +
                "val=" + val +
                ", left=" + left +
                ", right=" + right +
                '}';
    }
}