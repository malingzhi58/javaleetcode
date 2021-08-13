import java.util.Arrays;
import java.util.Deque;
import java.util.*;

public class Lc6 {
    public int[] maxSlidingWindow2(int[] nums, int k) {
        int[] res = new int[nums.length + 1 - k];
        Deque<Integer> deque = new LinkedList<>();
        for (int i = 0; i < nums.length; i++) {
            while (!deque.isEmpty() && nums[deque.peekLast()] < nums[i]) {
                deque.pollLast();
            }
            deque.offerLast(i);
            while (!deque.isEmpty() && deque.peekFirst() <= i - k) {
                deque.pollFirst();
            }
            if (i >= k) {
                res[i - k] = nums[deque.peekFirst()];
            }
        }
        return res;
    }

    public int maxIceCream(int[] costs, int coins) {
        if (costs == null || costs.length == 0) {
            return 0;
        }
        int res = 0, sum = 0, i = 0;
        Arrays.sort(costs);
        for (; i < costs.length; i++) {
            sum += costs[i];
            if (sum < coins) {
                res++;
            } else if (sum == coins) {
                res++;
                break;
            } else {
                res--;
                break;
            }
        }

        return res;
    }

    public int[] maxSlidingWindow(int[] nums, int k) {
        int n = nums.length;
        PriorityQueue<int[]> pq = new PriorityQueue<int[]>(new Comparator<int[]>() {
            public int compare(int[] pair1, int[] pair2) {
//                return pair1[0] != pair2[0] ? pair2[0] - pair1[0] : pair2[1] - pair1[1];
                return pair1[0] != pair2[0] ? pair2[0] - pair1[0] : pair1[1] - pair2[1];
            }
        });
        for (int i = 0; i < k; ++i) {
            pq.offer(new int[]{nums[i], i});
        }
        int[] ans = new int[n - k + 1];
        ans[0] = pq.peek()[0];
        for (int i = k; i < n; ++i) {
            pq.offer(new int[]{nums[i], i});
            while (pq.peek()[1] <= i - k) {
                pq.poll();
            }
            ans[i - k + 1] = pq.peek()[0];
        }
        return ans;
    }

    public int[] maxSlidingWindow3(int[] nums, int k) {
        int n = nums.length;
        PriorityQueue<int[]> pq = new PriorityQueue<>(new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] != o2[0] ? o2[0] - o1[0] : o2[1] - o1[1];
            }
        });
        int[] ans = new int[n - k + 1];
        for (int i = 0; i < k; i++) {
            pq.offer(new int[]{nums[i], i});
        }
        for (int i = k; i < n; i++) {
            while (pq.peek()[1] <= i - k) {
                pq.poll();
            }
            ans[i - k + 1] = pq.peek()[0];
            pq.offer(new int[]{nums[i], i});
        }
        return ans;
    }

    public boolean checkInclusion(String s1, String s2) {
        Map<Character, Integer> map = new HashMap<>();
        int count = 0, l = 0, r = 0;
        for (Character character : s1.toCharArray()) {
            map.put(character, map.getOrDefault(character, 0) + 1);
        }
        Map<Character, Integer> tmpMap = new HashMap<>();
        while (r < s2.length()) {
            if (!map.containsKey(s2.charAt(r))) {
                r++;
                l++;
                tmpMap.clear();
                count = 0;
            } else {
                tmpMap.put(s2.charAt(r), tmpMap.getOrDefault(s2.charAt(r), 0) + 1);
                count++;
                while (tmpMap.get(s2.charAt(r)) > map.get(s2.charAt(r)) || r - l >= s1.length()) {
                    tmpMap.put(s2.charAt(l), tmpMap.getOrDefault(s2.charAt(l), 0) - 1);
                    l++;
                    count--;
                }
                r++;

            }
            if (count == s1.length()) {
                return true;
            }
        }
        return false;
    }

    public boolean checkInclusion4(String s1, String s2) {
        int l1 = s1.length(), l2 = s2.length(), left = 0;
        int[] count = new int[26];
        for (int i = 0; i < l1; i++) {
            count[s1.charAt(i) - 'a']--;
        }
        for (int ri = 0; ri < l2; ri++) {
            count[s2.charAt(ri)]++;
            while (count[s2.charAt(ri) - 'a'] > 0) {
                count[s2.charAt(left) - 'a']--;
                left++;
            }
            if (ri - left + 1 == l1) {
                return true;
            }
        }
        return false;
    }

    public String frequencySort(String s) {
        StringBuffer sb = new StringBuffer();
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < s.length(); i++) {
            map.put(s.charAt(i) - 'A', map.getOrDefault(s.charAt(i) - 'A', 0) + 1);
        }
        PriorityQueue<int[]> pq = new PriorityQueue<>(new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o2[1] - o1[1];
            }
        });
        for (Map.Entry<Integer, Integer> entry : map.entrySet()
        ) {
            pq.offer(new int[]{entry.getKey(), entry.getValue()});
        }
        while (!pq.isEmpty()) {
            int[] tmp = pq.poll();
            int num = tmp[1];
            char character = (char) (tmp[0] + 'A');
            while (num > 0) {
                sb.append(character);
                num--;
            }
        }
        return sb.toString();
    }

    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(target - nums[i])) {
                return new int[]{map.get(target - nums[i]), i};
            }
            map.put(nums[i], i);
        }
        return new int[]{};
    }

    public class ListNode {
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

    public ListNode addTwoNumbers2(ListNode l1, ListNode l2) {
        int carry = 0;
        ListNode res = l1;
        while (l1 != null && l2 != null) {
            int sum = l1.val + l2.val + carry;
            l1.val = sum % 10;
            carry = sum / 10;
            l1 = l1.next;
            l2 = l2.next;
        }
        while (l1 != null && l1.next != null) {
            int sum = l1.val + carry;
            l1.val = sum % 10;
            carry = sum / 10;
            l1 = l1.next;
        }
        while (l2 != null && l2.next != null) {
            int sum = l2.val + carry;
            l2.val = sum % 10;
            carry = sum / 10;
            l2 = l2.next;
        }
        if (l1 != null) {
            int sum = l1.val + carry;
            l1.val = sum % 10;
            carry = sum / 10;
            if (carry != 0) {
                l1.next = new ListNode(carry);
            }
        }
        if (l2 != null) {
            int sum = l2.val + carry;
            l2.val = sum % 10;
            carry = sum / 10;
            if (carry != 0) {
                l2.next = new ListNode(carry);
            }
        }

        return res;
    }

    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        int carry = 0;
        ListNode res = new ListNode(-1), pre = res;
        while (l1 != null || l2 != null || carry > 0) {
            int n1 = l1 == null ? 0 : l1.val;
            int n2 = l2 == null ? 0 : l2.val;
            int sum = carry + n1 + n2;
            res.next = new ListNode(sum % 10);
            System.out.println(sum % 10);
            carry = sum / 10;
            res = res.next;
            if (l1 != null) {
                l1 = l1.next;
            }
            if (l2 != null) {
                l2 = l2.next;
            }
        }
        return pre.next;
    }

    public List<String> letterCombinations(String digits) {
        List<String> res = new ArrayList<>();
        if (digits == null || digits.length() == 0) {
            return res;
        }
        StringBuffer sb = new StringBuffer();
        Map<Character, String> map = new HashMap<>();
        map.put('2', "abc");
        map.put('3', "def");
        map.put('4', "ghi");
        map.put('5', "jkl");
        map.put('6', "mno");
        map.put('7', "pqrs");
        map.put('8', "tuv");
        map.put('9', "wxyz");
        dfs(sb, 0, res, digits, map);
        return res;
    }

    private void dfs(StringBuffer sb, int pos, List<String> res, String digits, Map<Character, String> map) {
        if (pos == digits.length()) {
            res.add(sb.toString());
            return;
        }
        String cur = map.get(digits.charAt(pos));
        for (int i = 0; i < cur.length(); i++) {
            sb.append(cur.charAt(i));
            dfs(sb, pos + 1, res, digits, map);
            sb.deleteCharAt(sb.length() - 1);
        }

    }

//    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
//        int total =nums1.length+nums2.length;
//        int[] res = new int[total];
//        int i =0,r =0,index=0;
//        while(i<nums1.length&&r<nums2.length){
//            if(nums1[i]<nums2[r]){
//                res[index++] = nums1[i++];
//            }else{
//                res[index++] = nums2[r++];
//            }
//        }
//        while(i<nums1.length){
//            res[index++] = nums1[i++];
//        }
//        while(r<nums2.length){
//            res[index++] = nums2[r++];
//        }
//        if((total&1)==1){
//            return res[total/2];
//        }else{
//            return (res[total/2]+res[total/2-1])/2.0;
//        }
//    }

    //    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
//        int length1 = nums1.length, length2 = nums2.length;
//        int totalLength = length1 + length2;
//        if (totalLength % 2 == 1) {
//            int midIndex = totalLength / 2;
//            double median = getKthElement(nums1, nums2, midIndex + 1);
//            return median;
//        } else {
//            int midIndex1 = totalLength / 2 - 1, midIndex2 = totalLength / 2;
//            double median = (getKthElement(nums1, nums2, midIndex1 + 1) + getKthElement(nums1, nums2, midIndex2 + 1)) / 2.0;
//            return median;
//        }
//    }
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int length1 = nums1.length, length2 = nums2.length;
        int totalLength = length1 + length2;
        if (totalLength % 2 == 1) {
            int midIndex = totalLength / 2;
            double median = getKthElement2(nums1, nums2, midIndex + 1);
            return median;
        } else {
            int midIndex1 = totalLength / 2 - 1, midIndex2 = totalLength / 2;
            double median = (getKthElement2(nums1, nums2, midIndex1 + 1) + getKthElement2(nums1, nums2, midIndex2 + 1)) / 2.0;
            return median;
        }
    }

    public int getKthElement2(int[] nums1, int[] nums2, int k) {
        int s1 = 0, s2 = 0;
        while (true) {
            int len1 = nums1.length - 1 - s1 + 1, len2 = nums2.length - 1 - s2 + 1;
            if (len1 == 0) return nums2[s2 + k - 1];
            if (len2 == 0) return nums1[s1 + k - 1];
            if (k == 1) return Math.min(nums1[s1], nums2[s2]);
            int pivot1 = s1 + Math.min(len1, k / 2) - 1, pivot2 = s2 + Math.min(len2, k / 2) - 1;
            if (nums1[pivot1] < nums2[pivot2]) {
                k -= (pivot1 - s1 + 1);
                s1 = pivot1 + 1;
            } else {
                k -= (pivot2 - s2 + 1);
                s2 = pivot2 + 1;
            }
        }
    }

    public int getKthElement(int[] nums1, int[] nums2, int k) {
        /* 主要思路：要找到第 k (k>1) 小的元素，那么就取 pivot1 = nums1[k/2-1] 和 pivot2 = nums2[k/2-1] 进行比较
         * 这里的 "/" 表示整除
         * nums1 中小于等于 pivot1 的元素有 nums1[0 .. k/2-2] 共计 k/2-1 个
         * nums2 中小于等于 pivot2 的元素有 nums2[0 .. k/2-2] 共计 k/2-1 个
         * 取 pivot = min(pivot1, pivot2)，两个数组中小于等于 pivot 的元素共计不会超过 (k/2-1) + (k/2-1) <= k-2 个
         * 这样 pivot 本身最大也只能是第 k-1 小的元素
         * 如果 pivot = pivot1，那么 nums1[0 .. k/2-1] 都不可能是第 k 小的元素。把这些元素全部 "删除"，剩下的作为新的 nums1 数组
         * 如果 pivot = pivot2，那么 nums2[0 .. k/2-1] 都不可能是第 k 小的元素。把这些元素全部 "删除"，剩下的作为新的 nums2 数组
         * 由于我们 "删除" 了一些元素（这些元素都比第 k 小的元素要小），因此需要修改 k 的值，减去删除的数的个数
         */

        int length1 = nums1.length, length2 = nums2.length;
        int index1 = 0, index2 = 0;

        while (true) {
            // 边界情况
            if (index1 == length1) {
                return nums2[index2 + k - 1];
            }
            if (index2 == length2) {
                return nums1[index1 + k - 1];
            }
            if (k == 1) {
                return Math.min(nums1[index1], nums2[index2]);
            }
            // 正常情况
            int half = k / 2;
            int newIndex1 = Math.min(index1 + half, length1) - 1;
            int newIndex2 = Math.min(index2 + half, length2) - 1;
            int pivot1 = nums1[newIndex1], pivot2 = nums2[newIndex2];
            if (pivot1 <= pivot2) {
                k -= (newIndex1 - index1 + 1);
                index1 = newIndex1 + 1;
            } else {
                k -= (newIndex2 - index2 + 1);
                index2 = newIndex2 + 1;
            }
        }
    }

    public double findMedianSortedArrays3(int[] nums1, int[] nums2) {
        int n = nums1.length;
        int m = nums2.length;
        int left = (n + m + 1) / 2;
        int right = (n + m + 2) / 2;
        return (helper(nums1, 0, nums2, 0, left) + helper(nums1, 0, nums2, 0, right)) / 2;
    }

    private double helper(int[] nums1, int s1, int[] nums2, int s2, int k) {
        int len1 = nums1.length - 1 - s1 + 1;
        int len2 = nums2.length - 1 - s2 + 1;
        if (len1 > len2) return helper(nums2, s2, nums1, s1, k);
        if (len1 == 0) return nums2[s2 + k - 1];
        if (k == 1) return Math.min(nums1[s1], nums2[s2]);
        int i = s1 + Math.min(k / 2, len1) - 1;
        int j = s2 + Math.min(k / 2, len2) - 1;
        if (nums1[i] < nums2[j]) {
            return helper(nums1, i + 1, nums2, s2, k - (i - s1 + 1));
        } else {
            return helper(nums1, s1, nums2, j + 1, k - (j - s2 + 1));
        }
    }

    public double findMedianSortedArrays2(int[] nums1, int[] nums2) {
        int n = nums1.length;
        int m = nums2.length;
        int left = (n + m + 1) / 2;
        int right = (n + m + 2) / 2;
        //将偶数和奇数的情况合并，如果是奇数，会求两次同样的 k 。
        return (getKth(nums1, 0, n - 1, nums2, 0, m - 1, left) + getKth(nums1, 0, n - 1, nums2, 0, m - 1, right)) * 0.5;
    }

    private int getKth(int[] nums1, int start1, int end1, int[] nums2, int start2, int end2, int k) {
        int len1 = end1 - start1 + 1;
        int len2 = end2 - start2 + 1;
        //让 len1 的长度小于 len2，这样就能保证如果有数组空了，一定是 len1
        if (len1 > len2) return getKth(nums2, start2, end2, nums1, start1, end1, k);
        if (len1 == 0) return nums2[start2 + k - 1];

        if (k == 1) return Math.min(nums1[start1], nums2[start2]);

        int i = start1 + Math.min(len1, k / 2) - 1;
        int j = start2 + Math.min(len2, k / 2) - 1;

        if (nums1[i] > nums2[j]) {
            return getKth(nums1, start1, end1, nums2, j + 1, end2, k - (j - start2 + 1));
        } else {
            return getKth(nums1, i + 1, end1, nums2, start2, end2, k - (i - start1 + 1));
        }
    }

    public String longestPalindrome(String s) {
        if (s == null || s.length() == 0) {
            return "";
        }
        int max = 0;
        String res = "";
        for (int i = 0; i < s.length(); i++) {
            for (int j = i + 1; j < s.length(); j++) {
                if (j - i + 1 > max && isPalindrome(i, j, s)) {
                    max = j - i + 1;
                    res = s.substring(i, j + 1);
                }
            }
        }
        return res;
    }

    private boolean isPalindrome(int i, int j, String s) {
        while (i < j) {
            if (s.charAt(i) == s.charAt(j)) {
                i++;
                j--;
            } else {
                return false;
            }
        }
        return true;
    }

    public String convert(String s, int numRows) {
        if (numRows == 1) {
            return s;
        }
        List<List<Character>> res = new ArrayList<>();
        for (int i = 0; i < numRows; i++) {
            res.add(new ArrayList<>());
        }
        int loop = 0, index = 0;
        while (index < s.length()) {
            while (loop < numRows && index < s.length()) {
                res.get(loop++).add(s.charAt(index++));
            }
            loop--;
            loop--;
            while (loop >= 0 && index < s.length()) {
                res.get(loop--).add(s.charAt(index++));
            }
            loop++;
            loop++;
        }
        StringBuffer sb = new StringBuffer();
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < res.get(i).size(); j++) {
                sb.append(res.get(i).get(j));
            }
        }
        return sb.toString();
    }

    //    public int[] findErrorNums(int[] nums) {
//        if(nums==null||nums.length==0){
//            return new int[]{};
//        }
//        Map<Integer,Integer> set = new HashMap<>();
//        for(int i=0;i<nums.length;i++){
//            if(set.containsKey(nums[i])){
//                if(nums[i]!=i+1){
//                    return new int[]{nums[i],i+1};
//                }else{
//                    return new int[]{nums[i],set.get(nums[i])+1};
//                }
//            }
//            set.put(nums[i],i);
//
//        }
//        return new int[]{};
//    }
//    public int[] findErrorNums(int[] nums) {
//        if (nums == null || nums.length == 0) {
//            return new int[]{};
//        }
//        Arrays.sort(nums);
//        for (int i = 0; i < nums.length; i++) {
//            if(nums[i]!=i+1){
//                return new int[]{nums[i],i+1};
//            }
//        }
//        return new int[]{};
//    }
    public int[] findErrorNums(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new int[]{};
        }
        int[] res = new int[nums.length];
        for (int i = 0; i < nums.length; i++) {
            res[nums[i] - 1]++;
        }
        int miss = -1, replica = -1;
        for (int i = 0; i < nums.length; i++) {
            if (res[i] == 0) {
                miss = i + 1;
            }
            if (res[i] == 2) {
                replica = i + 1;
            }
            if (miss != -1 && replica != -1) {
                return new int[]{replica, miss};
            }
        }
        return new int[]{replica, miss};
    }

    public int myAtoi(String s) {
        if (s == null || s.length() == 0) {
            return 0;
        }
        s = s.trim();
        int flag = 1;
        int index = 0;
        long sum = 0;
        if (s.charAt(index) == '-') {
            flag = -1;
            index++;
        }
        if (s.charAt(index) == '+') {
            index++;
        }
        if (index >= s.length() || s.charAt(index) > '9' || s.charAt(index) < '0') {
            return 0;
        }
        while (index < s.length() && s.charAt(index) == '0') {
            index++;
        }
        StringBuffer sb = new StringBuffer();
        while (index < s.length() && (s.charAt(index) <= '9' && s.charAt(index) >= '0')) {
            sb.append(s.charAt(index++));
        }
//        sb.reverse();
        for (int i = 0; i < sb.length(); i++) {
            sum = sum * 10 + (sb.charAt(i) - '0');
        }
        if (sum >= Integer.MAX_VALUE && flag == 1) {
            return Integer.MAX_VALUE;
        }
        if (sum > Integer.MAX_VALUE && flag == -1) {
            return Integer.MIN_VALUE;
        }
        return (int) sum * flag;

    }

    public int[] buildArray(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new int[]{};
        }
        int[] ans = new int[nums.length];
        for (int i = 0; i < nums.length; i++) {
            ans[i] = nums[nums[i]];
        }
        return ans;
    }

    //    public boolean canCross(int[] stones) {
//        Set<Integer> set = new HashSet<>();
//        for (int i = 0; i < stones.length; i++) {
//            set.add(stones[i]);
//        }
//        return dfs2(stones,set,stones[0],1);
//    }
//
//    private boolean dfs2(int[] stones, Set<Integer> set, int startPos, int jump) {
//        if(!set.contains(startPos)){
//            return false;
//        }
//        if(stones[stones.length-1]==startPos){
//            return true;
//        }
//        for(int i=-1;i<=1;i++){
//            int cur = jump+i;
//            if(cur<=0) continue;
//            if(dfs2(stones,set,startPos+cur,cur)){
//                return true;
//            }
//        }
//        return false;
//    }
    public int eliminateMaximum(int[] dist, int[] speed) {
        int[][] array = new int[dist.length][3];
        for (int i = 0; i < dist.length; i++) {
            array[i] = new int[]{dist[i], speed[i], 0};
        }
        Comparator<int[]> comparator = new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                int res = (o1[0] - o1[1]) - (o2[0] - o2[1]);
                if (res != 0) {
                    return res;
                } else {
                    return (o1[0] - o1[1] * 2) - (o2[0] - o2[1] * 2);
                }
            }
        };
        Arrays.sort(array, comparator);
        int killed = 0, killIndex = 0;
        while (killed < dist.length) {
            boolean hasKilled = false;
            for (int i = 0; i < dist.length; i++) {
                if (array[i][2] == 1) {
                    continue;
                }
                if (!hasKilled) {
                    array[i][2] = 1;
                    killed++;
                    hasKilled = true;
                    continue;
                }
                if (array[i][2] == 0) {
                    array[i][0] -= array[i][1];
                }
                if (array[i][2] == 0 && array[i][0] <= 0) {
                    return killed;
                }
            }
            Arrays.sort(array, comparator);
        }
        return killed;
    }

    public int countGoodNumbers(long n) {
        int res = 1;
        boolean flag = true;
        for (int i = 0; i < n; i++) {
            if (flag) {
                res *= 5;
                flag = false;
            } else {
                res *= 4;
                flag = true;
            }
            res %= ((int) Math.pow(10, 9) + 7);
        }
        return (int) (res % ((int) Math.pow(10, 9) + 7));
//        return (int)( res %((int)Math.pow(10,9)+7));
    }

    public double myPow2(double x, int n) {
        if (x == 0.0f) return 0.0d;
        long b = n;
        double res = 1.0;
        if (b < 0) {
            x = 1 / x;
            b = -b;
        }
        while (b > 0) {
            if ((b & 1) == 1)
                res *= x;
            x *= x;
            b >>= 1;
        }
        return res;
    }

    public double myPow3(double x, int n) {
        if (x == 0) return 0;
        return n > 0 ? helper(x, n) : helper(1 / x, -n);
    }

    private double helper(double x, int n) {
        if (n == 0) {
            return 1;
        }
        double half = helper(x, n / 2);
        if ((n & 1) == 1) {
            return half * half * x;
        } else {
            return half * half;
        }
    }

    public double myPow(double x, int n) {
        if (x == 0) return 0;
        if (n < 0) {
            n = -n;
            x = 1 / x;
        }
        double res = 1;
        double tmp = x;
        while (n > 0) {
            if ((n & 1) == 1) {
                res *= tmp;
            }
            tmp *= tmp;
            n >>= 1;
        }
        return res;
    }

    long MOD = 1000000007;

    public int countGoodNumbers2(long n) {
        return (int) ((fastPow(5, (n + 1) / 2) * fastPow(4, (n / 2))) % MOD);
    }

    private double fastPow(double base, long n) {
        double res = 1;
        double tmp = base;
        while (n > 0) {
            if ((n & 1) == 1) {
                res *= tmp;
                res %= MOD;
            }
            tmp *= tmp;
            n >>= 1;
        }
        return res;
    }

    public int climbStairs(int n) {
        int pre = 1, cur = 1, index = 2;
        if (n == 1) return 1;
        while (index <= n) {
            int tmp = pre + cur;
            pre = cur;
            cur = tmp;
            index++;
        }
        return cur;
    }
//    private long fastPow(long base, long n) {
//        long res =1;
//        long tmp =base;
//        while(n>0){
//            if((n&1)==1){
//                res *= tmp;
//                res %= MOD;
//            }
//            tmp *= tmp;
//            n>>=1;
//        }
//        return res;
//    }

    public int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        for (int i = 1; i < amount + 1; i++) {
            dp[i] = Integer.MAX_VALUE;
        }
        Arrays.sort(coins);
        for (int i = 1; i < amount + 1; i++) {
            for (int j = 0; j < coins.length; j++) {
                if (coins[j] > i) {
                    break;
                }
                if (dp[i - coins[j]] != Integer.MAX_VALUE) {
                    dp[i] = Math.min(dp[i - coins[j]] + 1, dp[i]);
                }
            }
        }
        return dp[amount] == Integer.MAX_VALUE ? -1 : dp[amount];
    }

    public int lengthOfLIS3(int[] nums) {
        int[] dp = new int[nums.length];
        int max = 0;
        for (int i = 0; i < nums.length; i++) {
            dp[i] = 1;
            for (int j = 0; j < i; j++) {
                if (nums[j] < nums[i]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                    max = Math.max(max, dp[i]);
                }
            }
        }
        return max;
    }

    public int lengthOfLIS2(int[] nums) {
        int[] tails = new int[nums.length];
        for (int i = 0; i < nums.length; i++) {
            tails[i] = Integer.MAX_VALUE;
        }
        int res = 0;
        for (int num : nums) {
//            int i = 0, j = res;
//            while(i < j) {
//                int m = (i + j) / 2;
//                if(tails[m] < num) i = m + 1;
//                else j = m;
//            }
            if (num < tails[res]) {
                tails[res] = num;
            } else {
                res++;
                tails[res] = num;
            }
//            tails[i] = num;
//            if(res == j) res++;
        }
        return res + 1;
    }

    public int lengthOfLIS(int[] nums) {
        int[] tails = new int[nums.length];
        tails[0] = nums[0];
        int index = 0;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] > tails[index]) {
                index++;
                tails[index] = nums[i];
            } else if (nums[i] < tails[index]) {
                int l = 0, r = index;
                while (l < r) {
                    int mid = l + (r - l) / 2;
                    if (tails[mid] == nums[i]) {
                        l = mid;
                        break;
                    } else if (tails[mid] > nums[i]) {
                        r = mid;
                    } else {
                        l = mid + 1;
                    }
                }
                tails[l] = nums[i];
            }

        }
        return index + 1;
    }

    public String countOfAtoms2(String formula) {
        this.i = 0;
        this.n = formula.length();
        this.formula = formula;
        Stack<Map<String, Integer>> stack = new Stack<>();
        stack.push(new HashMap<String, Integer>());
        while (i < n) {
            char ch = formula.charAt(i);
            if (ch == '(') {
                i++;
                stack.push(new HashMap<String, Integer>());

            } else if (ch == ')') {
                i++;
                Map<String, Integer> popMap = stack.pop();
                Map<String, Integer> topMap = stack.peek();
                int num = parseNum2();
                for (Map.Entry<String, Integer> entry : popMap.entrySet()) {
                    topMap.put(entry.getKey(), topMap.getOrDefault(entry.getKey(), 0) + entry.getValue() * num);
                }
            } else {
                String element = parseAtom2();
                int num = parseNum2();
                Map<String, Integer> topMap = stack.peek();
                topMap.put(element, topMap.getOrDefault(element, 0) + num);
            }
        }
        Map<String, Integer> map = stack.pop();
        TreeMap<String, Integer> treeMap = new TreeMap<>(map);

        StringBuffer sb = new StringBuffer();
        for (Map.Entry<String, Integer> entry : treeMap.entrySet()) {
            String atom = entry.getKey();
            int count = entry.getValue();
            sb.append(atom);
            if (count > 1) {
                sb.append(count);
            }
        }
        return sb.toString();
    }

    private int parseNum2() {
        if (i == n || !Character.isDigit(formula.charAt(i))) {
            return 1; // 不是数字，视作 1
        }
        int res = 0;
        if (formula.charAt(i) <= '9' && formula.charAt(i) >= '0') {
            res += (formula.charAt(i++) - '0');
            while (i < n && Character.isDigit(formula.charAt(i))) {
                res = res * 10 + (formula.charAt(i++) - '0');
            }
        }
        return res;
    }

    private String parseAtom2() {
        StringBuffer sb = new StringBuffer();
        if (formula.charAt(i) <= 'Z' && formula.charAt(i) >= 'A') {
            sb.append(formula.charAt(i++));
            while (i < n && Character.isLowerCase(formula.charAt(i))) {
                sb.append(formula.charAt(i++));
            }
        }
        return sb.toString();
    }

    int i, n;
    String formula;

    public String countOfAtoms(String formula) {
        this.i = 0;
        this.n = formula.length();
        this.formula = formula;

        Deque<Map<String, Integer>> stack = new LinkedList<Map<String, Integer>>();
        stack.push(new HashMap<String, Integer>());
        while (i < n) {
            char ch = formula.charAt(i);
            if (ch == '(') {
                i++;
                stack.push(new HashMap<String, Integer>()); // 将一个空的哈希表压入栈中，准备统计括号内的原子数量
            } else if (ch == ')') {
                i++;
                int num = parseNum(); // 括号右侧数字
                Map<String, Integer> popMap = stack.pop(); // 弹出括号内的原子数量
                Map<String, Integer> topMap = stack.peek();
                for (Map.Entry<String, Integer> entry : popMap.entrySet()) {
                    String atom = entry.getKey();
                    int v = entry.getValue();
                    topMap.put(atom, topMap.getOrDefault(atom, 0) + v * num); // 将括号内的原子数量乘上 num，加到上一层的原子数量中
                }
            } else {
                String atom = parseAtom();
                int num = parseNum();
                Map<String, Integer> topMap = stack.peek();
                topMap.put(atom, topMap.getOrDefault(atom, 0) + num); // 统计原子数量
            }
        }

        Map<String, Integer> map = stack.pop();
        TreeMap<String, Integer> treeMap = new TreeMap<String, Integer>(map);

        StringBuffer sb = new StringBuffer();
        for (Map.Entry<String, Integer> entry : treeMap.entrySet()) {
            String atom = entry.getKey();
            int count = entry.getValue();
            sb.append(atom);
            if (count > 1) {
                sb.append(count);
            }
        }
        return sb.toString();
    }

    public String parseAtom() {
        StringBuffer sb = new StringBuffer();
        sb.append(formula.charAt(i++)); // 扫描首字母
        while (i < n && Character.isLowerCase(formula.charAt(i))) {
            sb.append(formula.charAt(i++)); // 扫描首字母后的小写字母
        }
        return sb.toString();
    }

    public int parseNum() {
        if (i == n || !Character.isDigit(formula.charAt(i))) {
            return 1; // 不是数字，视作 1
        }
        int num = 0;
        while (i < n && Character.isDigit(formula.charAt(i))) {
            num = num * 10 + formula.charAt(i++) - '0'; // 扫描数字
        }
        return num;
    }

    public int longestCommonSubsequence(String text1, String text2) {
        int[][] dp = new int[text2.length()][text1.length()];
        for (int i = 0; i < text1.length(); i++) {
            if (text1.charAt(i) == text2.charAt(0)) {
                dp[0][i] = 1;
                for (int j = i + 1; j < text1.length(); j++) {
                    dp[0][j] = 1;
                }
                break;
            }
        }
        for (int i = 0; i < text2.length(); i++) {
            if (text2.charAt(i) == text1.charAt(0)) {
                dp[i][0] = 1;
                for (int j = i + 1; j < text2.length(); j++) {
                    dp[j][0] = 1;
                }
                break;
            }
        }
        for (int i = 1; i < text2.length(); i++) {
            for (int j = 1; j < text1.length(); j++) {
                if (text2.charAt(i) == text1.charAt(j)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }

            }
        }
        return dp[text2.length() - 1][text1.length() - 1];
    }

    public boolean canCross(int[] stones) {
        Set<Integer> set = new HashSet<>();
        for (int i = 0; i < stones.length; i++) {
            set.add(stones[i]);
        }
        Set<Integer> trace = new HashSet<>();
        return dfs2(stones, set, stones[0], 0, trace);
    }

    private boolean dfs2(int[] stones, Set<Integer> set, int startPos, int jump, Set<Integer> trace) {
        Integer key = startPos * 1000 + jump;

        if (trace.contains(key)) {
            return false;
        } else {
            trace.add(key);
        }
        if (!set.contains(startPos)) {
            return false;
        }

        if (stones[stones.length - 1] == startPos) {
            return true;
        }
        for (int i = -1; i <= 1; i++) {

            int cur = jump + i;
            if (cur <= 0) continue;
            if (dfs2(stones, set, startPos + cur, cur, trace)) {
                return true;
            }
        }
        return false;
    }

    public boolean canCross2(int[] stones) {
        // 回溯法递归
        int n = stones.length;
        Map<Integer, Boolean> map = new HashMap<>();
        return DFS(stones, 0, 0, n, map);
    }

    /**
     * index: 表示现在所处的索引
     * k: 表示上一步跳跃了几个单元格
     * n: 表示数组长度
     * map: 表示经历过的状态
     **/
    private boolean DFS(int[] stones, int index, int k, int n, Map<Integer, Boolean> map) {
        // 递归终止条件
        // System.out.println("index:" + index + "  k:" + k);
        if (index == n - 1) {
            return true;
        }
        int key = index * 1000 + k;
        if (map.containsKey(key)) {
            return false;
        } else {
            map.put(key, true);
        }
        for (int i = index + 1; i < n; i++) {
            int gap = stones[i] - stones[index];
            if (k - 1 <= gap && gap <= k + 1) {
                if (DFS(stones, i, gap, n, map)) {
                    return true;
                }
            } else if (gap > k + 1) {
                break;
            } else {
                continue;
            }
        }
        return false;
    }

    public boolean canCross3(int[] stones) {
        int len = stones.length;
        boolean[][] dp = new boolean[len][len];
        dp[0][0] = true;
        for (int i = 1; i < len; i++) {
            if (stones[i] - stones[i - 1] > i) {
                return false;
            }
        }
        for (int i = 1; i < len; i++) {
            for (int j = i - 1; j >= 0; j--) {
                int k = stones[i] - stones[j];
                if (k > j + 1) break;
                dp[i][k] = dp[j][k - 1] || dp[j][k] || dp[j][k + 1];
                if (i == len - 1 && dp[i][k]) {
                    return true;
                }

            }
        }
        return false;
    }

    public boolean wordBreak(String s, List<String> wordDict) {
        Set<String> set = new HashSet<>(wordDict);
        boolean[] dp = new boolean[s.length() + 1];
        dp[0] = true;
        for (int i = 1; i <= s.length(); i++) {
            for (int j = 0; j < i; j++) {
                if (dp[j] && set.contains(s.substring(j, i))) {
                    dp[i] = true;
                }
            }
        }
        return dp[s.length() - 1];
    }

    public int maxArea(int[] height) {
        int max = 0, le = 0, ri = height.length - 1;

        while (le < ri) {
            int area = (ri - le) * Math.min(height[le], height[ri]);
            if (area > max) {
                max = area;
            }
            if (height[le] < height[ri]) {
                le++;
            } else {
                ri--;
            }
        }
        return max;
    }

    int res = 0;

    public int combinationSum4(int[] nums, int target) {
        dfs3(nums, target, 0);
        return res;
    }

    private void dfs3(int[] nums, int target, int pos) {
        if (target == 0) {
            res++;
            return;
        }
        if (pos == nums.length) {
            return;
        }
        for (int i = pos; i < nums.length; i++) {
            if (nums[i] < target) {
                dfs3(nums, target - nums[i], i + 1);
            }
        }
    }

    public int rob(int[] nums) {
        int[] dp = new int[nums.length + 1];
        dp[1] = nums[0];
        for (int i = 2; i <= nums.length; i++) {
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i - 1]);
        }
        return dp[nums.length];
    }

    public int countPairs2(int[] deliciousness) {
        Arrays.sort(deliciousness);
        int res = 0;
        Set<Integer> okSet = new HashSet<>();
        Set<Integer> notOkSet = new HashSet<>();
        for (int i = 0; i < deliciousness.length; i++) {
            for (int j = i + 1; j < deliciousness.length; j++) {
                int cur = deliciousness[i] + deliciousness[j];
                if (okSet.contains(cur)) {
                    res++;
                    continue;
                } else if (notOkSet.contains(cur)) {
                    continue;
                } else {
                    int oldCur = cur;
                    boolean isPower = true;
                    while (cur != 1) {
                        if (cur % 2 != 0) {
                            notOkSet.add(oldCur);
                            isPower = false;
                            break;
                        } else {
                            cur /= 2;
                        }
                    }
                    if (isPower) {
                        okSet.add(oldCur);
                        res++;
                    }
                }
            }
        }
        return res;
    }

    public int countPairs(int[] deliciousness) {
        int res = 0, max = 0;
        for (Integer each : deliciousness) {
            max = Math.max(each, max);
        }
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < deliciousness.length; i++) {
            for (int j = 1; j <= 2 * max; j <<= 1) {

                if (j >= deliciousness[i] && map.containsKey(j - deliciousness[i])) {
                    res = (res + map.get(j - deliciousness[i])) % 1000000007;
                }
            }
            map.put(deliciousness[i], map.getOrDefault(deliciousness[i], 0) + 1);
        }
        return res;
    }

    public int minPathSum(int[][] grid) {
        int[][] dp = new int[grid.length][grid[0].length];
        dp[0][0] = grid[0][0];
        for (int i = 1; i < grid[0].length; i++) {
            dp[0][i] = dp[0][i - 1] + grid[0][i];
        }
        for (int i = 1; i < grid.length; i++) {
            dp[i][0] = dp[i - 1][0] + grid[i][0];
        }
        for (int i = 1; i < grid[0].length; i++) {
            for (int j = 1; j < grid.length; j++) {
                dp[j][i] = Math.min(dp[j - 1][i], dp[j][i - 1]) + grid[j][i];
            }
        }
        return dp[grid.length - 1][grid[0].length - 1];
    }

    public int uniquePaths2(int m, int n) {
        int[][] dp = new int[m][n];

        for (int i = 0; i < m; i++) {
            dp[i][0] = 1;
        }
        for (int i = 1; i < n; i++) {
            dp[0][i] = 1;
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m - 1][n - 1];
    }

    public int uniquePaths(int m, int n) {
        int[] pre = new int[n];
        int[] cur = new int[n];
        Arrays.fill(pre, 1);
        Arrays.fill(cur, 1);
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                cur[j] = cur[j - 1] + pre[j];
            }
            pre = cur.clone();
        }
        return cur[n - 1];
    }

    class NumArray {
        int[] dp;

        public NumArray(int[] nums) {
            dp = new int[nums.length + 1];
            for (int i = 0; i < nums.length; i++) {
                dp[i + 1] += dp[i] + nums[i];
            }
        }

        public int sumRange(int left, int right) {
            return dp[right + 1] - dp[left];
        }
    }

    public int numberOfArithmeticSlices2(int[] nums) {
        int[] dp = new int[nums.length];
        int sum = 0;
        for (int i = 2; i < nums.length; i++) {
            if (nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]) {
                dp[i] = dp[i - 1] + 1;
                sum += dp[i];
            }
        }
        return sum;
    }

    public int numberOfArithmeticSlices(int[] nums) {
        int sum = 0, pre = 0, cur = 0;
        for (int i = 2; i < nums.length; i++) {
            if (nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]) {
                cur = pre + 1;
                sum += cur;
            } else {
                cur = 0;
            }
            pre = cur;
        }
        return sum;
    }

    public int numSquares(int n) {
        int[] dp = new int[n + 1];
//        dp[0] = 1;
        int max = 0;
        for (int i = 1; i < n; i++) {
            if (i * i >= n) {
                max = i;
            }
        }
        for (int i = 1; i < n + 1; i++) {
            dp[i] = i;
            for (int j = 1; j <= max; j++) {
                if (j * j > i) break;
                dp[i] = Math.min(dp[i], dp[i - j * j] + 1);
            }
        }
        return dp[n];
    }

    public int numDecodings(String s) {
        if (s.charAt(0) == '0') {
            return 0;
        }
        int len = s.length();
        int[] dp = new int[len + 1];
        dp[0] = 1;
        for (int i = 1; i <= len; i++) {
            if (s.charAt(i - 1) != '0') {
                dp[i] += dp[i - 1];
            }
            if (i > 1 && s.charAt(i - 2) != '0' && Integer.parseInt(s.substring(i - 2, i)) <= 26) {
                dp[i] += dp[i - 2];
            }
        }
        return dp[s.length()];
    }

    private boolean isValid(String s, int left, int right) {
        String curString = s.substring(left, right);
        if (curString.startsWith("0")) return false;
        if (Integer.parseInt(curString) > 26)
            return false;
        else
            return true;
    }

    //    public int findLongestChain(int[][] pairs) {
//        int max = Integer.MIN_VALUE,min = Integer.MAX_VALUE;
//        for(int[] each:pairs){
//            if(each[0]<min){
//                min = each[0];
//            }
//            if(each[1]>max){
//                max = each[1];
//            }
//        }
//        int[] dp = new int[max-min+1];
//        for (int i = 0; i < pairs.length; i++) {
//            dp[pairs[i][0]]=i+1;
//            dp[pairs[i][1]]=i+1;
//        }
//        int res =0,cur =0,count =0;
//        for (int i = min; i <=max ; i++) {
//            if(dp[i]!=0){
////                continue last count
//                if(dp[i]>cur){
//                    count=0;
//                    count++;
//
//                }
//                if(dp[i]==cur){
//                    count++;
//                    if(count>=2){
//                        res++;
//                        count=0;
//                        cur = dp[i];
//                    }
//                }
//            }
//        }
//
//    }
    public int numSubarraysWithSum(int[] nums, int goal) {
        int sum = 0, res = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
            if (sum >= goal) {
                if (map.containsKey(sum - goal)) {
                    res += map.get(sum - goal);
                }
            }
            map.put(sum, map.getOrDefault(sum, 0) + 1);
        }
        return res;
    }



    public int numSubarraysWithSum2(int[] nums, int goal) {
        int left = 0, right = 0, lastLeft = 0, res = 0, outerLeft = 0, outerRight = 0, curGoal = goal, len = nums.length;
        while (left < len) {
            if (curGoal == 0) {
                right = left;
                outerLeft = left - 1;
                while (outerLeft >= 0 && outerLeft > lastLeft && nums[outerLeft] == 0) {
                    outerLeft--;
                }
                int leftNum = left - outerLeft;
                outerRight = right + 1;
                while (outerRight < len && nums[outerRight] == 0) {
                    outerRight++;
                }
                int rightNum = outerRight - right;
                res += (leftNum * rightNum);
                left++;
                curGoal = goal;
                while (left < len && nums[left] != 1) {
                    left++;
                }
            } else {
                right++;
                curGoal -= nums[right];
            }
        }

        return res;
    }

    public static void main(String[] args) {
        Lc6 lc6 = new Lc6();
//        lc6.canCross(new int[]{0, 1, 3, 5, 6, 8, 12, 17});
//        int res = lc6.countPairs(new int[]{1, 3, 5, 7, 9});
//        int res = lc6.uniquePaths(1, 1);
//        int res = lc6.numSquares(12);
//        int res = lc6.numDecodings("226");
        int res = lc6.numSubarraysWithSum(new int[]{1, 0, 1, 0, 1}, 2);
        System.out.println(res);

//        System.out.println("res".substring(0,3));

//        double res =lc6.myPow(2,10);
//        int res = lc6.lengthOfLIS(new int[]{10, 9, 2, 5, 3, 4});
//        System.out.println(res);
//        TreeSet<String> set = new TreeSet<>();
//        set.add("D");
//        set.add("A");
//        set.add("BB");
//        for (String each:set
//             ) {
//            System.out.println(each);
//        }


//        Map<int[],Integer> map = new HashMap<>();
//        map.put(new int[]{1,2},1);
//        System.out.println(map.get(new int[]{1,2}));
//        map.put(new int[]{1,2},map.get(new int[]{1,2})+100);
//        System.out.println(map.get(new int[]{1,2}));
//        map.put(new int[]{2,1},1);
//        System.out.println(map.get(new int[]{2,1}));


//        int res=lc6.eliminateMaximum(new int[]{46,33,44,42,46,36,7,36,31,47,38,42,43,48,48,25,28,44,49,47,29,32,30,6,42,9,39,48,22,26,50,34,40,22,10,45,7,43,24,18,40,44,17,39,36},
//        new int[]{1,2,1,3,1,1,1,1,1,1,1,1,1,1,7,1,1,3,2,2,2,1,2,1,1,1,1,1,1,1,1,6,1,1,1,8,1,1,1,3,6,1,3,1,1});
//        System.out.println(res);
//        lc6.findMedianSortedArrays(new int[]{1, 2}, new int[]{3, 4});
//        lc6.findMedianSortedArrays(new int[]{1,3},new int[]{2});
//        lc6.convert("PAYPALISHIRING", 3);
//        int res = lc6.myAtoi("+1");
//        System.out.println(res);
//        System.out.println(Integer.MIN_VALUE);
//        System.out.println(Integer.MIN_VALUE - 1);
//        System.out.println(Integer.MAX_VALUE + 1);
//        lc6.maxSlidingWindow(new int[]{1, 3, -1, -3, 5, 3, 6, 7}, 3);
//        boolean res =lc6.checkInclusion("ky"
//                ,"ainwkckifykxlribaypk");
//        System.out.println(res);
//        System.out.println((int) 'a');
//        System.out.println((int) 'A');
//        PriorityQueue<int[]> pq = new PriorityQueue<>();
////        pq.
//        Map<Integer, Integer> map = new HashMap<>();
//        map.put('b' - 'a', 1);
//        int res = map.get(1);
//        System.out.println(res);
//        int res2 = map.get('b' - 'a');
//        System.out.println(res2);
    }
}
