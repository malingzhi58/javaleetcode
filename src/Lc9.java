import sun.jvm.hotspot.debugger.win32.coff.DebugVC50SymbolEnums;

import java.util.*;
import java.util.concurrent.Semaphore;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.IntConsumer;

public class Lc9 {
    int count = 0;

    public int numDistinct2(String s, String t) {
        dfs(s, t, 0, 0);
        return count;
    }

    private void dfs(String s, String t, int start, int index) {
        if (index == t.length()) {
            count++;
            return;
        }
        for (int i = start; i < s.length(); i++) {
            if (s.charAt(i) == t.charAt(index)) {
                dfs(s, t, i + 1, index + 1);
            }
        }
    }

    public int numDistinct3(String s, String t) {
        int[][] dp = new int[t.length() + 1][s.length() + 1];
        for (int i = 0; i < s.length() + 1; i++) {
            dp[0][i] = 1;
        }
        for (int i = 1; i < t.length() + 1; i++) {
            for (int j = 1; j < s.length() + 1; j++) {
                dp[i][j] = dp[i][j - 1];
                if (s.charAt(j - 1) == t.charAt(i - 1)) {
                    dp[i][j] += dp[i - 1][j - 1];
                }
            }
        }
        return dp[t.length()][s.length()];
    }

    int[][] map;

    public int numDistinct(String s, String t) {
        map = new int[s.length()][t.length()];
        for (int i = 0; i < s.length(); i++) {
            for (int j = 0; j < t.length(); j++) {
                map[i][j] = -1;
            }
        }
        return helper(s, t, s.length() - 1, t.length() - 1);
    }

    private int helper(String s, String t, int slen, int tlen) {
//        if (slen < 0 || tlen < 0) return 1;
        if (tlen < 0) return 1;
        if (slen < 0) return 0;
        if (map[slen][tlen] != -1) return map[slen][tlen];
        if (s.charAt(slen) == t.charAt(tlen)) {
            int left = helper(s, t, slen - 1, tlen);
            int right = helper(s, t, slen - 1, tlen - 1);
//            System.out.println(left+right);
            map[slen][tlen] = left + right;
            return left + right;
        } else {
            int left = helper(s, t, slen - 1, tlen);
//            System.out.println(left);
            map[slen][tlen] = left;
            return left;
        }
    }

    public int findKthLargest(int[] nums, int k) {
        quickSort(nums, 0, nums.length - 1);
        return nums[nums.length - k];
    }

    Random generator = new Random();

    private void quickSort(int[] nums, int start, int end) {
        if (start >= end) return;

        int index = generator.nextInt(end - start + 1) + start;
        swap(nums, start, index);
        int l = start, r = end, tmp = nums[index], base = nums[start];
        while (l < r) {
            while (l < r && base <= nums[r]) {
                r--;
            }
            while (l < r && base >= nums[l]) {
                l++;
            }
            swap(nums, l, r);
        }
        nums[l] = nums[start];
        nums[start] = tmp;
        quickSort(nums, start, l - 1);
        quickSort(nums, l + 1, end);

    }

    private void swap(int[] nums, int start, int index) {
        int tmp = nums[start];
        nums[start] = nums[index];
        nums[index] = tmp;
    }

    public String frequencySort(String s) {
        Map<Integer, Integer> map = new HashMap<>();
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> b[1] - a[1]);
        for (int i = 0; i < s.length(); i++) {
            map.put(s.charAt(i) - 'A', map.getOrDefault(s.charAt(i) - 'A', 0) + 1);
        }
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            pq.add(new int[]{entry.getKey(), entry.getValue()});
        }
        StringBuilder sb = new StringBuilder();
        while (!pq.isEmpty()) {
            int[] tmp = pq.poll();
            for (int i = 0; i < tmp[1]; i++) {
                sb.append((char) (tmp[0] + 'A'));
            }
        }
        return sb.toString();
    }

    public int maximumElementAfterDecrementingAndRearranging(int[] arr) {
        Arrays.sort(arr);
        int len = arr.length;
        arr[0] = 1;
        for (int i = 1; i < len; i++) {
            if (arr[i] - arr[i - 1] > 1) {
                arr[i] = arr[i - 1] + 1;
            }
        }
        return arr[len - 1];
    }

    public int[][] merge(int[][] intervals) {
        List<int[]> list = new ArrayList<>();
        Arrays.sort(intervals, (a, b) -> a[0] - b[0]);
        list.add(intervals[0]);
        for (int i = 1; i < intervals.length; i++) {
            int[] pre = list.get(list.size() - 1);
            if (pre[1] < intervals[i][0]) {
                list.add(intervals[i]);
            } else {
                pre[1] = Math.max(pre[1], intervals[i][1]);
            }
        }
        int[][] res = new int[list.size()][2];
        for (int i = 0; i < list.size(); i++) {
            res[i] = list.get(i);
        }
        return res;
    }

    public int[][] insert(int[][] intervals, int[] newInterval) {
        if (intervals.length == 0) {
            return new int[][]{newInterval};
        }
        List<int[]> res = new ArrayList<>();
        int i = 0, len = intervals.length;
        while (i < len && intervals[i][1] < newInterval[0]) {
            res.add(intervals[i]);
            i++;
        }
        while (i < len && intervals[i][1] >= newInterval[0]) {
            newInterval[0] = Math.min(newInterval[0], intervals[i][0]);
            newInterval[1] = Math.max(newInterval[1], intervals[i][1]);
            i++;
        }
        res.add(newInterval);
        while (i < len && intervals[i][0] > newInterval[1]) {
            res.add(intervals[i]);
            i++;
        }
        int[][] array = new int[res.size()][2];
        for (i = 0; i < res.size(); i++) {
            array[i] = res.get(i);
        }
        return array;
    }

    public String largestNumber(int[] nums) {
//        use double for [999999991,9]
        PriorityQueue<char[]> pq = new PriorityQueue<>(
                new Comparator<char[]>() {
                    @Override
                    public int compare(char[] o1, char[] o2) {
                        //o1 is the short one
                        StringBuilder sb1 = new StringBuilder();
                        sb1.append(o1).append(o2);
                        StringBuilder sb2 = new StringBuilder();
                        sb2.append(o2).append(o1);
                        if (Double.parseDouble(sb1.toString()) > Double.parseDouble(sb2.toString())) {
                            return -1;
                        } else if (Double.parseDouble(sb1.toString()) < Double.parseDouble(sb2.toString())) {
                            return 1;
                        } else {
                            return 0;
                        }
                    }
                }
        );
        for (int i = 0; i < nums.length; i++) {
            pq.offer(String.valueOf(nums[i]).toCharArray());
        }
        StringBuilder sb = new StringBuilder();
        while (!pq.isEmpty()) {
            sb.append(pq.poll());
        }
//  for case [0,0]
        if (sb.charAt(0) == '0' && sb.length() > 1 && sb.charAt(1) == '0') return "0";
        return sb.toString();
    }

    public int maximumGap2(int[] nums) {
        if (nums.length < 2) return 0;
        Arrays.sort(nums);
        int gap = 0;
        for (int i = 1; i < nums.length; i++) {
            gap = Math.max(gap, nums[i] - nums[i - 1]);
        }
        return gap;
    }

    public int maximumGap(int[] nums) {
        if (nums.length < 2) return 0;
        TreeSet<Integer> set = new TreeSet<>();
        for (int i = 0; i < nums.length; i++) {
            set.add(nums[i]);
        }
        int gap = 0, pre = -1;
        for (Integer each : set) {
            if (pre == -1) pre = each;
            else {
                gap = Math.max(gap, each - pre);
                pre = each;
            }
        }
        return gap;
    }

    public int maximumGap6(int[] nums) {
        if (nums.length < 2)
            return 0;
        int n = nums.length;
        int min = Arrays.stream(nums).min().getAsInt();
        int max = Arrays.stream(nums).max().getAsInt();
        int bucketSize = Math.max(1, (max - min) / (n - 1));
        int bucketNum = (max - min) / bucketSize + 1;
        int[][] bucket = new int[bucketNum][2];
        for (int i = 0; i < bucketNum; i++) {
            Arrays.fill(bucket[i], -1);
        }
        for (int i = 0; i < n; i++) {
            int idx = (nums[i] - min) / bucketSize;
            if (bucket[idx][0] == -1) {
                bucket[idx][0] = bucket[idx][1] = nums[i];
            } else {
                bucket[idx][0] = Math.min(bucket[idx][0], nums[i]);
                bucket[idx][1] = Math.max(bucket[idx][1], nums[i]);
            }
        }
        int ret = 0;
        int pre = -1;
        for (int i = 0; i < bucketNum; i++) {
            if (bucket[i][0] == -1) continue;
            if (pre == -1) {
                ret = Math.max(ret, bucket[i][0] - bucket[pre][1]);
            }
            pre = i;
        }
        return ret;
    }

    public int maximumGap4(int[] nums) {
        if (nums.length < 2)
            return 0;
        int gap = Integer.MIN_VALUE, min = nums[0], max = nums[0];
        for (int num : nums) {
            min = Math.min(num, min);
            max = Math.max(num, max);
        }
        if (min == max)
            return 0;
        List<List<Integer>> buckets = bucketSort(nums, min, max);
        int prevMax = Integer.MIN_VALUE;
        for (List<Integer> bucket : buckets) {
            if (bucket.size() != 0 && prevMax != Integer.MIN_VALUE) {
                gap = Math.max(bucket.get(0) - prevMax, gap);
            }
            if (bucket.size() != 0) {
                prevMax = bucket.get(bucket.size() - 1);
            }
        }
        return gap;
    }

    // 桶排序
    public List<List<Integer>> bucketSort(int[] nums, int min, int max) {
        if (nums == null)
            return null;
        int n = nums.length;
        // 向上取整
        int per = (int) Math.ceil((double) (max - min) / (n - 1));
        int bucketNum = (int) Math.ceil((double) (max - min) / per + 1);
        List<List<Integer>> buckets = new ArrayList<>();
        for (int i = 0; i < bucketNum; i++) {
            buckets.add(new ArrayList<>());
        }
        for (int num : nums) {
            buckets.get(index(num, per, min)).add(num);
        }
        for (List<Integer> bucket : buckets) {
            Collections.sort(bucket);
        }
        return buckets;
    }

    public int index(int a, int per, int min) {
        return (a - min) / per;
    }

    public int maximumGap5(int[] nums) {
        if (nums.length < 2) return 0;
        int max_value = Arrays.stream(nums).max().getAsInt(), min_value = Arrays.stream(nums).min().getAsInt();
        if (max_value == min_value) return 0;
        int bucket_len = Math.max(1, (max_value - min_value) / (nums.length - 1));
        int bucket_num = (max_value - min_value) / bucket_len + 1;
        List<List<Integer>> buckets = new ArrayList<>();
        for (int i = 0; i < bucket_num; i++) buckets.add(new ArrayList<>());
        for (int n : nums) {
            buckets.get((n - min_value) / bucket_len).add(n);
        }
        int res = 0, pre_max = Integer.MAX_VALUE;
        for (List<Integer> bucket : buckets) {
            if (bucket.size() != 0) {
                if (pre_max != Integer.MAX_VALUE) {
                    res = Math.max(res, bucket.stream().min((o1, o2) -> o1 - o2).get() - pre_max);
                }
                pre_max = bucket.stream().max((o1, o2) -> o1 - o2).get();
            }
        }
        return res;
    }

    public boolean canBeIncreasing2(int[] nums) {
        int abnormal = -1, count = 0;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] < nums[i - 1]) {
                count++;
                abnormal = i;
            }
            if (count > 1) return false;
        }
        if (abnormal == 1) return true;
        if (abnormal > 1 && nums[abnormal - 2] < nums[abnormal]) return true;
        return false;
    }

    public boolean canBeIncreasing(int[] nums) {
        int abnormal = -1, count = 0, pre = -1;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] < nums[i - 1]) {
                count++;
                abnormal = i;
            }
            if (count > 1) return false;
        }
        if (abnormal == -1) return true;
        for (int i = 1; i < nums.length; i++) {
            if (i == abnormal - 1) continue;
            if (pre != -1 && nums[i] < nums[pre]) return false;
            pre = i;
        }
        return true;
    }

    public void rotate(int[][] matrix) {
        int rowlen = matrix.length, collen = matrix[0].length;
        int[][] res = new int[rowlen][collen];
        for (int i = 0; i < rowlen; i++) {
            for (int j = 0; j < collen; j++) {
                res[j][rowlen - 1 - i] = matrix[i][j];
            }
        }
        for (int i = 0; i < rowlen; i++) {
            for (int j = 0; j < collen; j++) {
                matrix[i][j] = res[i][j];
            }
        }
        return;
    }

    public boolean isUnique(String astr) {
        int mark = 0;
        for (int i = 0; i < astr.length(); i++) {
            int movebit = astr.charAt(i) - 'a';
            int aftermove = (1 << movebit);
            if ((mark & aftermove) > 0) return false;
            else
                mark = (mark | aftermove);
        }
        return true;
    }

    public boolean CheckPermutation(String s1, String s2) {
        int[] array = new int[58];
        if (s1.length() != s2.length()) return false;
        for (int i = 0; i < s1.length(); i++) {
            array[s1.charAt(i) - 'A']++;
        }
        for (int i = 0; i < s2.length(); i++) {
            array[s2.charAt(i) - 'A']--;
        }
        for (int i = 0; i < 58; i++) {
            if (array[i] != 0) return false;
        }
        return true;

    }

    public String replaceSpaces(String S, int length) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < S.length() && i <= length; i++) {
            if (S.charAt(i) == ' ') {
                sb.append("%20");
            } else {
                sb.append(S.charAt(i));
            }
        }
        return sb.toString().substring(length);
    }

    public boolean canPermutePalindrome(String s) {
        Map<Character, Integer> array = new HashMap<>();
        for (int i = 0; i < s.length(); i++) {
            array.put(s.charAt(i), array.getOrDefault(s.charAt(i), 0) + 1);
        }
        int count = 0;
        for (Map.Entry<Character, Integer> entry : array.entrySet()) {
            if (entry.getValue() % 2 != 0) {
                count++;
            }
            if (count > 1) return false;
        }
        return true;
    }

    public int minDistance(String word1, String word2) {
        int flen = word1.length(), slen = word2.length();
        int[][] dp = new int[flen + 1][slen + 1];
        for (int i = 1; i <= flen; i++) {
            dp[i][0] = dp[i - 1][0] + 1;
        }
        for (int i = 1; i <= slen; i++) {
            dp[0][i] = dp[0][i - 1] + 1;
        }
        for (int i = 1; i <= flen; i++) {
            for (int j = 1; j <= slen; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.min(Math.min(dp[i - 1][j], dp[i - 1][j - 1]), dp[i][j - 1]) + 1;
                }
            }
        }
        return dp[flen][slen];
    }

    public boolean oneEditAway(String first, String second) {
        int flen = first.length(), slen = second.length();
        int[][] dp = new int[flen + 1][slen + 1];
        for (int i = 1; i <= flen; i++) {
            dp[i][0] = dp[i - 1][0] + 1;
        }
        for (int i = 1; i <= slen; i++) {
            dp[0][i] = dp[0][i - 1] + 1;
        }
//        dp意义最少经过多少次edit,这个是会看string 顺序的， 注意 ab， bc 是false
        for (int i = 1; i <= flen; i++) {
            for (int j = 1; j <= slen; j++) {
                if (first.charAt(i - 1) == second.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.min(Math.min(dp[i - 1][j], dp[i - 1][j - 1]), dp[i][j - 1]) + 1;
                }
            }
        }
        return dp[flen][slen] <= 1;
    }

    public boolean oneEditAway2(String first, String second) {
        int flen = first.length(), slen = second.length();
        int[][] dp = new int[flen + 1][slen + 1];
        for (int i = 1; i <= flen; i++) {
            for (int j = 1; j <= slen; j++) {

                if (first.charAt(i - 1) == second.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        if (flen == slen) return dp[flen][slen] >= slen - 1;
        else {
            return flen > slen ? dp[flen][slen] >= flen - 1 : dp[flen][slen] >= slen - 1;
        }
    }

    public int maxSubArray(int[] nums) {
        int pre = 0, cur = 0, min = Arrays.stream(nums).min().getAsInt(), max = min, maxNum = Arrays.stream(nums).max().getAsInt();
        if (maxNum <= 0) return min;
        for (int i = 0; i < nums.length; i++) {
            cur = Math.max(pre + nums[i], 0);
            pre = cur;
            max = Math.max(max, pre);
        }
        return max;
    }

    public int triangleNumber(int[] nums) {
        Arrays.sort(nums);
        int count = 0;
        for (int i = 0; i < nums.length - 2; i++) {

            for (int j = i + 1; j < nums.length - 1; j++) {
                int l = j + 1, r = nums.length - 1;
                int k = binarySearch(nums, l, r, nums[i] + nums[j]);
                if (nums[i] + nums[j] > nums[k]) {
                    count += k - l + 1;
                }
            }
        }
        return count;
    }

    private int binarySearch(int[] nums, int l, int r, int target) {
        while (l < r) {
            int mid = (r - l + 1) / 2 + l;
            if (nums[mid] >= target) {
                r = mid - 1;
            } else {
                l = mid;
            }
        }
        return l;
    }

    public int triangleNumber2(int[] nums) {
        Arrays.sort(nums);
        int count = 0;
        for (int i = 0; i < nums.length - 2; i++) {
            for (int j = i + 1; j < nums.length - 1; j++) {
                for (int k = j + 1; k < nums.length; k++) {
                    if (nums[i] + nums[j] < nums[k]) {
                        break;
                    }
                    if (nums[k] - nums[i] >= nums[j]) {
                        break;
                    }
                    count++;
                }
            }
        }
        return count;
    }

    public String compressString(String S) {
        if (S.length() == 0) return S;
        StringBuilder sb = new StringBuilder();
        int l = 0, r = 0;
        while (r < S.length()) {
            if (S.charAt(l) == S.charAt(r)) {
                r++;
            } else {
                sb.append(S.charAt(l));
                sb.append((r - l));
                l = r;
            }
        }
        sb.append(S.charAt(l));
        sb.append((r - l));
        if (sb.length() >= S.length()) return S;
        return sb.toString();
    }

    public void setZeroes(int[][] matrix) {
        int rowlen = matrix.length, collen = matrix[0].length;
        int[] row = new int[rowlen], col = new int[collen];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                if (matrix[i][j] == 0) {
                    row[i] = 1;
                    col[j] = 1;
                }
            }
        }
        for (int i = 0; i < rowlen; i++) {
            if (row[i] == 1) {
                Arrays.fill(matrix[i], 0);
            }
        }
        for (int i = 0; i < collen; i++) {
            if (col[i] == 1) {
                for (int j = 0; j < rowlen; j++) {
                    matrix[j][i] = 0;
                }
            }
        }

    }

    public boolean isFlipedString2(String s1, String s2) {
        if (s1.length() != s2.length()) return false;
        String tar = s2 + s2;
        if (tar.contains(s1)) return true;
        else return false;
    }

    public boolean isFlipedString(String s1, String s2) {
        if (s1.length() != s2.length()) return false;
        int r = 1;
        while (r < s1.length()) {
            if (s2.contains(s1.substring(0, r))) {
                r++;
            }
        }
        String first = s1.substring(0, r - 1), sec = s1.substring(r - 1, s1.length());
        if (s2.equals(first + sec) || s2.equals(sec + first)) {
            return true;
        } else return false;
    }

    public ListNode removeDuplicateNodes2(ListNode head) {
        Set<Integer> set = new HashSet<>();
        ListNode dum = new ListNode();
        dum.next = head;
        ListNode pre = dum;
        while (head != null) {
            if (!set.contains(head.val)) {
                set.add(head.val);
                pre.next = head;
                pre = pre.next;
            }
            head = head.next;
        }
        pre.next = null;
        return dum.next;
    }

    public ListNode removeDuplicateNodes(ListNode head) {
        ListNode cur = head;
        while (cur != null) {
            ListNode sec = cur;
            while (sec.next != null && sec.next.val == sec.val) {
                sec.next = sec.next.next;
//                if(){
//
//                }else{
//
//                }
            }
            cur = cur.next;
        }
        return head;
    }

    public int kthToLast(ListNode head, int k) {
        ListNode first = head, sec = head;
        while (k > 0) {
            first = first.next;
            k--;
        }
        while (first != null) {
            first = first.next;
            sec = sec.next;
        }
        return sec.val;
    }

    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        int carry = 0;
        ListNode dum = new ListNode(-1), pre = dum;
        dum.next = l1;
        boolean changed = false;
        while (l1 != null || l2 != null) {
            int v1 = l1 != null ? l1.val : 0;
            int v2 = l2 != null ? l2.val : 0;
            int sum = v1 + v2 + carry;
            if (l1 != null) {
                l1.val = sum % 10;
                l1 = l1.next;
            } else {
                if (!changed) {
                    pre.next = l2;
                    changed = true;
                }
            }
            if (l2 != null) {
                if (changed) l2.val = sum % 10;
                l2 = l2.next;
            }
            carry = sum / 10;
            pre = pre.next;
        }
        if (carry != 0) {
            pre.next = new ListNode(carry);
        }
        return dum.next;
    }

    ListNode front;

    public boolean isPalindrome2(ListNode head) {
        front = head;
        return recursive(head.next);
    }

    private boolean recursive(ListNode root) {
        if (root != null) {
            if (!recursive(root.next)) {
                return false;
            }
            if (root.val != front.val) {
                return false;
            }
            front = front.next;
        }
        return true;
    }

    public boolean isPalindrome(ListNode head) {
        if (head == null) return true;
        ListNode firstEnd = findEnd(head);
        ListNode reversed = reverse(firstEnd.next);
        while (head != null && reversed != null) {
            if (head.val != reversed.val) return false;
            head = head.next;
            reversed = reversed.next;
        }
        return true;
    }

    private ListNode reverse(ListNode root) {
        ListNode dum = new ListNode();
        while (root != null) {
            ListNode next = root.next;
            root.next = dum.next;
            dum.next = root;
            root = next;
        }
        return dum.next;
    }

    private ListNode findEnd(ListNode head) {
        ListNode fast = head, slow = head;
//        if the condition is .next.next and .next, we choose the first one when the number is even
//        however, if the condition is .next and fast, we choose the second one when the number is even
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        return slow;
    }

    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null && headB == null) return null;
        if (headA == null || headB == null) return null;
        ListNode a = headA, b = headB;
        while (a != null || b != null) {
            a = a == null ? headB : a.next;
            b = b == null ? headA : b.next;
            if (a == b) return a;
        }
        return null;
    }

    public ListNode detectCycle(ListNode head) {
        if (head == null) return head;
        ListNode fast = head, slow = head;
        while (fast != null || slow != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (fast == slow) return slow;
        }
        return null;
    }

    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, Integer> map = new HashMap<>();
        List<List<String>> res = new ArrayList<>();
        int count =0;
        for (int i = 0; i < strs.length; i++) {
            char[] tmp = strs[i].toCharArray();
            Arrays.sort(tmp);
            String tmpString = Arrays.toString(tmp);
            if(map.containsKey(tmpString)){
                res.get(map.get(tmpString)).add(strs[i]);
            }else{
                map.put(tmpString,count);
                res.add(new ArrayList<>(Arrays.asList(strs[i])));
                count++;
            }
        }
        return res;
    }
//
//    public ListNode sortList(ListNode head) {
//        if (head == null) return head;
//
//    }

    public static void main(String[] args) {
        Lc9 lc9 = new Lc9();
//        lc9.numDistinct("abb","ab");
//        for (int i = 0; i < 20; i++) {
//            int res = lc9.findKthLargest(new int[]{3, 1, 2, 4}, 2);
//            System.out.println(res);
//        }
        int[][] a = new int[][]{{1, 3}, {6, 9}};
        int[] b = new int[]{2, 5};
//        lc9.insert(a, b);
//        char[] tmp = String.valueOf(10).toCharArray();
//
//        System.out.println(tmp[0]);
//        StringBuilder sb = new StringBuilder();
//        sb.append("00");
//        double c = Double.parseDouble(sb.toString());
//        System.out.println((int) c);
//        boolean res = lc9.canBeIncreasing(new int[]{1, 2, 10, 5, 7});
//        System.out.println(res);
//        FizzBuzz fb = new FizzBuzz(10);
//        new Thread(fb).start();
//        System.out.println('z' - 'A' + 1);
//        System.out.println((int) 'z');
//        System.out.println((int) 'A');


        lc9.canPermutePalindrome("aab");

    }
}

class Solution3 {
    Random random = new Random();

    public int findKthLargest(int[] nums, int k) {
        return quickSelect(nums, 0, nums.length - 1, nums.length - k);
    }

    public int quickSelect(int[] a, int l, int r, int index) {
        int q = randomPartition(a, l, r);
        if (q == index) {
            return a[q];
        } else {
            return q < index ? quickSelect(a, q + 1, r, index) : quickSelect(a, l, q - 1, index);
        }
    }

    public int randomPartition(int[] a, int l, int r) {
        int i = random.nextInt(r - l + 1) + l;
        swap(a, i, r);
        return partition(a, l, r);
    }

    public int partition(int[] a, int l, int r) {
        int x = a[r], i = l - 1;
        for (int j = l; j < r; ++j) {
            if (a[j] <= x) {
                swap(a, ++i, j);
            }
        }
        swap(a, i + 1, r);
        return i + 1;
    }

    public void swap(int[] a, int i, int j) {
        int temp = a[i];
        a[i] = a[j];
        a[j] = temp;
    }
}
//class FizzBuzz {
//    private int n;
//    Semaphore isBuzz = new Semaphore(0);
//    Semaphore isFuzz = new Semaphore(0);
//    Semaphore isBuzzFuzz = new Semaphore(0);
//    Semaphore isother = new Semaphore(0);
//    int count = 1;
//    public FizzBuzz(int n) {
//        this.n = n;
//    }
//
//    // printFizz.run() outputs "fizz".
//    public void fizz(Runnable printFizz) throws InterruptedException {
//        isFuzz.acquire();
//        while(count<=n){
//            printFizz.run();
//            isother.release();
//        }
//    }
//
//    // printBuzz.run() outputs "buzz".
//    public void buzz(Runnable printBuzz) throws InterruptedException {
//        isBuzz.acquire();
//        while(count<=n){
//            printBuzz.run();
//            isother.release();
//        }
//    }
//
//    // printFizzBuzz.run() outputs "fizzbuzz".
//    public void fizzbuzz(Runnable printFizzBuzz) throws InterruptedException {
//        isBuzzFuzz.acquire();
//        while(count<=n){
//            printFizzBuzz.run();
//            isother.release();
//        }
//    }
//
//    // printNumber.accept(x) outputs "x", where x is an integer.
//    public void number(IntConsumer printNumber) throws InterruptedException {
//        while(count<=n){
//            if(count%3==0&&count%5==0){
//                isBuzzFuzz.release();
//            }else if(count%3==0){
//                isFuzz.release();
//            }else if(count%5==0){
//                isBuzz.release();
//            }else{
//                printNumber.accept(count);
//                isother.release();
//            }
//            isother.acquire();
//            count++;
//        }
//        isBuzz.release();
//        isFuzz.release();
//        isBuzzFuzz.release();
//    }
//}

//class FizzBuzz {
//    private int n;
//    volatile int num = 1;
//
//    public FizzBuzz(int n) {
//        this.n = n;
//    }
//
//    // printFizz.run() outputs "fizz".
//    public void fizz(Runnable printFizz) throws InterruptedException {
//        while (num <= n) {
//            if (num % 3 == 0 && num % 5 != 0) {
//                printFizz.run();
//                num += 1;
//            } else {
//                Thread.yield();
//            }
//        }
//    }
//
//    // printBuzz.run() outputs "buzz".
//    public void buzz(Runnable printBuzz) throws InterruptedException {
//        while (num <= n) {
//            if (num % 3 != 0 && num % 5 == 0) {
//                printBuzz.run();
//                num += 1;
//            } else {
//                Thread.yield();
//            }
//        }
//    }
//
//
//    // printFizzBuzz.run() outputs "fizzbuzz".
//    public void fizzbuzz(Runnable printFizzBuzz) throws InterruptedException {
//        while (num <= n) {
//            if (num % 3 == 0 && num % 5 == 0) {
//                printFizzBuzz.run();
//                num += 1;
//            } else {
//                Thread.yield();
//            }
//        }
//    }
//
//    // printNumber.accept(x) outputs "x", where x is an integer.
//    public void number(IntConsumer printNumber) throws InterruptedException {
//        while (num <= n) {
//            if (num % 3 != 0 && num % 5 != 0) {
//                printNumber.accept(num);
//                num += 1;
//            } else {
//                Thread.yield();
//            }
//        }
//    }
//}


//class FizzBuzz {
//    private int n;
//    private int i=1;
//
//    public FizzBuzz(int n) {
//        this.n = n;
//    }
//
//    // printFizz.run() outputs "fizz".
//    public void fizz(Runnable printFizz) throws InterruptedException {
//        synchronized(this){
//            while(i<=n){
//                if(i%3==0&&i%5!=0){
//                    printFizz.run();
//                    i++;
//                    this.notifyAll();
//                }
//                else{
//                    this.wait();
//                }
//
//            }
//        }
//    }
//
//    // printBuzz.run() outputs "buzz".
//    public void buzz(Runnable printBuzz) throws InterruptedException {
//        synchronized(this){
//            while(i<=n){
//                if(i%5==0&&i%3!=0){
//                    printBuzz.run();
//                    i++;
//                    this.notifyAll();
//                }
//                else{
//                    this.wait();
//                }
//
//            }
//        }
//    }
//
//    // printFizzBuzz.run() outputs "fizzbuzz".
//    public void fizzbuzz(Runnable printFizzBuzz) throws InterruptedException {
//        synchronized(this){
//            while(i<=n){
//                if(i%3==0&&i%5==0){
//                    printFizzBuzz.run();
//                    i++;
//                    this.notifyAll();
//                }
//                else{
//                    this.wait();
//                }
//
//            }
//        }
//    }
//
//    // printNumber.accept(x) outputs "x", where x is an integer.
//    public void number(IntConsumer printNumber) throws InterruptedException {
//        synchronized(this){
//            while(i<=n){
//                if(i%3!=0&&i%5!=0){
//                    printNumber.accept(i);
//                    i++;
//                    this.notifyAll();
//                }
//                else{
//                    this.wait();
//                }
//
//            }
//        }
//    }
//}

class FizzBuzz {
    private int n;
    private int i = 1;
    private Lock lock = new ReentrantLock();
    private Condition condition = lock.newCondition();

    public FizzBuzz(int n) {
        this.n = n;
    }

    // printFizz.run() outputs "fizz".
    public void fizz(Runnable printFizz) throws InterruptedException {
        while (i <= n) {
            lock.lock();
            if (i % 3 == 0 && i % 5 != 0) {
                printFizz.run();
                i++;
                condition.signalAll();
            } else {
                condition.await();
            }
            lock.unlock();
        }

    }

    // printBuzz.run() outputs "buzz".
    public void buzz(Runnable printBuzz) throws InterruptedException {
        while (i <= n) {
            lock.lock();
            if (i % 5 == 0 && i % 3 != 0) {
                printBuzz.run();
                i++;
                condition.signalAll();
            } else {
                condition.await();
            }
            lock.unlock();
        }
    }

    // printFizzBuzz.run() outputs "fizzbuzz".
    public void fizzbuzz(Runnable printFizzBuzz) throws InterruptedException {
        while (i <= n) {
            lock.lock();
            if (i % 3 == 0 && i % 5 == 0) {
                printFizzBuzz.run();
                i++;
                condition.signalAll();
            } else {
                condition.await();
            }
            lock.unlock();
        }
    }

    // printNumber.accept(x) outputs "x", where x is an integer.
    public void number(IntConsumer printNumber) throws InterruptedException {
        while (i <= n) {
            lock.lock();
            if (i % 3 != 0 && i % 5 != 0) {
                printNumber.accept(i);
                i++;
                condition.signalAll();
            } else {
                condition.await();
            }
            lock.unlock();
        }
    }
}

//class FizzBuzz {
//    private int n;
//    private int i=1;
//    private Lock lock=new ReentrantLock();
//
//    public FizzBuzz(int n) {
//        this.n = n;
//    }
//
//    // printFizz.run() outputs "fizz".
//    public void fizz(Runnable printFizz) throws InterruptedException {
//        while(i<=n){
//            lock.lock();
//            if(i%3==0&&i%5!=0){
//                printFizz.run();
//                i++;
//            }
//            lock.unlock();
//        }
//
//    }
//
//    // printBuzz.run() outputs "buzz".
//    public void buzz(Runnable printBuzz) throws InterruptedException {
//        while(i<=n){
//            lock.lock();
//            if(i%5==0&&i%3!=0){
//                printBuzz.run();
//                i++;
//            }
//            lock.unlock();
//        }
//    }
//
//    // printFizzBuzz.run() outputs "fizzbuzz".
//    public void fizzbuzz(Runnable printFizzBuzz) throws InterruptedException {
//        while(i<=n){
//            lock.lock();
//            if(i%3==0&&i%5==0){
//                printFizzBuzz.run();
//                i++;
//            }
//            lock.unlock();
//        }
//    }
//
//    // printNumber.accept(x) outputs "x", where x is an integer.
//    public void number(IntConsumer printNumber) throws InterruptedException {
//        while(i<=n){
//            lock.lock();
//            if(i%3!=0&&i%5!=0){
//                printNumber.accept(i);
//                i++;
//            }
//            lock.unlock();
//        }
//    }
//}

class MinStack {
    Stack<Integer> min = new Stack<>();
    Stack<Integer> stack = new Stack<>();
    /** initialize your data structure here. */
    public MinStack() {
        min.push(Integer.MAX_VALUE);
    }

    public void push(int x) {
        stack.push(x);
        if(x<min.peek()){
            min.push(x);
        }else{
            min.push(min.peek());
        }
    }

    public void pop() {
        stack.pop();
        min.pop();
    }

    public int top() {
        return stack.peek();
    }

    public int getMin() {
        return min.peek();
    }
}

class AnimalShelf {
    LinkedList<int[]> linkedList=new LinkedList<>();

    public AnimalShelf() {

    }

    public void enqueue(int[] animal) {
        linkedList.add(animal);
    }

    public int[] dequeueAny() {
        if(linkedList.size() !=0){
            int[] temp = linkedList.getFirst();
            linkedList.remove(temp);
            return temp;
        }
        return new int[]{-1,-1};
    }

    public int[] dequeueDog() {
        for(int i=0;i<linkedList.size();i++){
            int[]  temp = linkedList.get(i);
            if(temp[1] == 1){
                linkedList.remove(temp);
                return temp;
            }
        }
        return new int[]{-1,-1};
    }

    public int[] dequeueCat() {
        for(int i=0;i<linkedList.size();i++){
            int[]  temp = linkedList.get(i);
            if(temp[1] == 0){
                linkedList.remove(temp);
                return temp;
            }
        }
        return new int[]{-1,-1};
    }
}

/**
 * Your AnimalShelf object will be instantiated and called as such:
 * AnimalShelf obj = new AnimalShelf();
 * obj.enqueue(animal);
 * int[] param_2 = obj.dequeueAny();
 * int[] param_3 = obj.dequeueDog();
 * int[] param_4 = obj.dequeueCat();
 */
