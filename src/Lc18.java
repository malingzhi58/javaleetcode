import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.HashMap;
import java.util.*;

public class Lc18 {
    public int kthLargestValue(int[][] matrix, int k) {
        int row = matrix.length;
        int col = matrix[0].length;
        int[] nums = new int[row * col];
        nums[0] = matrix[0][0];

        for (int i = 1; i < col; i++) {
            nums[i] = nums[i - 1] ^ matrix[0][i];
        }
        for (int i = 1; i < row; i++) {
            nums[col * i] = nums[col * (i - 1)] ^ matrix[i][0];
        }

        for (int i = 1; i < row; i++) {
            for (int j = 1; j < col; j++) {
                nums[i * col + j] = nums[i * col + j - 1] ^ nums[(i - 1) * col + j] ^
                        nums[(i - 1) * col + j - 1] ^ matrix[i][j];
            }
        }
        return findKthLargest(nums, k);
    }

    private int findKthLargest(int[] nums, int k) {
        int left = 0, right = nums.length - 1, len = nums.length;
        return findKth(nums, left, right, len - k);
    }

    private int findKth(int[] nums, int left, int right, int k) {
        int mid = quickSelect(nums, left, right);
        if (mid == k) {
            return nums[mid];
        } else if (mid < k) {
            return findKth(nums, mid + 1, right, k);
        } else {
            return findKth(nums, left, mid - 1, k);
        }
    }

    private int quickSelect(int[] nums, int left, int right) {
        Random random = new Random();
        int pivot = random.nextInt(right - left + 1) + left;
        swap(nums, pivot, left);
        int i = left, j = right, x = nums[left];
        while (i < j) {
            while (i < j && nums[j] >= x) j--;
            while (i < j && nums[i] <= x) i++;
            swap(nums, i, j);
            System.out.println(i + ':' + j);
        }
        swap(nums, left, i);
        return i;
    }

    public void swap(int[] arr, int left, int right) {
        int temp = arr[left];
        arr[left] = arr[right];
        arr[right] = temp;
    }

    public int findKthLargest2(int[] nums, int k) {
        int len = nums.length;
        return findKth3(nums, 0, len - 1, len - k);
    }

    private int findKth3(int[] nums, int left, int right, int k) {
        int mid = quickSelect(nums, left, right);
        if (mid == k) return nums[k];
        else if (mid > k) return findKth3(nums, left, mid - 1, k);
        else return findKth3(nums, mid + 1, right, k);
    }

    public int[] smallestK(int[] arr, int k) {
        if (k == 0) return new int[]{};
        return findKth2(arr, 0, arr.length - 1, k - 1);
    }

    private int[] findKth2(int[] arr, int left, int right, int k) {
        int mid = quickSelect(arr, left, right);
        if (mid == k) {
            quickSort(arr, 0, k - 1);
            return Arrays.copyOfRange(arr, 0, k + 1);
        } else if (mid > k) {
            return findKth2(arr, left, mid - 1, k);
        } else {
            return findKth2(arr, mid + 1, right, k);

        }
    }

    private void quickSort(int[] arr, int left, int right) {
        if (left >= right) return;
        int i = left, j = right, x = arr[left];
        while (i < j) {
            while (i < j && arr[j] >= x) j--;
            while (i < j && arr[i] <= x) i++;
            swap(arr, i, j);
        }
        swap(arr, left, i);
        quickSort(arr, left, i - 1);
        quickSort(arr, i + 1, right);
    }

    public int[] xorQueries(int[] arr, int[][] queries) {
        Map<Integer, Integer> map = new HashMap<>();
        int sum = 0;
        map.put(0, 0);
        for (int i = 0; i < arr.length; i++) {
            sum ^= arr[i];
            map.put(i + 1, sum);
        }
        int[] res = new int[queries.length];
        for (int i = 0; i < queries.length; i++) {
            res[i] = map.get(queries[i][1] + 1) ^ map.get(queries[i][0]);
        }
        return res;
    }

    public int[] topKFrequent(int[] nums, int k) {
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> b[1] - a[1]);
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            map.put(nums[i], map.getOrDefault(nums[i], 0) + 1);
        }
        for (Integer each : map.keySet()) {
            pq.offer(new int[]{each, map.get(each)});
        }
        int[] res = new int[k];
        int id = 0;
        while (!pq.isEmpty()) {
            res[id++] = pq.poll()[0];
            if (id == k) break;
        }

        return res;
    }

    public int checkRecord(int n) {
        long[][][] dp = new long[n + 1][2][3];
        int MOD = (int) 1e9 + 7;
        dp[0][0][0] = 1;
        dp[0][0][1] = 1;
        dp[0][1][0] = 1;
        for (int i = 1; i < n; i++) {
            dp[i][0][0] = (dp[i - 1][0][0] + dp[i - 1][0][1] + dp[i - 1][0][2]) % MOD;
            dp[i][1][0] = (dp[i - 1][0][0] + dp[i - 1][0][1] + dp[i - 1][0][2] + dp[i - 1][1][0] + dp[i - 1][1][1] + dp[i - 1][1][2]) % MOD;
            dp[i][0][1] = dp[i - 1][0][0];
            dp[i][0][2] = dp[i - 1][0][1];
            dp[i][1][1] = dp[i - 1][1][0];
            dp[i][1][2] = dp[i - 1][1][1];
        }
        long res = 0;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                res += dp[n - 1][i][j];
                res = res % MOD;
            }
        }
        return (int) res;
    }

    public int maxArea(int h, int w, int[] horizontalCuts, int[] verticalCuts) {
        int verPre = 0, verCur = 0, horPre = 0, horCur = 0;
        int MOD = (int) 1e9 + 7;
        long max = 0;
        Arrays.sort(horizontalCuts);
        Arrays.sort(verticalCuts);
        int[] newHor = new int[horizontalCuts.length + 1];
        int[] newVer = new int[verticalCuts.length + 1];
        for (int i = 0; i < horizontalCuts.length; i++) {
            newHor[i] = horizontalCuts[i];
        }
        newHor[horizontalCuts.length] = h;
        for (int i = 0; i < verticalCuts.length; i++) {
            newVer[i] = verticalCuts[i];
        }
        newVer[verticalCuts.length] = w;
        for (int i = 0; i < newVer.length; i++) {
            verCur = newVer[i];
            horPre = 0;
            for (int j = 0; j < newHor.length; j++) {
                horCur = newHor[j];
                long tmp = ((long) (verCur - verPre)) * ((long) (horCur - horPre));
                tmp = tmp % MOD;
                max = Math.max(max, tmp);
                horPre = horCur;
            }
            verPre = verCur;
        }
//        int tmp = (w-verPre)*(h-horPre);
//        max = Math.max(max,tmp);
        return (int) max;
    }

    public int maxArea2(int h, int w, int[] horizontalCuts, int[] verticalCuts) {
        Arrays.sort(horizontalCuts);
        Arrays.sort(verticalCuts);
        int lenx = horizontalCuts.length;
        int leny = verticalCuts.length;
        long maxX = horizontalCuts[0];
        long maxY = verticalCuts[0];
        for (int i = 1; i < lenx; i++) {
            maxX = Math.max(horizontalCuts[i] - horizontalCuts[i - 1], maxX);
        }
        maxX = Math.max(maxX, h - horizontalCuts[lenx - 1]);
        for (int i = 1; i < leny; i++) {
            maxY = Math.max(verticalCuts[i] - verticalCuts[i - 1], maxY);
        }
        maxY = Math.max(maxY, w - verticalCuts[leny - 1]);
        return (int) ((maxX * maxY) % (1000000007));
    }

    public int numMatchingSubseq(String s, String[] words) {
        int res = 0;
        Set<String> set = new HashSet<>();
        Set<String> falseset = new HashSet<>();
        for (String each : words) {
            if (set.contains(each)) {
                res++;
                continue;
            }
            if (falseset.contains(each)) {
                continue;
            }
            int id1 = 0, id2 = 0;
            while (id1 < s.length() && id2 < each.length()) {
                if (s.charAt(id1) != each.charAt(id2)) {
                    id1++;
                } else {
                    id1++;
                    id2++;
                }
            }
            if (id2 == each.length()) {
                set.add(each);
                res++;
            } else {
                falseset.add(each);
            }
        }
        return res;
    }
    public List<Boolean> camelMatch(String[] queries, String pattern) {
        List<Boolean> res = new ArrayList<>();
        StringBuilder p2 =new StringBuilder();
        for (int i = 0; i < pattern.length(); i++) {
            if(pattern.charAt(i)<='Z'&&pattern.charAt(i)>='A'){
                p2.append(pattern.charAt(i));
            }
        }
        for(String each:queries){
            StringBuilder sb = new StringBuilder();
            boolean ans = true;
            for (int i = 0; i < each.length(); i++) {
                if(each.charAt(i)>='A'&&each.charAt(i)<='Z'){
                    sb.append(each.charAt(i));
                    if(sb.length()>p2.length()){
                        ans = false;
                        break;
                    }
                }
            }
            if(!p2.toString().equals(sb.toString())){
                ans = false;
            }
            res.add(ans);
        }
        return res;
    }
    int[] res2;
    int dNum;
    int res3=0x3f3f3f3f;
    public int minDifficulty(int[] jobDifficulty, int d) {
        if(jobDifficulty.length<d) return -1;

        res2 = new int[d];
        dNum = d;
//        int INF = 0x3f3f3f3f;
//        for (int i = 0; i < d; i++) {
//            res2[i]=INF;
//        }
        dfs(jobDifficulty,0,d);

        return res3;
    }

    private void dfs(int[] jobDifficulty, int start, int left) {
        if(left==1){
            for (int i = start; i <jobDifficulty.length ; i++) {
                res2[dNum-left]=Math.max(res2[dNum-left],jobDifficulty[i]);
            }
            int sum =0;
            for (int i = 0; i < dNum; i++) {
                sum+=res2[i];
            }
            res3= Math.min(res3,sum);
            return;
        }

        for (int i = start; i <= jobDifficulty.length-left; i++) {
            res2[dNum-left]=Math.max(res2[dNum-left],jobDifficulty[i]);
            dfs(jobDifficulty, i+1, left-1);
            res2[dNum-left+1]=0;
        }
    }

    public static void main(String[] args) {
        Lc18 lc18 = new Lc18();
//        lc18.kthLargestValue()
        int[] s1 = {3, 1, 2, 5, 4};
//        int r1 =lc18.findKthLargest(s1,2);
//        System.out.println(r1);
        int[] s2 = {1, 3, 5, 7, 2, 4, 6, 8};
//        int[] r2 = lc18.smallestK(s2,4);
        int[] s3 = {1, 1, 1, 2, 2, 3};
//        lc18.topKFrequent(s3,2);
        int[] s4 = {2};
        int[] s5 = {2};

//        int r4 = lc18.maxArea(1000000000, 1000000000, s4, s5);
//        System.out.println(r4);
//        lc18.maxArea2(1000000000, 1000000000, s4, s5);
//        lc18.maxArea(5, 4, s4, s5);

        TrieS6 trieS6 =new TrieS6();
        String[] s6 ={"time","me","bell"};
//        trieS6.minimumLengthEncoding(s6);

        int[] s7 ={6,5,4,3,2,1};
        lc18.minDifficulty(s7,2);
    }
}

class TrieS6 {
    class Trie {
        Trie[] children = new Trie[26];
        String word = "";
    }

    Trie root = new Trie();

    void insert(String word) {
        Trie cur = root;
        StringBuilder sb =new StringBuilder(word);
        String w2 = sb.reverse().toString();
        char[] arr = w2.toCharArray();
        for (int i = 0; i < arr.length; i++) {
            if (cur.children[arr[i] - 'a'] == null) {
                cur.children[arr[i] - 'a'] = new Trie();
            }
            cur = cur.children[arr[i] - 'a'];
        }
        for (int i = 0; i < 26; i++) {
            if(cur.children[i]!=null){
                return;
            }
        }
        if (w2.length() > cur.word.length()) {
            cur.word = word;
        }
    }

    List<String> res = new ArrayList<>();
    public int minimumLengthEncoding(String[] words) {
        Arrays.sort(words,(a,b)->b.length()-a.length());

        for(String each:words){
            insert(each);
        }
        dfs(root);
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < res.size(); i++) {
            sb.append(res.get(i));
            sb.append("#");
        }
        return sb.length();
    }

    private void dfs(Trie root) {
        for (int i = 0; i < 26; i++) {
            if(root.children[i]!=null){
                if(!root.children[i].word.equals("")){
                    res.add(root.children[i].word);
                }
                dfs(root.children[i]);
            }
        }
    }
}

class MagicDictionary {
    List<String> dic = new ArrayList<>();

    /**
     * Initialize your data structure here.
     */
    public MagicDictionary() {

    }

    public void buildDict(String[] dictionary) {
        for (String each : dictionary) {
            dic.add(each);
        }
    }

    public boolean search(String searchWord) {
        boolean find = false;
        for (String each : dic) {
            if (each.length() != searchWord.length() || each.equals(searchWord)) {
                continue;
            }
            for (int i = 0; i < each.length(); i++) {
                String a = each.substring(0, i) + each.substring(i + 1);
                String b = searchWord.substring(0, i) + searchWord.substring(i + 1);
                if (a.equals(b)) {
                    return true;
                }
            }
        }
        return find;
    }
}

