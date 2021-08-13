import java.util.*;

public class Lc14 {

    int curPos = 0;

    public boolean isRobotBounded(String instructions) {
        //turn right - 1, turn left +1
        int[][] directions = {{0, 1}, {-1, 0}, {0, -1}, {1, 0}};
        int[] cur = {0, 0};
        int n = 5;
        while (n-- >= 0) {
            cur = operate(cur, directions, instructions);
            if (cur[0] == 0 && cur[1] == 0) return true;
        }
        return false;
    }

    private int[] operate(int[] cur, int[][] directions, String instructions) {
        for (int i = 0; i < instructions.length(); i++) {
            if (instructions.charAt(i) == 'G') {
                cur[0] += directions[curPos][0];
                cur[1] += directions[curPos][1];
            }
            if (instructions.charAt(i) == 'L') {
                curPos += 1;
                curPos %= 4;
            }
            if (instructions.charAt(i) == 'R') {
                curPos -= 1;
                curPos += 4;
                curPos %= 4;
            }
        }
        return cur;
    }

    //    public List<Integer> eventualSafeNodes(int[][] graph) {
//        int len = graph.length;
//        //0 for not visiting, 1 for no valid, 2 for valid
//        int[] isLoop = new int[len];
//        for (int i = 0; i < len; i++) {
//            Set<Integer> vis = new HashSet<>();
//            if(isLoop[i]==0){
//                if(!dfs(graph,i,vis,isLoop)){
//                    //for loop set
//                    isLoop[i]=1;
//                }else{
//                    isLoop[i]=2;
//                }
//            }
//        }
//        List<Integer> res = new ArrayList<>();
//        for (int i = 0; i < len; i++) {
//            if(isLoop[i]==2) res.add(i);
//        }
//        return res;
//    }
//
//    private boolean dfs(int[][] graph, int start, Set<Integer> vis, int[] isLoop) {
//        if(vis.contains(start)){
//            return false;
//        }
//        if(isLoop[start]==2) return true;
//        if(isLoop[start]==1) return false;
//        vis.add(start);
//        for (int i = 0; i < graph[start].length; i++) {
//            if(graph[start][i]==start) {
//                vis.remove(start);
//                return false;
//            }
//            boolean res = dfs(graph,graph[start][i],vis, isLoop);
//            if(!res){
//                isLoop[graph[start][i]]=1;
//                vis.remove(start);
//                return false;
//            }else{
//                isLoop[graph[start][i]]=2;
//            }
//        }
//        vis.remove(start);
//        return true;
//    }
//    public List<Integer> eventualSafeNodes(int[][] graph) {
//        int n = graph.length;
//        List<List<Integer>> rg = new ArrayList<List<Integer>>();
//        for (int i = 0; i < n; ++i) {
//            rg.add(new ArrayList<Integer>());
//        }
//        int[] inDeg = new int[n];
//        for (int x = 0; x < n; ++x) {
//            for (int y : graph[x]) {
//                rg.get(y).add(x);
//            }
//            inDeg[x] = graph[x].length;
//        }
//
//        Queue<Integer> queue = new LinkedList<Integer>();
//        for (int i = 0; i < n; ++i) {
//            if (inDeg[i] == 0) {
//                queue.offer(i);
//            }
//        }
//        while (!queue.isEmpty()) {
//            int y = queue.poll();
//            for (int x : rg.get(y)) {
//                if (--inDeg[x] == 0) {
//                    queue.offer(x);
//                }
//            }
//        }
//
//        List<Integer> ans = new ArrayList<Integer>();
//        for (int i = 0; i < n; ++i) {
//            if (inDeg[i] == 0) {
//                ans.add(i);
//            }
//        }
//        return ans;
//    }
    public List<Integer> eventualSafeNodes(int[][] graph) {
        int n = graph.length;
        List<List<Integer>> inDegree = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            inDegree.add(new ArrayList<>());
        }
        int[] outDegree = new int[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < graph[i].length; j++) {
                inDegree.get(graph[i][j]).add(j);
            }
            outDegree[i] = graph[i].length;
        }
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            if (outDegree[i] == 0) {
                queue.offer(i);
            }
        }
        while (!queue.isEmpty()) {
            int cur = queue.poll();
            List<Integer> list = inDegree.get(cur);
            for (int i = 0; i < list.size(); i++) {
                int to = list.get(i);
                outDegree[to]--;
                if (outDegree[to] == 0) {
                    queue.offer(to);
                }
            }
        }
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (outDegree[i] == 0) res.add(i);
        }
        return res;
    }
//    public double[] medianSlidingWindow(int[] nums, int k) {
//        int[] window = new int[k];
//        int len =nums.length,idx=0;
//        double[] res = new double[len-k+1];
//        for (int i = 0; i < k; i++) {
//            window[i]=nums[i];
//        }
//        Arrays.sort(window);
//        res[idx++]=calMediean(window);
//        for (int i = k; i <len ; i++) {
//            int deleteIndex = findNum(window,nums[i-k]);
//            window[deleteIndex] = nums[i];
//            Arrays.sort(window);
//            res[idx++]=calMediean(window);
//        }
//        return res;
//    }
//
//    private int findNum(int[] window, int num) {
//        return Arrays.binarySearch(window,num);
//    }
//
//    private double calMediean(int[] window) {
//        if((window.length&1)==1){
//            return window[window.length/2];
//        }else{
//            return ((double) window[window.length / 2] + (double) window[window.length / 2 - 1]) /2;
//        }
//    }

    //    public double[] medianSlidingWindow(int[] nums, int k) {
//        int n = nums.length;
//        double[] ans = new double[n-k+1];
//        Set<int[]> set = new TreeSet<>((a, b)->a[0]==b[0] ? Integer.compare(a[1], b[1]) : Integer.compare(a[0], b[0]));
//        for(int i=0; i<k; i++) set.add(new int[]{nums[i], i});
//        for(int i=k, j=0; i<n; i++, j++){
//            ans[j] = findMid(set);
//            set.add(new int[]{nums[i], i});
//            set.remove(new int[]{nums[i-k], i-k});
//        }
//        ans[n-k] = findMid(set);
//        return ans;
//    }
//
//    double findMid(Set<int[]> set){
//        int mid = (set.size() - 1) / 2;
//        Iterator<int[]> it = set.iterator();
//        while(mid-->0) it.next();
//        return set.size()%2 == 0 ? ((double)it.next()[0] + it.next()[0]) / 2 : it.next()[0];
//    }

    public double[] medianSlidingWindow(int[] nums, int k) {
        int len = nums.length;
        double[] res = new double[len - k + 1];
        //大顶堆用于存放小于中位数的值
        //a.compareTo(b) a>b return 1
        PriorityQueue<Long> bigHeap = new PriorityQueue<>((o1, o2) -> {
            if (o2 > o1) {
                return 1;
            } else if (o2 < o1) {
                return -1;
            } else {
                return 0;
            }
        });
        //小顶堆用于存放大于等于中位数的值
        PriorityQueue<Long> smallHeap = new PriorityQueue<>();
        for (int i = 0; i < k; i++) {
            smallHeap.add((long) nums[i]);
        }
        int half = k / 2, idx = 0;
        while (half > 0) {
            bigHeap.add(smallHeap.poll());
            half--;
        }
        if ((k & 1) == 1) {
            res[idx++] = (double) smallHeap.peek();
        } else {
            res[idx++] = ((double) smallHeap.peek() + (double) bigHeap.peek()) / 2;
        }
        for (int i = k; i < len; i++) {
            int addNum = nums[i];
            int delNum = nums[i - k];
            if (!smallHeap.isEmpty() && delNum >= smallHeap.peek()) {
                smallHeap.remove((long) delNum);
            } else {
                bigHeap.remove((long) delNum);
            }
            if (!smallHeap.isEmpty() && addNum >= smallHeap.peek()) {
                smallHeap.add((long) addNum);
            } else {
                bigHeap.add((long) addNum);
            }
            while (smallHeap.size() > bigHeap.size() + 1) {
                bigHeap.add(smallHeap.poll());
            }
            while (smallHeap.size() < bigHeap.size()) {
                smallHeap.add(bigHeap.poll());
            }
            if ((k & 1) == 1) {
                res[idx++] = (double) smallHeap.peek();
            } else {
                res[idx++] = ((double) smallHeap.peek() + (double) bigHeap.peek()) / 2;
            }
        }

        return res;
    }
//    public double[] medianSlidingWindow(int[] nums, int k) {
//        //大顶堆用于存放小于中位数的值
//        PriorityQueue<Long> bigHeap = new PriorityQueue<>((o1, o2) -> {
//            if (o2 - o1 > 0) return 1;
//            if (o2.equals(o1)) return 0;
//            return -1;
//        });
//        //小顶堆用于存放大于等于中位数的值
//        PriorityQueue<Long> smallHeap = new PriorityQueue<>();
//        //初始化 两个堆
//        for (int i = 0; i < k; i++) {
//            smallHeap.add((long) nums[i]);
//        }
//        int half = k / 2;
//        while (half > 0) {
//            bigHeap.add(smallHeap.poll());
//            half--;
//        }
//        double[] res = new double[nums.length - k + 1];
//        // 根据k值来得到第一个窗口的中位数
//        if (k % 2 == 0) res[0] = (double) (smallHeap.peek() + bigHeap.peek()) / 2;
//        else res[0] = smallHeap.peek();
//        for (int i = k; i < nums.length; i++) {//从k开始遍历数字
//            int curNumber = nums[i];//当前遍历的数字
//            int removeNumber = nums[i - k];//滑动窗口移除的数字
//
//            if (!bigHeap.isEmpty() && removeNumber <= bigHeap.peek())//如果移除的数字小于等于大顶堆的堆顶，那么从大顶堆中移除
//                bigHeap.remove((long) removeNumber);
//            else smallHeap.remove((long) removeNumber);//否则从小顶堆中移除
//
//            if (smallHeap.size() != 0 && curNumber >= smallHeap.peek()) {//若当前的数字大于小顶堆的堆顶，那么就加入小顶堆中
//                smallHeap.add((long) curNumber);
//            } else bigHeap.add((long) curNumber);//否则加入大顶堆
//
//            //维护两个堆的大小，小顶堆的大小 = 大顶堆的大小 或者 小顶堆的大小 = 大顶堆的大小 + 1
//            while (bigHeap.size() > smallHeap.size()) smallHeap.add(bigHeap.poll());
//            while (smallHeap.size() > bigHeap.size() + 1) bigHeap.add(smallHeap.poll());
//            // 根据k值来得到中位数
//            if (k % 2 == 0) res[i - k + 1] = (double) (smallHeap.peek() + bigHeap.peek()) / 2;
//            else res[i - k + 1] = smallHeap.peek();
//        }
//        return res;
//    }

    public int[][] merge(int[][] intervals) {
        List<int[]> list = new ArrayList<>();
        int len = intervals.length;
        Arrays.sort(intervals, (a, b) -> a[0] - b[0]);
        list.add(intervals[0]);
        for (int i = 1; i < len; i++) {
            int[] pre = list.get(list.size() - 1);
            if (intervals[i][0] > pre[1]) {
                list.add(intervals[i]);
            } else if (intervals[i][0] <= pre[1]) {
                pre[0] = Math.min(pre[0], intervals[i][0]);
                pre[1] = Math.max(pre[1], intervals[i][1]);
            }
        }
        int[][] res = new int[list.size()][2];
        for (int i = 0; i < list.size(); i++) {
            res[i] = list.get(i);
        }
        return res;

    }

    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int len = nums1.length + nums2.length;
        if ((len & 1) == 1) {
            return findKthMost(nums1, nums2, (len + 1) / 2);

        } else {
            double f1 = findKthMost(nums1, nums2, (len) / 2 + 1);
            double f2 = findKthMost(nums1, nums2, (len) / 2);
            return (f1 + f2) / 2;
        }
    }

    private double findKthMost(int[] nums1, int[] nums2, int target) {
        int s1 = 0, s2 = 0, n1 = nums1.length, n2 = nums2.length;
        while (true) {

            if (s1 == n1) return nums2[s2 + target - 1];
            if (s2 == n2) return nums1[s1 + target - 1];
            if (target == 1) return Math.min(nums1[s1], nums2[s2]);
            int half = target / 2;
            int ne1 = Math.min(n1, s1 + half) - 1, ne2 = Math.min(n2, s2 + half) - 1;
            if (nums1[ne1] < nums2[ne2]) {
                target -= (ne1 - s1 + 1);
                s1 = ne1 + 1;
            } else {
                target -= (ne2 - s2 + 1);
                s2 = ne2 + 1;
            }
        }
    }

    public List<String> fullJustify(String[] words, int maxWidth) {
        List<String> res = new ArrayList<>();
        int len = words.length, cur = 0, sum = 0, pre = 0, count = 0;
        while (cur < len) {
            sum = words[cur].length();
            count = 1;
            cur++;
            while (cur < len && sum < maxWidth) {
                if (sum + 1 + words[cur].length() <= maxWidth) {
                    sum += 1 + words[cur].length();
                    cur++;
                    count++;
                } else {
                    break;
                }
            }
            boolean skip = false;
            if (cur == len) {
                skip = true;
            }
            int space = maxWidth - sum;
            int even = 0, split = 0;
            if (count > 1) {
                even = space / (count - 1);
                split = space % (count - 1);

            }
            StringBuilder sb = new StringBuilder();
            for (int i = pre; i < cur; i++) {
                sb.append(words[i]);
                if (i == cur - 1) break;
                sb.append(' ');
                if (even > 0 && !skip) {
                    for (int j = 0; j < even; j++) {
                        sb.append(' ');
                    }

                }
                if (split > 0 && !skip) {
                    sb.append(' ');
                    split--;
                }
            }
            if (sb.length() != maxWidth) {
                int size = maxWidth - sb.length();
                for (int i = 0; i < size; i++) {
                    sb.append(' ');
                }
            }
            res.add(sb.toString());
            pre = cur;
        }
        return res;
    }

//    public int trap(int[] height) {
//        int len = height.length;
//        int[] leftmax = new int[len];
//        int[] rightmax = new int[len];
//        for (int i = 1; i < len; i++) {
//            leftmax[i]=Math.max(leftmax[i-1],height[i-1]);
//        }
//        for (int i = len-2; i >=0; i--) {
//            rightmax[i]=Math.max(rightmax[i+1],height[i+1]);
//        }
//        int res =0;
//        for (int i = 1 ; i <len ; i++) {
//            int min  = Math.min(leftmax[i],rightmax[i]);
//            if(min>height[i]){
//                res+=(min-height[i]);
//            }
//        }
//        return res;
//    }

    //小顶堆！
//    public int trap(int[] height) {
//        int len = height.length;
//
//    }

    public int shortestPathLength(int[][] graph) {
        int n = graph.length;
        Queue<int[]> queue = new LinkedList<int[]>(); // 三个属性分别为 idx, mask, dist
        boolean[][] vis = new boolean[n][1 << n];
        for (int i = 0; i < n; i++) {
            queue.add(new int[]{i, 1 << i, 0});
//            vis[i][1<<i]=true;
        }
        Map<String, Integer> map = new HashMap<>();
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                int[] cur = queue.poll();
                int idx = cur[0], mask = cur[1], dis = cur[2];

//                String tmp = idx+" "+" "+mask+" "+dis;
//                System.out.println("trace "+tmp);
//                map.put(tmp,map.getOrDefault(tmp,0)+1);
//                System.out.println("map "+tmp+" "+map.get(tmp));
                if (vis[idx][mask]) continue;
                if (mask == ((1 << n) - 1)) return dis;
                vis[idx][mask] = true;
                for (int j = 0; j < graph[idx].length; j++) {
                    int newTarget = graph[idx][j];
                    int newMask = mask | (1 << newTarget);
                    if (!vis[newTarget][newMask]) {
                        queue.offer(new int[]{newTarget, newMask, dis + 1});
                    }
                }
            }
        }
        return 0;
    }

    //    public int shortestPathLength(int[][] graph) {
//        int n = graph.length;
//        Queue<int[]> queue = new LinkedList<int[]>(); // 三个属性分别为 idx, mask, dist
//        boolean[][] vis =new boolean[n][1<<n];
//        for (int i = 0; i < n; i++) {
//            queue.add(new int[]{i,1<<i,0});
//            vis[i][1<<i]=true;
//        }
//        while(!queue.isEmpty()){
//            int size = queue.size();
//            for (int i = 0; i < size; i++) {
//                int[] cur =queue.poll();
//                int idx = cur[0],mask = cur[1],dis = cur[2];
//                if(mask==((1<<n)-1)) return dis;
//                for (int j = 0; j < graph[idx].length; j++) {
//                    int newTarget = graph[idx][j];
//                    int newMask = mask|(1<<newTarget);
//                    if(!vis[newTarget][newMask]) {
//                        queue.offer(new int[]{newTarget, newMask, dis + 1});
//                        vis[newTarget][newMask]=true;
//                    }
//                }
//            }
//        }
//        return 0;
//    }
//    public boolean circularArrayLoop(int[] nums) {
//        int len = nums.length;
//        for (int i = 0; i < len; i++) {
//            int start = i, cur = i;
//            boolean isPositive = nums[start] > 0;
//            boolean skip = false;
//            int count = 0;
//            Set<Integer> seen = new HashSet<>();
//            while (!seen.contains(cur)) {
//                if ((isPositive && nums[cur] < 0) || (!isPositive && nums[cur] > 0)) {
//                    break;
//                }
//                seen.add(cur);
//                count++;
//                cur = cur + nums[cur] ;
//                while(cur<0){
//                    cur+=len;
//                }
//                cur %= len;
//            }
//            if (cur == start && count > 1) return true;
//
//        }
//        return false;
//    }

    public boolean circularArrayLoop(int[] nums) {
        int len = nums.length;
        for (int i = 0; i < len; i++) {
            int cur = i;
            boolean isPositive = nums[i] > 0;
            int count = 0;
//            Set<Integer> seen = new HashSet<>();
            while (nums[cur]!=0) {
                if ((isPositive && nums[cur] < 0) || (!isPositive && nums[cur] > 0)) {
                    break;
                }
//                seen.add(cur);
                count++;
                int pre =cur;
                cur = ((cur + nums[cur])%len+len)%len ;
                nums[pre]=0;
            }
            if (cur == i && count > 1) return true;

        }
        return false;
    }

    public static void main(String[] args) {
//        LRUCache lruCache = new LRUCache(2);
//        lruCache.put(2,1);
//        lruCache.put(1,1);
//        lruCache.put(2,3);
//        lruCache.put(4,1);
//        int r1 =lruCache.get(1);
//        int r2 = lruCache.get(2);
//        System.out.println(r1);
//        System.out.println(r2);

        Lc14 lc14 = new Lc14();
        int[][] s1 = {{}, {0, 2, 3, 4}, {3}, {4}, {}};
//        lc14.eventualSafeNodes(s1);
        int[] s2 = {2147483647, 2147483647};
        int[] s3 = {1, 3, -1, -3, 5, 3, 6, 7};
//        double[] r1 = lc14.medianSlidingWindow(s3, 3);
//        System.out.println(Arrays.toString(r1));

        int[] s4 = {0, 0, 0, 0, 0};
        int[] s5 = {-1, 0, 0, 0, 0, 0, 1};
//        double r2 = lc14.findMedianSortedArrays(s4, s5);
//        System.out.println(r2);

//        System.out.println(new int[]{1, 2}.hashCode());
//        System.out.println(new int[]{1, 2}.hashCode());
//        Set<int[]> set = new TreeSet<>((a, b)->a[0]==b[0] ? Integer.compare(a[1], b[1]) : Integer.compare(a[0], b[0]));
//        Set<int[]> set = new TreeSet<>((a,b)->a[0]-b[0]);
//        Set<int[]> set = new HashSet<>();
//        set.add(new int[]{1, 2});
//        System.out.println(set.contains(new int[]{1, 2}));

        Integer a = 1;
        Integer b = 2;
//        System.out.println(a.compareTo(b));
//        System.out.println(b.compareTo(a));
        String[] s6 = {"This", "is", "an", "example", "of", "text", "justification."};
        String[] s7 = {"Science", "is", "what", "we", "understand", "well", "enough", "to", "explain", "to", "a", "computer.", "Art", "is", "everything", "else", "we", "do"};
//        lc14.fullJustify(s7, 20);

        int[] s8 = {0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1};
//        int r5 =lc14.trap(s8);
//        System.out.println(r5);

        int[][] s9 = {{1, 2, 3}, {0}, {0}, {0}};
        int[][] s10 = {{1}, {0, 2, 4}, {1, 3, 4}, {2}, {1, 2}};
        int[][] s11 = {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, {0, 2, 5, 6, 8}, {0, 1, 4, 5, 6, 9, 10, 11}, {0, 4, 5, 6, 8, 9, 10, 11}, {0, 2, 3, 5, 6, 8, 10}, {0, 1, 2, 3, 4, 6, 8, 9, 10, 11}, {0, 1, 2, 3, 4, 5, 8, 10, 11}, {0, 8}, {0, 1, 3, 4, 5, 6, 7, 9, 10, 11}, {0, 2, 3, 5, 8, 10}, {0, 2, 3, 4, 5, 6, 8, 9}, {0, 2, 3, 5, 6, 8}};
//        int r9 = lc14.shortestPathLength(s11);
//        System.out.println(r9);
//        System.out.println(1<<0);
        int[] s12 = {2, -1, 1, 2, 2};
        int[] s13 = {1, 2, 3, 4, 5};
        int[] s14 = {-2,-3,-9};
        boolean r12 = lc14.circularArrayLoop(s12);
        System.out.println(r12);

    }
}

// class LRUCache {
//    class DLinkedNode {
//        int key;
//        int value;
//        DLinkedNode prev;
//        DLinkedNode next;
//        public DLinkedNode() {}
//        public DLinkedNode(int _key, int _value) {key = _key; value = _value;}
//    }
//
//    private Map<Integer, DLinkedNode> cache = new HashMap<Integer, DLinkedNode>();
//    private int size;
//    private int capacity;
//    private DLinkedNode head, tail;
//
//    public LRUCache(int capacity) {
//        this.size = 0;
//        this.capacity = capacity;
//        // 使用伪头部和伪尾部节点
//        head = new DLinkedNode();
//        tail = new DLinkedNode();
//        head.next = tail;
//        tail.prev = head;
//    }
//
//    public int get(int key) {
//        DLinkedNode target = cache.get(key);
//        if(target==null) return -1;
//        MoveToHead(target);
//        return target.value;
//    }
//
//     private void MoveToHead(DLinkedNode target) {
//        RemoveNode(target);
//        AddToHead(target);
//     }
//
//     private void AddToHead(DLinkedNode target) {
//        target.next=head.next;
//        head.next.prev=target;
//        head.next=target;
//        target.prev=head;
//     }
//
//     private void RemoveNode(DLinkedNode target) {
//        target.prev.next=target.next;
//        target.next.prev=target.prev;
//     }
//
//     public void put(int key, int value) {
//         DLinkedNode target = cache.get(key);
//         if(target==null){
//             DLinkedNode newNode = new DLinkedNode(key,value);
//             cache.put(key,newNode);
//             AddToHead(newNode);
//             if(cache.size()>capacity){
//                 int toberemoved = tail.prev.key;
//                 RemoveNode(tail.prev);
//                 cache.remove(toberemoved);
//             }
//         }else{
//             target.value=value;
//             MoveToHead(target);
//         }
//     }
//
//}

//public class Solution {
//
//    private int rows;
//    private int cols;
//
//    public int numIslands(char[][] grid) {
//        rows = grid.length;
//        if (rows == 0) {
//            return 0;
//        }
//        cols = grid[0].length;
//
//        // 空地的数量
//        int spaces = 0;
//        UnionFind unionFind = new UnionFind(rows * cols);
//        int[][] directions = {{1, 0}, {0, 1}};
//        for (int i = 0; i < rows; i++) {
//            for (int j = 0; j < cols; j++) {
//                if (grid[i][j] == '0') {
//                    spaces++;
//                } else {
//                    // 此时 grid[i][j] == '1'
//                    for (int[] direction : directions) {
//                        int newX = i + direction[0];
//                        int newY = j + direction[1];
//                        // 先判断坐标合法，再检查右边一格和下边一格是否是陆地
//                        if (newX < rows && newY < cols && grid[newX][newY] == '1') {
//                            unionFind.union(getIndex(i, j), getIndex(newX, newY));
//                        }
//                    }
//                }
//            }
//        }
//        return unionFind.getCount() - spaces;
//    }
//
//    private int getIndex(int i, int j) {
//        return i * cols + j;
//    }
//
//    private class UnionFind {
//        /**
//         * 连通分量的个数
//         */
//        private int count;
//        private int[] parent;
//
//        public int getCount() {
//            return count;
//        }
//
//        public UnionFind(int n) {
//            this.count = n;
//            parent = new int[n];
//            for (int i = 0; i < n; i++) {
//                parent[i] = i;
//            }
//        }
//
//        private int find(int x) {
//            while (x != parent[x]) {
//                parent[x] = parent[parent[x]];
//                x = parent[x];
//            }
//            return x;
//        }
//
//        public void union(int x, int y) {
//            int xRoot = find(x);
//            int yRoot = find(y);
//            if (xRoot == yRoot) {
//                return;
//            }
//
//            parent[xRoot] = yRoot;
//            count--;
//        }
//    }
//}


//class Solution {
//    // 邻接表存储的图
//    List<List<Integer>> graph = new ArrayList<List<Integer>>();
//    // 入度数组
//    int[] Indeg = new int[5005];
//
//    // 拓扑排序
//    boolean toposort(int n) {
//        Queue<Integer> q = new LinkedList<Integer>();
//
//        // 首先将入度为 0 的点存入队列
//        for(int i = 0; i < n; i++) {
//            if(Indeg[i] == 0) {
//                q.offer(i);
//            }
//        }
//
//        while(!q.isEmpty()) {
//            // 每次弹出队头元素
//            int cur = q.poll();
//            for(int x : graph.get(cur)) {
//                // 将以其为起点的有向边删除，更新终点入度
//                Indeg[x]--;
//                if(Indeg[x] == 0) q.offer(x);
//            }
//        }
//
//        // 仍有入度不为 0 的点，说明图中有环
//        for(int i = 0; i < n; i++) {
//            if(Indeg[i] != 0) return true;
//        }
//        return false;
//    }
//
//    public boolean circularArrayLoop(int[] nums) {
//        int n = nums.length;
//        for(int i = 0; i < n; i++) {
//            graph.add(new ArrayList<Integer>());
//        }
//
//        // 先处理正向边 nums[i] > 0 的情况
//        for(int i = 0; i < n; i++) {
//            int end = ((i + nums[i]) % n + n) % n;
//            if(nums[i] <= 0 || i == end) continue;
//            graph.get(i).add(end);
//            Indeg[end]++;
//        }
//
//
//        if(toposort(n)) return true;
//
//        graph.clear();
//        for(int i = 0; i < n; i++) {
//            graph.add(new ArrayList<Integer>());
//        }
//        for(int i = 0; i < n; i++) Indeg[i] = 0;
//
//        // 再处理负向边 nums[i] < 0 的情况
//        for(int i = 0; i < n; i++) {
//            int end = ((i + nums[i]) % n + n) % n;
//            if(nums[i] >= 0 || i == end) continue;
//            graph.get(i).add(end);
//            Indeg[end]++;
//        }
//
//        if(toposort(n)) return true;
//
//        return false;
//    }
//}
