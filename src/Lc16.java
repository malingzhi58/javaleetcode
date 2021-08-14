import java.util.*;

public class Lc16 {
    public int countGoodSubstrings(String s) {
        char[] arr = s.toCharArray();
        if (arr.length < 3) return 0;
        Map<Character, Integer> map = new HashMap<>();
        int i = 0;
        for (; i < 3; i++) {
            map.put(arr[i], map.getOrDefault(arr[i], 0) + 1);
        }
        int sum = 0;
        if (map.size() == 3) sum++;
        for (; i < arr.length; i++) {
            int t1 = map.get(arr[i - 3]);
            if (t1 == 1) {
                map.remove(arr[i - 3]);
            } else {
                map.put(arr[i - 3], t1 - 1);
            }
            map.put(arr[i], map.getOrDefault(arr[i], 0) + 1);
            if (map.size() == 3) sum++;
        }
        return sum;
    }

    class Ratio {
        int index;
        double dif;

        public Ratio(int index, double dif) {
            this.index = index;
            this.dif = dif;
        }
    }

    public double maxAverageRatio(int[][] classes, int extraStudents) {
        PriorityQueue<double[]> queue = new PriorityQueue<>((a, b) -> {
            double x1 = (a[0] + 1) / (a[1] + 1) - (a[0] / a[1]);
            double y1 = (b[0] + 1) / (b[1] + 1) - (b[0] / b[1]);
            if (y1 > x1) {
                return 1;
            } else if (y1 < x1) return -1;
            else return 0;
        });
        for (int[] each : classes) {
            queue.offer(new double[]{each[0], each[1]});
        }
        for (int i = 0; i < extraStudents; i++) {
            double[] top = queue.poll();
            queue.offer(new double[]{top[0] + 1, top[1] + 1});
        }
        double res = 0;
        while (!queue.isEmpty()) {
            double[] each = queue.poll();
            res += each[0] / each[1];
        }
        res /= classes.length;
        return res;
    }

    private int[] arr;
    private int len;
    private int[] memo;
    private int[][] gcdArr;

    private int gcd(int m, int n) {
        if(m % n == 0)
            return n;
        else{
            int tmp = m%n;
            return gcd(n, tmp);
        }
    }

    private int dp(int num, int idx) {
        if (num == 0) {
            return 0;
        }

        int key = (num << 3) + idx;
        if (memo[key] != 0) {
            return memo[key];
        }

        int ansMax = 0;
        for (int i = 0; i < len; i++) {
            if ((num & (1 << i)) != 0) {
                // 说明这一位可以选
                for (int j = 0; j < len; j++) {
                    if (j == i) {
                        continue;
                    }

                    if ((num & (1 << j)) != 0) {
                        ansMax = Math.max(ansMax, idx * gcdArr[i][j] + dp(num ^ (1 << i) ^ (1 << j), idx + 1));
                    }
                }
            }
        }

        memo[key] = ansMax;
        return ansMax;
    }

    private void calcGcd() {
        gcdArr = new int[len][len];
        for (int i = 0; i < len; i++) {
            for (int j = 0; j < len; j++) {
                gcdArr[i][j] = gcd(arr[i], arr[j]);
            }
        }
    }

    public int maxScore(int[] nums) {
        arr = nums;
        len = arr.length;
        memo = new int[1 << (len + 3)];
        calcGcd();
        return dp((1 << len) - 1, 1);
    }

    public static void main(String[] args) {
        Lc16 lc16 = new Lc16();
//        int r1 = lc16.countGoodSubstrings("icolgrjedehnd");
//        System.out.println(r1);
        int[][] s2 = {{1, 2}, {3, 5}, {2, 2}};
//        double r2= lc16.maxAverageRatio(s2,2);
//        System.out.println(r2);
//        System.out.println(10 % 12);
        int[] s3 = {3,4,6,8};
        lc16.maxScore(s3);
    }
}

//class Solution {
//
//    public double maxAverageRatio(int[][] classes, int extraStudents) {
//
//        int n = classes.length;
//        // 定义优先队列，优先级按照增加 1 名学生之后能够产生的最大贡献来排序
//        PriorityQueue<double[]> queue = new PriorityQueue<double[]>((o1, o2) -> {
//
//            double x = ((o2[0] + 1) / (o2[1] + 1) - o2[0] / o2[1]);
//            double y = ((o1[0] + 1) / (o1[1] + 1) - o1[0] / o1[1]);
//            if (x > y) return 1;
//            if (x < y) return -1;
//            return 0;
//        });
//
//        // 转化为 double，方便小数计算
//        for (int[] c : classes) {
//
//            queue.offer(new double[]{c[0], c[1]});
//        }
//
//        // 分配学生，每次分配 1 名
//        while (extraStudents > 0) {
//
//            double[] maxClass = queue.poll(); //取出能够产生最大影响的班级
//            maxClass[0] += 1.0; //通过的人数
//            maxClass[1] += 1.0; //班级总人数
//
//            queue.offer(maxClass); //将更新后的重新加入队列中
//            extraStudents--;
//        }
//
//        // 计算最终结果
//        double res = 0;
//        while (!queue.isEmpty()) {
//
//            double[] c = queue.poll();
//            res += (c[0] / c[1]);
//        }
//        return res / n;
//    }
//}

//class Solution {
//    private int[] arr;
//    private int len;
//    private int[] memo;
//    private int[][] gcdArr;
//
//    private int gcd(int m, int n) {
//        return m % n == 0 ? n : gcd(n, m % n);
//    }
//
//    private int dp(int num, int idx) {
//        if (num == 0) {
//            return 0;
//        }
//
//        int key = (num << 3) + idx;
//        if (memo[key] != 0) {
//            return memo[key];
//        }
//
//        int ansMax = 0;
//        for (int i = 0; i < len; i++) {
//            if ((num & (1 << i)) != 0) {
//                // 说明这一位可以选
//                for (int j = 0; j < len; j++) {
//                    if (j == i) {
//                        continue;
//                    }
//
//                    if ((num & (1 << j)) != 0) {
//                        ansMax = Math.max(ansMax, idx * gcdArr[i][j] + dp(num ^ (1 << i) ^ (1 << j), idx + 1));
//                    }
//                }
//            }
//        }
//
//        memo[key] = ansMax;
//        return ansMax;
//    }
//
//    private void calcGcd() {
//        gcdArr = new int[len][len];
//        for (int i = 0; i < len; i++) {
//            for (int j = 0; j < len; j++) {
//                gcdArr[i][j] = gcd(arr[i], arr[j]);
//            }
//        }
//    }
//
//    public int maxScore(int[] nums) {
//        arr = nums;
//        len = arr.length;
//        memo = new int[1 << (len + 3)];
//        calcGcd();
//        return dp((1 << len) - 1, 1);
//    }
//
//}
