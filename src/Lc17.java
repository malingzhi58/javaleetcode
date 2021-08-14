public class Lc17 {
    public int maxScore(int[] nums) {
        int N = nums.length;//数组长度 以[3, 4, 6, 8]为例
        int[] dp = new int[1 << N];//dp[st] 表示st这个状态下，N次操作后的最大分数和后
        for (int i = 0; i < 1 << N; i++) {//枚举状态 从 0000 0001 ... 1111[表示全部数都在的]
            int cnt = count(i);//计算当前i中1的个数，1表示这个索引下这个数存在
//                System.out.printf("i:%d,bin:%s,cnt:%d\n", i, PrintUtils.toBinaryString(i, 4), cnt);
            if ((cnt & 1) == 1) continue;//奇数跳过 当1的个数是奇数个时，需要跳过，只有偶数个数才能做gcd
            for (int j = 0; j < N; j++) {//第1个数
                for (int k = j + 1; k < N; k++) {//当前数后面开始枚举第2个数
                    //获取这个两个数组成的10进制的数，如j =0 , k =1
                    //1<<j = 0001 1<<1 = 0010
                    //0001 | 0010 = 0011 也就是十进制的3
                    int st = (1 << j) | (1 << k);
//                        System.out.printf("  j:%d,k:%d,st:%d,st_bin:%s,if:%s\n", j, k, st, PrintUtils.toBinaryString(st, 4), (st & i) == st);
                    if ((st & i) == st) {//i这个状态是否包含st这个状态，包含才有意义
                        dp[i] = Math.max(dp[i], dp[i - st] + gcd(nums[j], nums[k]) * cnt / 2);//cnt是当前1的个数 取这个状态转移来的之前的状态 i-st这个状态
                    }
                }
            }
        }
        return dp[(1 << N) - 1];//- 优先于 <<  相当于取二进制位上各位都为1的结果 恰好是整个数组

    }
    //计算i的1的个数
    public int count(int i) {
        int ans = 0;
        while (i != 0) {
            ans += i & 1;
            i >>>= 1;
        }
        return ans;
    }

    //gcd
    public int gcd(int a, int b) {
//            System.out.printf("a:%d,b:%d\n", a, b);
        return b == 0 ? a : gcd(b, a % b);
    }

    public static void main(String[] args) {

    }
}
