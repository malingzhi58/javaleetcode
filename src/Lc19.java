import java.util.Random;

public class Lc19 {
    Random random = new Random();

    public int findKthLargest(int[] nums, int k) {
        int n = nums.length;
        return quickSelect(nums, 0, n - 1, n - k);
    }

    private int quickSelect(int[] nums, int left, int right, int index) {
        int mid = randomPartition(nums, left, right);
        if (mid == index) {
            return nums[mid];
        } else if (mid > index) {
            return quickSelect(nums, left, mid - 1, index);
        } else {
            return quickSelect(nums, mid + 1, right, index);
        }
    }

    private int randomPartition(int[] nums, int left, int right) {
        int i = random.nextInt(right - left + 1) + left;
        swap(nums, i, right);
        return Partitison(nums, left, right);
    }

    private int randomPartition2(int[] nums, int left, int right) {
        int i = random.nextInt(right - left + 1) + left;
        swap(nums, i, left);
        return Partitison2(nums, left, right);
    }

    private int Partitison2(int[] nums, int left, int right) {
        int i = left;
        int j = right;
        int x = nums[left];
        while (i < j) {
            while (j > i && nums[j] > x) j--;
            while (j > i && nums[i] < x) i++;
            swap(nums, i, j);
        }
        swap(nums, i, left);
        return i;
    }

    private int Partitison(int[] nums, int left, int right) {
        int i = left - 1;
        int j = right;
        int x = nums[right];
        while (true) {
            while (nums[++i] < x) ;
            while (j > 0 && nums[--j] > x) ;
            if (i < j) {
                swap(nums, i, j);
            } else {
                break;
            }
        }
        swap(nums, i, right);
        return i;
    }

    public void swap(int[] arr, int left, int right) {
        int temp = arr[left];
        arr[left] = arr[right];
        arr[right] = temp;
    }

    public static void main(String[] args) {
        Lc19 lc19 =new Lc19();
        int[] s1 = {3,1,2,5,4};
        int r1 =lc19.findKthLargest(s1,2);
        System.out.println(r1);
    }
}
