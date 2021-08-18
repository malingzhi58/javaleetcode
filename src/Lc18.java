import java.util.Arrays;
import java.util.Random;

public class Lc18 {
    public int kthLargestValue(int[][] matrix, int k) {
        int row = matrix.length;
        int col = matrix[0].length;
        int[] nums = new int[row * col];
        nums[0] = matrix[0][0];

        for (int i = 1; i < col; i++) {
            nums[i] = nums[i -1] ^ matrix[0][i];
        }
        for (int i = 1; i < row; i++) {
            nums[col * i] = nums[col * (i - 1)] ^ matrix[i][0];
        }

        for (int i = 1; i < row; i++) {
            for (int j = 1; j < col; j++) {
                nums[i*col + j] = nums[i * col + j - 1] ^ nums[(i - 1) * col + j] ^
                        nums[(i - 1) * col + j - 1] ^  matrix[i][j];
            }
        }
        return findKthLargest(nums, k);
    }

    private int findKthLargest(int[] nums, int k) {
        int left = 0,right =nums.length-1,len = nums.length;
        return findKth(nums,left,right,len-k);
    }

    private int findKth(int[] nums, int left, int right, int k) {
        int mid = quickSelect(nums,left,right);
        if(mid ==k){
            return nums[mid];
        }else if(mid<k){
            return findKth(nums,mid+1,right,k);
        }else{
            return findKth(nums,left,mid-1,k);
        }
    }

    private int quickSelect(int[] nums, int left, int right) {
        Random random = new Random();
        int pivot = random.nextInt(right-left+1)+left;
        swap(nums,pivot,left);
        int i = left,j = right,x = nums[left];
        while(i<j){
            while(i<j&&nums[j]>=x)j--;
            while(i<j&&nums[i]<=x)i++;
            swap(nums,i,j);
            System.out.println(i+':'+j);
        }
        swap(nums,left,i);
        return i;
    }
    public void swap (int[] arr, int left, int right){
        int temp = arr[left];
        arr[left] = arr[right];
        arr[right] = temp;
    }


    public int[] smallestK(int[] arr, int k) {
        if(k==0) return new int[]{};
        return findKth2(arr,0,arr.length-1,k-1);
    }

    private int[] findKth2(int[] arr, int left, int right, int k) {
        int mid = quickSelect(arr,left,right);
        if(mid==k){
            quickSort(arr,0,k-1);
            return Arrays.copyOfRange(arr,0,k+1);
        }else if(mid>k){
            return findKth2(arr,left,mid-1,k);
        }else{
            return findKth2(arr,mid+1,right,k);

        }
    }

    private void quickSort(int[] arr, int left, int right) {
        if(left>=right) return;
        int i =left,j = right,x = arr[left];
        while(i<j){
            while(i<j&&arr[j]>=x)j--;
            while(i<j&&arr[i]<=x)i++;
            swap(arr,i,j);
        }
        swap(arr,left,i);
        quickSort(arr,left,i-1);
        quickSort(arr,i+1,right);
    }

    public int[] xorQueries(int[] arr, int[][] queries) {

    }

    public static void main(String[] args) {
        Lc18 lc18 =new Lc18();
//        lc18.kthLargestValue()
        int[] s1 = {3,1,2,5,4};
//        int r1 =lc18.findKthLargest(s1,2);
//        System.out.println(r1);
        int[] s2 ={1,3,5,7,2,4,6,8};
        int[] r2 = lc18.smallestK(s2,4);
    }
}
