// https://blog.csdn.net/luke2834/article/details/81807569
class Solution {
public:
    int INF = 2e8+5;
    bool PredictTheWinner(vector<int>& nums) {
        if (!(nums.size() & 1))
            return true;
        int ret = FindMax(nums, 0, nums.size() - 1, -INF, INF, 0);
        return ret >= 0;
    }
    int FindMax(vector<int>& nums, int start, int end, int alpha, int beta, int pre){
        if (start == end){
            return pre + nums[start];
        }
        int ret = FindMin(nums, start + 1, end, alpha, beta, pre + nums[start]);
        alpha = max(alpha, ret);
        if (alpha >= beta)
            return alpha;
        ret = max(ret, FindMin(nums, start, end - 1, alpha, beta, pre + nums[end]));
        return ret;
    }
    int FindMin(vector<int>& nums, int start, int end, int alpha, int beta, int pre){
        if (start == end){
            return pre - nums[start];
        }
        int ret = FindMax(nums, start + 1, end, alpha, beta, pre - nums[start]);
        beta = min(beta, ret);
        if (alpha >= beta)
            return beta;
        ret = min(ret, FindMax(nums, start, end - 1, alpha, beta, pre - nums[end]));
        return ret;
    }
};
