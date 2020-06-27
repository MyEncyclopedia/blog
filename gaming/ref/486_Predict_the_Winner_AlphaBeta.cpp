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
    int FindMax(vector<int>& nums, int st, int ed, int alpha, int beta, int pre){
        if (st == ed){
            return pre + nums[st];
        }
        int ret = FindMin(nums, st + 1, ed, alpha, beta, pre + nums[st]);
        alpha = max(alpha, ret);
        if (alpha >= beta)
            return alpha;
        ret = max(ret, FindMin(nums, st, ed - 1, alpha, beta, pre + nums[ed]));
        return ret;
    }
    int FindMin(vector<int>& nums, int st, int ed, int alpha, int beta, int pre){
        if (st == ed){
            return pre - nums[st];
        }
        int ret = FindMax(nums, st + 1, ed, alpha, beta, pre - nums[st]);
        beta = min(beta, ret);
        if (alpha >= beta)
            return beta;
        ret = min(ret, FindMax(nums, st, ed - 1, alpha, beta, pre - nums[ed]));
        return ret;
    }
};
