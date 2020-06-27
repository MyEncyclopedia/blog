// https://leetcode.com/problems/can-i-win/discuss/268506/C%2B%2B-solution-using-alpha-beta-pruning-with-explaination
class Solution {
public:
    // 1 means the first player wins, -1 means the second player wins otherwise the game 
	// is not determined yet
    bool canIWin(int maxChoosableInteger, int desiredTotal) {
        if (desiredTotal > maxChoosableInteger * (maxChoosableInteger + 1) / 2) return false;
        if (desiredTotal <= 0) return true;
        int value = minimax(0, desiredTotal, 1, maxChoosableInteger, -2, 2);
        return (value == 1);
    }
private:
    unordered_map<int, int> win;
	// state is the bitwise or of all the current choosing points
    int minimax(int state, int total, int player, int maximum, int alpha, int beta) {
        if (win.find(state) != win.end()) {
            return win[state];
        }
        
        int value, i;
		// the maxplayer is playing
        if (player == 1) {
            value = -2;
			// if a winning state is reached return 1
            for (i = 0; i < maximum; i++) {
                if ((((1 << i) & state) == 0) && total <= i + 1) {
                    value = 1;
                    
                } 
            }
            // cannot win by directly choose one number
            if (value == -2) {
                for (i = 0; i < maximum; i++) {
                    if (((1 << i) & state) == 0) {
                        int tmp = minimax((state | (1 << i)), total - i - 1, !player, maximum, alpha, beta);
                        value = max(value, tmp);
                        alpha = max(alpha, value);
						// the beta cut-off
                        if (value == 1) break;
                        if (alpha >= beta) break;
                    } 
                }
            }
        } else {
            value = 2;
			// similar cases for the minimum player
            for (i = 0; i < maximum; i++) {
                if ((((1 << i) & state)== 0) && total <= i + 1) {
                    value = -1;
                } 
            }
            
            if (value == 2) {
                
                for (i = 0; i < maximum; i++) {
                    if (((1 << i) & state) == 0) {
                         int tmp = minimax((state | (1 << i)), total - i - 1, !player, maximum, alpha, beta);
                        value = min(value, tmp);
                        if (value == -1) break;
						// the alpha cutoff
                        beta = min(beta, value);
                        if (alpha >= beta) break;
                    } 
                }
            }
        }
        // memorize the state
        win[state] = value;
        return value;
    }
};
