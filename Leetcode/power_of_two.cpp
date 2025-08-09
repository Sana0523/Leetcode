// power_of_two.cpp
// LeetCode #231 Power of Two
// https://leetcode.com/problems/power-of-two/description/ 
class Solution {
public:
    bool isPowerOfTwo(int n) {
        if(n<=0)
            return false;
        while(n%2==0)
        {
            n=n/2;
        }
        if(n==1)
            return true;
        return false;
    }
};