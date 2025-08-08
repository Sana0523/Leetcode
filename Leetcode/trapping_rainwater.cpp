// LeetCode #42 Trapping Rain Water
// two pointer approach
// https://leetcode.com/problems/trapping-rain-water/description/
// keep track of the maximum height from left and right side
// at each step, we can calculate the water that can be trapped at that position
class Solution {
public:
    int trap(vector<int>& height) {
        int sum=0,lmax=0,rmax=0;
        int l=0,r,n=height.size();
        r=n-1;
        while(l<=r)
        {
            // at this point if height[l]<height[r] wkt, someone on right is always greater or equal to lmax
            if(height[l]<=height[r]) 
            {
                if(height[l]>lmax)
                    lmax=height[l];
                else
                    sum+=lmax-height[l];
                l++;
            }
            else
            {
                if(height[r]>rmax)
                    rmax=height[r];
                else
                    sum+=rmax-height[r];
                r--;
            }
        }
        return sum;
    }
};