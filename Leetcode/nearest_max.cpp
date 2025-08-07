// https://leetcode.com/problems/next-greater-element-ii/description/
//  #503-Maximum nearest element
// approach- Use a stack to find the next greater element in a circular array.
// circular array means we can treat the array as if it were repeated twice.(i.e., we can iterate through the array twice to find the next greater element for each element).
// where we maintain a decreasing stack to find the next greater element for each element in the array.
// index=0 to n-1, we can use index i%n to access the elements in a circular manner.
class Solution {
public:
    vector<int> nextGreaterElements(vector<int>& nums) {
        vector<int> nge(nums.size(),-1);
        stack<int> st;
        int n=nums.size();
        for(int i=2*n-1;i>=0;i--)
        {
            while(!st.empty() && nums[i%n]>=st.top())
                st.pop();
            // maintain a decreasing stack
            if(i<n)
            {
                if(!st.empty() && st.top()>nums[i]) 
                    nge[i]=st.top();
            }
            st.push(nums[i%n]);
        }
        return nge;
    }
};