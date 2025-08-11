// sum of subarray minimums
// #907 
// https://leetcode.com/problems/sum-of-subarray-minimums/
// Given an array of integers arr, find the sum of min(b) for every (contiguous) subarray b of arr.
// for every element in arr, find the next smaller element on the right and the previous smaller element on the left.
class Solution {
public:
    vector<int> findnse(vector<int>& nums)
    {
        stack<int> s;
        int n=nums.size();
        vector<int> ns(n,n);
        for(int i=n-1;i>=0;i--)
        {
            while(!s.empty() && nums[i]<nums[s.top()])
                s.pop();
            ns[i]=s.empty()? n:s.top();
            s.push(i);
        }
        return ns;
    }
    vector<int> findpse(vector<int>& nums)
    {
        stack<int> s;
        int n=nums.size();
        vector<int> ps(n,-1);
        for(int i=0;i<n;i++)
        {
            while(!s.empty() && nums[i]<=nums[s.top()])
                s.pop();
            ps[i]=s.empty()? -1:s.top();
            s.push(i);
        }
        return ps;
    }
    int sumSubarrayMins(vector<int>& arr) {
        int sum=0,modu=(int) pow(10,9)+7;
        vector<int> nse=findnse(arr);
        vector<int> pse=findpse(arr);
        for(int i=0;i<arr.size();i++)
        {
            long long left=i-pse[i];
            long long right=nse[i]-i;
            long long val=(arr[i] * left % modu) * right % modu;
            sum=(sum+val)%modu;
        }
        return sum;
    }
};