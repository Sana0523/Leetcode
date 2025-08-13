// Sum of Subarray Ranges
// https://leetcode.com/problems/sum-of-subarray-ranges/  
// Given an integer array nums, return the sum of the subarray ranges of nums.
// A subarray range is the difference between the maximum and minimum elements in that subarray.
// calculate the next greater element, previous greater element, next smaller element, and previous smaller element for each element in the array.
// Use these to calculate the contribution of each element as a maximum and minimum in all subarrays
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
    long long sumSubarrayMins(vector<int>& arr) {
        long long sum=0;
        vector<int> nse=findnse(arr);
        vector<int> pse=findpse(arr);
        for(int i=0;i<arr.size();i++)
        {
            long long left=i-pse[i];
            long long right=nse[i]-i;
            long long val=(arr[i] * left) * right;
            sum=(long long)(sum+val);
        }
        return sum;
    }
    

    vector<int> findnge(vector<int>& nums)
    {
        stack<int> s;
        int n=nums.size();
        vector<int> ng(n,n);
        for(int i=n-1;i>=0;i--)
        {
            while(!s.empty() && nums[i]>nums[s.top()])
                s.pop();
            ng[i]=s.empty()? n:s.top();
            s.push(i);
        }
        return ng;
    }
    vector<int> findpge(vector<int>& nums)
    {
        stack<int> s;
        int n=nums.size();
        vector<int> pg(n,-1);
        for(int i=0;i<n;i++)
        {
            while(!s.empty() && nums[i]>=nums[s.top()])
                s.pop();
            pg[i]=s.empty()? -1:s.top();
            s.push(i);
        }
        return pg;
    }

    long long sumSubarrayMax(vector<int>& arr) {
        long long sum=0;
        vector<int> nge=findnge(arr);
        vector<int> pge=findpge(arr);
        for(int i=0;i<arr.size();i++)
        {
            long long left=i-pge[i];
            long long right=nge[i]-i;
            long long val=(arr[i] * left) * right;
            sum=(long long)(sum+val);
        }
        return sum;
    }
    long long subArrayRanges(vector<int>& nums) {
        return sumSubarrayMax(nums)-sumSubarrayMins(nums);
    }
};