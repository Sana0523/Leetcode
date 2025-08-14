// LeetCode 402. Remove K Digits
// https://leetcode.com/problems/remove-k-digits/
// Given a string num representing a non-negative integer num, and an integer k, return the smallest possible integer after removing k digits from num.
// The result should not contain leading zeros, and if the result is empty, return "0".
// The solution uses a stack to maintain the digits of the resulting number, ensuring that the digits are in non-decreasing order while removing k digits.
// The algorithm iterates through each digit of the number, and if the current digit is smaller than the top of the stack and k is still greater than 0, it pops the stack to remove the larger digit. If k becomes 0, it stops removing digits. Finally, it constructs the result from the stack and removes leading zeros.
class Solution {
public:
    string removeKdigits(string num, int k) {
        stack<char> st;
        int n=num.size();
        string res="";
        if(n<=k)
            return "0";
        for(int i=0;i<n;i++)
        {
            while(!st.empty() && st.top()-'0'>num[i]-'0' && k>0)
            {
                st.pop();
                k--;
            }
            if(st.empty() || k==0 || st.top()-'0'<=num[i]-'0')
                st.push(num[i]);
        }
        while(k>0)
        {
            st.pop();
            k--;
        }
        while(!st.empty())
        {
            char x=st.top();
            res.push_back(x);
            st.pop();
        }
        reverse(res.begin(),res.end());
        int i=0;
        while(res[i]=='0')
            i++;
        res=res.substr(i);
        return res.empty()?"0":res;
    }
};