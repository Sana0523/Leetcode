// LeetCode 735. Asteroid Collision
// https://leetcode.com/problems/asteroid-collision/
// Given an array of integers asteroids, we need to simulate the collision of asteroids.
// Each asteroid is represented by an integer:  positive means moving right, negative means moving left.
// When two asteroids collide, the smaller one will explode. If they are of equal size, both will explode.
// The remaining asteroids will continue moving in the same direction.      
class Solution {
public:
    vector<int> asteroidCollision(vector<int>& asteroids) {
        vector<int> res;
        stack<int> st;
        int n=asteroids.size();
        for(int i:asteroids)
        {
            int x=0;
            while(x==0 && !st.empty() && st.top()>=0 && i<0)
            {
                if(st.top()==abs(i))
                {
                    st.pop();
                    x=1;
                }
                else if(st.top()>abs(i))
                {
                    x=1;
                }
                else
                {
                    st.pop();
                }
            }
            if(x==0)
                st.push(i);
        }
        while(!st.empty())
        {
            res.push_back(st.top());
            st.pop();
        }
        reverse(res.begin(),res.end());
        return res;
    }
};