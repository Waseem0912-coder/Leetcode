class Solution(object):
    def findDisappearedNumbers(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        k = set(nums)
        res = list()
        for x in range(1,len(nums)+1):
            if x not in k:
                res.append(x)
        return res
