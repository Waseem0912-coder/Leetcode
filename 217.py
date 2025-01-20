class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        check =set()
        for x in nums:
            if x in check:
                return True
            check.add(x)
        return False 
