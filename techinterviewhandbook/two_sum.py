class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        x=0
        y=0
        for a in nums:
            y =0
            for b in nums:
                if y ==x:
                    y = y+1
                    continue
                if y==0:
                    y= y+1
                    continue
                if(a+b)==target:
                    return [x,y]
                else:
                    y = y+1
                    continue
            x = x+1
            continue

def main():
    nums = [2,7,11,15]
    target = 9
    print(Solution().twoSum(nums,target))
    