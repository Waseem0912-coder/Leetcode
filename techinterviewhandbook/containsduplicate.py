#create a dictionary and add elements from array to it
#check if the element is already in the dictionary
#return true if it is already in the dictionary
#return false if it is not in the dictionary
#time complexity is O(n) and space complexity is O(n)
def containsDuplicate(nums):
    dict = {}
    for i in nums:
        if i in dict:
            return True
        else:
            dict[i] = 1
    return False
def main():
    nums = [1,2,3,1]
    print(containsDuplicate(nums))
if __name__ == "__main__":
    main()
    