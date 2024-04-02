# This is the class of the input root. Do not edit it.
class BinaryTree:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


def branchSums(root):
    # Write your code here.
    sums = list()
    calcBranchSums(root, 0, sums)
    return sums

def calcBranchSums(node, run_sum, sums):
    if node is None:
        return 
    newRunSum = run_sum+ node.value
    if(node.left is None and node.right is None):
        sums.append(newRunSum)
        return
    calcBranchSums(node.left, newRunSum, sums)
    calcBranchSums(node.right, newRunSum, sums)

    

    
        
