def invertBinaryTree(tree,k=0):
    if tree is None:
        return tree
    k = tree
    left =  invertBinaryTree(tree.right,k)
    right = invertBinaryTree(tree.left,k)
    k.left = left
    k.right = right
    return k
    
    
# This is the class of the input binary tree.
class BinaryTree:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
