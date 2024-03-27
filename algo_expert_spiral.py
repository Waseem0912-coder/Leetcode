# algoexpert Spiral traverse 

def spiralTraverse(array):
    # Write your code here.
    res = []
    startRow, endRow = 0, len(array)-1
    startCol, endCol = 0, len(array[0])-1


    while(startRow <= endRow and startCol <= endCol):
        for col in range(startCol, endCol+1):
            res.append(array[startRow][col])
        for row in range(startRow + 1, endRow+1):
            res.append(array[row][endCol])
        for col in reversed(range(startCol, endCol)):
            if startRow==endRow:
                break
            res.append(array[endRow][col])
        for row in reversed(range(startRow+1, endRow)):
            if startCol==endCol:
                break
            res.append(array[row][startCol])

        startRow+=1
        endRow-=1
        startCol+=1
        endCol-=1

    return res
