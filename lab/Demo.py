
# Create a function that receives two matrices as input and returns the matrix multiplication of the two
# If you don't know how to multiply two matrices, check:
# https://en.wikipedia.org/wiki/Matrix_multiplication#Definition
def scalar_product(v1, v2):
    #<------------INSERT CODE HERE------------>
    scalarProduct =0
    for i in range(len(v1)):
        temp = v1[i]*v2[i]
        scalarProduct += temp
    return scalarProduct

def matrix_multiplication(matrix1, matrix2):
    
    #<------------INSERT CODE HERE------------>
    if(len(matrix1[0]) != len(matrix2)):
        print("Can not calculate")
        return [[0]]

    rowNum = len(matrix1)
    colNum = len(matrix2[0])

    results =[[0]*colNum for _ in range(rowNum)]

    for i in range(rowNum):
        v1=matrix1[i]
        for j in range(colNum):
            v2=[]
            for k in range(len(matrix2)):
                v2.append(matrix2[k][j])
                if k + 1==len(matrix2):
                    result = scalar_product(v1,v2)
                    results[i][j]=result
    return results

matrixA = [[1, 3, 5], 
           [2, 4, 6]]

matrixB = [[1, 2, 3],
           [4, 5, 6], 
           [7, 8, 9]]

matrixC = [[8, 9],
           [8, 9],
           [8, 9],
           [8, 9]]

matrixD = [[10, 11], 
           [10, 11],
           [10, 11]]

print(matrix_multiplication(matrixA, matrixD))