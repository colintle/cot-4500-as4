import numpy as np

def question1(A, B, x):                                                                                                                                                                    
    matrix1 = np.diag(A)
    matrix2 = A - np.diagflat(matrix1)

    for _ in range(10):
        new = (B - np.dot(matrix2, x)) / matrix1
        if np.allclose(x, new, atol=1e-3):
            break
        x = new
    return x

def question2(A, B, x):
    for _ in range(100):
        old = np.copy(x)
        
        for i in range(A.shape[0]):
            sum1 = np.dot(A[i, :i], x[:i])
            sum2 = np.dot(A[i, i + 1:], old[i + 1:])
            x[i] = (B[i] - sum1 - sum2) / A[i, i]
        
        tolerance = np.linalg.norm(x - old, ord=np.inf) / np.linalg.norm(x, ord=np.inf)
        if tolerance < 1e-3:
            return x
        
def question3(A, B, w, x):
    for _ in range(100):
        old = np.copy(x)

        for i in range(A.shape[0]):
            sum1 = np.dot(A[i, :i], x[:i])
            sum2 = np.dot(A[i, i + 1:], old[i + 1:])
            x[i] = (1 - w) * old[i] + w / A[i, i] * (B[i] - sum1 - sum2)

        tolerance = np.linalg.norm(x - old, ord=np.inf) / np.linalg.norm(x, ord=np.inf)
        if tolerance < 1e-3:
            return x
        
def gaussian(a, b):
    n = len(b)
    
    for i in range(n):
        maximum = abs(a[i][i])
        maxRow = i
        for j in range(i+1, n):
            if abs(a[j][i]) > maximum:
                maximum = abs(a[j][i])
                maxRow = j

        for j in range(i, n):
            a[maxRow][j], a[i][j] = a[i][j], a[maxRow][j]
        b[maxRow], b[i] = b[i], b[maxRow]

        for j in range(i+1, n):
            temp = -a[j][i] / a[i][i]
            for k in range(i, n):
                if i == k:
                    a[j][k] = 0
                else:
                    a[j][k] += temp * a[i][k]
            b[j] += temp * b[i]

    x = [0 for _ in range(n)]
    for i in range(n-1, -1, -1):
        x[i] = b[i] / a[i][i]
        for k in range(i-1, -1, -1):
            b[k] -= a[k][i] * x[i]

    return x
     
def question4(A, B):
    x = np.array(gaussian(A, B))
    matrix1 = B - np.dot(A, x)

    for _ in range(100):
        y = np.array(gaussian(A, matrix1))
        new = x + y

        tolerance = np.linalg.norm(new - x, ord=np.inf)
        if tolerance < 1e-3:
            return new.tolist()

        x = new
        matrix1 = B - np.dot(A, x)



