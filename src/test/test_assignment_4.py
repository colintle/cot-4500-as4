from src.main.assignment_4 import question1, question2, question3, question4
import numpy as np

A = np.array([[10, -1, 2, 0],
              [-1, 11, -1, 3],
              [2, -1, 10, -1],
              [0, 3, -1, 8]], dtype=float)
B = np.array([6, 25, -11, 15], dtype=float)
x0 = np.zeros_like(B)

x = question1(A, B, x0)
print("Question 1")
print(x)
print()

x = question2(A, B, x0)
print("Question 2")
print(x)
print()

x = question3(A, B, 1.1, x0)
print("Question 3")
print(x)
print()

x = question4(A, B)
print("Question 4")
print(x)
print()