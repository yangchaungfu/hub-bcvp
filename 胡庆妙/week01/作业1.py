
import torch
import numpy


A = numpy.array([[1, 2, 3],
              [4, 5, 6]])

B = numpy.array([[7, 8],
              [9, 10],
              [11, 12]])


print("\n## 1（二维数组）标准矩阵乘法/点积:")
print("矩阵 A:")
print(A)
print("矩阵 B:")
print(B)

print("\nA @ B :")
print(A @ B)

print("\n也可以用 numpy.dot(A,B) :")
print(numpy.dot(A, B))



print("\n\n## 2（二维张量）标准矩阵乘法/点积:")
A = torch.tensor([[1, 2, 3],
              [4, 5, 6]])

B = torch.tensor([[7, 8],
              [9, 10],
              [11, 12]])

print("二维张量 A:")
print(A)
print("二维张量 B:")
print(B)

print("\nA @ B :")
print(A @ B)

print("\n也可以用 torch.mm(A,B) :")
print(torch.mm(A, B))



print("\n\n## 3（一维张量）点积:")
A = torch.tensor([1, 2, 3])

B = torch.tensor([7, 8, 1])

print("一维张量 A:")
print(A)
print("一维张量 B:")
print(B)


print("\nA @ B :")
print(A @ B)

print("\n也可以用 torch.dot(A,B) :")
print(torch.dot(A, B))


