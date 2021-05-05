# #################################################
# chapter 02-02
# #################################################
import numpy as np
x = np.array(12)
print(x)
print(x.ndim)

x = np.array([12, 3, 6, 14, 7])
print(x)
print(x.ndim)

x = np.array([
  [5, 78, 2, 34, 0],
  [5, 78, 2, 34, 0],
  [5, 78, 2, 34, 0]
])
print(x)
print(x.ndim)

x = np.array([x, x, x])
print(x)
print(x.ndim)

from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape)
print(train_images.ndim)
print(train_images.dtype)

digit = train_images[4]
import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

my_slice = train_images[10:100]
my_slice.shape

my_slice = train_images[10:100, :1, :1]
my_slice.shape

my_slice = train_images[:, 14:, 14:]
my_slice.shape

my_slice = train_images[:, 7:-7, 7:-7]
my_slice.shape

## algorithmn image
def naive_relu(x):
  assert len(x.shape) == 2
  x = x.copy()
  for i in range(x.shape[0]):
    for j in range(x.shape[1]):
      x[i, j] = max(x[i, j], 0)
  return x

import numpy as np
z = x + y
z = np.maximum(z, 0.)

## algorithmn image
def naive_add(x, y):
  assert len(x.shape) == 2
  assert x.shape == y.shape
  x = x.copy()
  for i in range(x.shape[0]):
    for j in range(x.shape[1]):
      x[i, j] += y[i, j]
  return x


## algorithm image
def naive_add_matrix_and_vector(x, y):
  assert len(x.shape) == 2
  assert len(y.shape) == 1
  assert x.shape[1] == y.shape[0]
  x = x.copy()
  for i in range(x.shape[0]):
    for j in range(x.shape[1]):
      x[i, j] += y[j]
  return x

## algorithm image
def naive_vector_dot(x, y):
  assert len(x.shape) == 1
  assert len(y.shape) == 1
  assert x.shape[0] == x.shape[0]
  z = 0
  for i in range(x.shape[0]):
    z += x[i] * y[i]
  return z

## algorithm image
def naive_matrix_vector_dot(x, y):
  assert len(x.shape) == 2
  assert len(y.shape) == 1
  assert x.shape[0] == y.shape[0]
  z = np.zeros(x.shape[0])
  for i in range(x.shape[0]):
    for j in range(x.shape[1]):
      z[i] += x[i, j] * y[j]
  return z

## algorithm image
def naive_matrix_dot(x, y):
  assert len(x.shape) == 2
  assert len(y.shape) == 2
  assert x.shape[1] == y.shape[0]
  z = np.zeros((x.shape[0], y.shape[1]))
  for i in range(x.shape[0]):
    for j in range(y.shape[1]):
      row_x = x[i, :]
      column_y = y[:, j]
      z[i, j] = naive_vector_dot(row_x, column_y)
  return z