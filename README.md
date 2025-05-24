# TensorPy
A lightweight, educational implementation of NumPy-like functionality in pure Python.

## Requirements

### Array Creation

**Common Parameters**:
- `shape`: Tuple or integer specifying array dimensions
- `dtype`: Data type (int or float), defaults to int
- `object`: List/tuple to initialize array
- `a`: Input array to copy

**Functions**:
1. `.empty(shape, dtype=int)`:
   - Creates an uninitialized array of given shape
   - Contents are arbitrary until explicitly set

2. `.array(object, dtype=int)`:
   - Creates an array from a list/tuple
   - Example: `.array([1,2,3]) → [1,2,3]`

3. `.zeros(shape, dtype=int)`:
   - Creates an array filled with 0
   - Example: `.zeros((2,2)) → [[0,0],[0,0]]`

4. `.ones(shape, dtype = int)`: 
    - Creates an array filled with ones

5. `.copy(a)`: 
    - Create a copy of the input array

### Array Manipulation

1. `.arange(start, stop, step, dtype=int)`:
   - Creates evenly spaced values within the interval
   - Similar to Python's range() but returns array
   - Example:
     ```
      Tensor.arange(0, 5, 1) # → [0, 1, 2, 3, 4]
      Tensor.arange(0.5, 2.0, 0.5) # → [0.5, 1.0, 1.5]
     ```
   - Corresponding method: `.arange(args)`
   - **Challenge**: float arithmetic (floating points are covered -> rounding up after addition)

2. `.reshape(a, newshape)`:
   - Reshapes the array without changing the data
   - Example:
     ```
     Tensor.reshape([1,2,3,4], (2,2)) → [[1,2],[3,4]]
     ```
   - Corresponding method: `.reshape()`
   - **Challenge**: check irregular/inconsistent input list

3. `.ndarray.flat ~= .ndarray.flatten(order = 'C' | 'F')`:
   - Flatten all the elements in the input matrix into a list
   - Example:
     ```
      t = Tensor([[1, 2], [3, 4]])
      t.flatten(order='C') # → [1, 2, 3, 4]
      t.flatten(order='F') # → [1, 3, 2, 4]
     ```
   - **Challenge**: F-style flattening
     
4. `.ndim()`:
   - One of the core methods of `Tensor` class, retrieving the number of dimensions (`n`)
     ```
      Tensor([[1, 2], [3, 4]]).ndim() # → 2
      Tensor([1, 2, 3]).ndim() # → 1
     ```
   - Corresponding methods:
     - `.flatten(order = 'C', 'F)`:
     - `.multi_processing_flatten(order = 'C', 'F')`
   
5. `.ndarray.T ~= .ndarray.transpose(axes : tuple = Optional)`:
   - Example:
     ```
     Tensor([[1, 2], [3, 4]]).transpose() # → [[1, 3], [2, 4]]
     Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).transpose((1, 0, 2)) # -> [[[1, 2], [5, 6]], [[3, 4], [7, 8]]]
     ```
   - Corresponding methods:
      - `.transpose()`:
      - `.multi_processing_transpose()`:
   - **Challenge**: Flip the axes with the given order (especially with increasing count of array dimensions)

### Linear Algebra
1. `.dot(A, B)`:
   - Output the dot product of any 2 N-D arrays `A`, `B`
   - `.iter_dot()`:
   - Example:
     ```
     A = [[ [1, 2, 3], [4, 5, 6] ], [ [7, 8, 9], [10, 11, 12] ]]
     B = [[1, 2], [3, 4], [5, 6]]
     -> A . B = [[ [22, 28], [49, 64] ], [ [76, 100], [103, 136] ]]
     ```
   - Generic Formula
### Mathematical Functions
1. `.prod()`:
   - Generic Formula
   - `.recursive_prod()`:
   - `.iter_prod()`: 
3. `.sum()`:
   - General Formula
   - Corresponding method: `.iter_sum()`
5. `.lcm()`:
6. `.gcd()`:
7. `.add(A, B)`:
   - General Formula:
     
     7.1: Case 1: Both A and B are scalar, return `A + B`
     
     7.2: Case 2: Either A or B is scalar:
     
        `A(i, j, ..., m) = A(i, j, ..., m) + B`, given:
     
        `s = A's shape, 0 <= i < s[0], 0 <= j < s[1], etc`
     
     7.3: Case 3: Both A and B are M-D array and N-D array:
     
        `A (d1, d2, ..., dm); B (e1, e2, ..., en)`
   
        - General Formula:
          `C_{i1}, C`
          
        - 7.3.1: Pad shapes with 1s to make A, B equally dimensional:
     
          `sA = (1,1,..,d1,...,dm); sB = (1,1,..., e1,..., en)`
     
        - 7.3.2: Find the final shape:
              **Rule 1**: For each `d_i` or `e_j` that is missing or equal to 1, we can treat it as 1 and select the higher-dimensional dimension of the other array, since in pure math, this aims to broadcast the final shape to get the higher dimension. For i.e:
             ```
               A (2,3) = [ [1,2,3] [4,5,6] ]
               B (3,) = [1,2,3]
             ```
             - The final shape is `(2,3)` since `B` is **broadcasted** to shape (1,3) then (2,3)
             - After broadcasting, we have:
             ```
             A (+) B = [ [1,2,3] [4,5,6] ] (+) [ [1,2,3]
                                                 [1,2,3] ]
             ```
             
             **Rule 2**: For any pair of matching dimensions between `A` and `B`, we select this dimension for our final shape.
      
        - 7.3.3: Get the indices from the final shape:
          
             **Rule 3**: For the dimension that is broadcast, add `0` to the corresponding index of the coordinate.
          
             **Rule 4**: To find the actual coordinates in the original `A` and `B`, remove the padded 1(s) out of the final coordinates (which are previously generated from the final shape of the result tensor).
          
   - Corresponding method: `.add()`
9. `.divide()`:
10. `.pow()`:
11. `.subtract()`:
12. `.max()`:
13. `.min()`:
14. `.sqrt()`:
    
### Polynomials
1. `.polyadd`:
2. `.polymul`:

### Screenshots

- 05.13.25:
  
  ![image](https://github.com/user-attachments/assets/f85c1555-d278-43bb-88b6-7e0b611d91ff)

- 05.15.25

  ![image](https://github.com/user-attachments/assets/ee1a3d25-11e6-407d-be97-5c98fc735c9d)

  


