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
   - **Challenge**: float arithmetic (floating points are covered -> rounding up after addition)

2. `.reshape(a, newshape)`:
   - Reshapes the array without changing the data
   - Example: `reshape([1,2,3,4], (2,2)) → [[1,2],[3,4]]`
   - **Challenge**: check irregular/inconsistent input list

3. `.ndarray.flat ~= .ndarray.flatten(order = 'C' | 'F')`:
   - Flatten all the elements in the input matrix into a list
   - **Challenge**: F-style flattening
     
4. `.ndim()`:
   -
   
5. `.ndarray.T ~= .ndarray.transpose(axes : tuple = Optional)`:
   - **Challenge**: Flip the axes with the given order (especially with increasing count of array dimensions)
   
### Mathematical Functions


### Polynomials

![image](https://github.com/user-attachments/assets/f85c1555-d278-43bb-88b6-7e0b611d91ff)


