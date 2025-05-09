# Numpa
A lightweight, educational implementation of NumPy-like functionality in pure Python.

## Requirements

### Array Creation

**Common Parameters**:
- `shape`: Tuple or integer specifying array dimensions
- `dtype`: Data type (int or float), defaults to int
- `object`: List/tuple to initialize array
- `a`: Input array to copy

**Functions**:
1. `numpa.empty(shape, dtype=int)`
   - Creates an uninitialized array of given shape
   - Contents are arbitrary until explicitly set

2. `numpa.array(object, dtype=int)`
   - Creates array from list/tuple
   - Example: `numpa.array([1,2,3]) → [1,2,3]`

3. `numpa.zeros(shape, dtype=int)`
   - Creates array filled with 0
   - Example: `numpa.zeros((2,2)) → [[0,0],[0,0]]`

4. `numpa.ones(shape, dtype = int)`: 
    - Creates an array filled with ones

5. `numpa.copy(a)`: 
    - Create a copy of input array

### Array Manipulation

1. `numpa.arange(start, stop, step, dtype=int)`
   - Creates evenly spaced values within interval
   - Similar to Python's range() but returns array

2. `numpa.reshape(a, newshape)`
   - Reshapes array without changing data
   - Example: `reshape([1,2,3,4], (2,2)) → [[1,2],[3,4]]`

3. `numpa.ndarray.flat`:

4. `numpa.ndim`:

5. `numpa.ndarray.flatten`:

6. `numpa.ndarray.T`:

7. `numpa.ndarray.transpose`:

### Mathematical Functions

### Polynomials


