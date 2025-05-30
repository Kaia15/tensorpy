import unittest
import numpy as np
from typing import Union
import math
from tensor import Tensor  

class Test(unittest.TestCase):
    """Test suite to validate Tensor class functionality against NumPy equivalents."""
    
    def setUp(self):
        """Set up common test data."""
        # Simple 1D, 2D, and 3D data for tests
        self.data_1d = [1, 2, 3, 4, 5]
        self.data_2d = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        self.data_3d = [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]
        
        # Create NumPy arrays
        self.np_1d = np.array(self.data_1d)
        self.np_2d = np.array(self.data_2d)
        self.np_3d = np.array(self.data_3d)
        
        # Create custom Tensor objects
        self.tensor_1d = Tensor(self.data_1d)
        self.tensor_2d = Tensor(self.data_2d)
        self.tensor_3d = Tensor(self.data_3d)

    def test_shape(self):
        """Test that tensor shapes match NumPy shapes."""
        self.assertEqual(self.tensor_1d.shape, self.np_1d.shape)
        self.assertEqual(self.tensor_2d.shape, self.np_2d.shape)
        self.assertEqual(self.tensor_3d.shape, self.np_3d.shape)
        
    def test_ones(self):
        """Test Tensor.ones against np.ones."""
        shapes_to_test = [(3,), (2, 3), (2, 3, 4)]
        
        for shape in shapes_to_test:
            np_ones = np.ones(shape)
            tensor_ones = Tensor.ones(shape)
            
            # Convert tensor data back to numpy for comparison
            tensor_as_np = np.array(tensor_ones.data)
            
            self.assertTrue(np.array_equal(np_ones, tensor_as_np))
            self.assertEqual(tensor_ones.shape, np_ones.shape)
    
    def test_zeros(self):
        """Test Tensor.zeros against np.zeros."""
        shapes_to_test = [(3,), (2, 3), (2, 3, 4)]
        
        for shape in shapes_to_test:
            np_zeros = np.zeros(shape)
            tensor_zeros = Tensor.zeros(shape)
            
            # Convert tensor data back to numpy for comparison
            tensor_as_np = np.array(tensor_zeros.data)
            
            self.assertTrue(np.array_equal(np_zeros, tensor_as_np))
            self.assertEqual(tensor_zeros.shape, np_zeros.shape)
    
    def test_arange(self):
        """Test Tensor.arange against np.arange with specific attention to float handling."""
        # Test cases for integer arguments
        int_test_cases = [
            (10,),              # Single argument (end)
            (1, 10),            # Two arguments (start, end)
            (1, 10, 2),         # Three arguments (start, end, step)
            (-5, 5),            # Negative start
            (10, 0, -1),        # Negative step
            # (0, 0, 1)           # Empty range
        ]
        
        for args in int_test_cases:
            np_arange = np.arange(*args)
            tensor_arange = Tensor.arange(*args)
            
            # Convert tensor data back to numpy for comparison
            tensor_as_np = np.array(tensor_arange.data)
            
            self.assertTrue(np.array_equal(np_arange, tensor_as_np), 
                          f"Failed with integer args {args}: Expected {np_arange}, got {tensor_arange.data}")
            self.assertEqual(len(tensor_arange.data), len(np_arange))
        
        # Test cases for float arguments (with special attention to precision)
        float_test_cases = [
            (0.0, 5.0, 1.0),     # Basic float with integer-like values
            (0, 5, 0.5),         # Mix of int and float
            (0.5, 5.5, 0.5),     # All float values
            (0.1, 0.5, 0.1),     # Small float values
            (1.5, -1.5, -0.5),   # Negative step with floats
            (0.567, 5.432, 0.123) # Testing precision handling
        ]
        
        for args in float_test_cases:
            np_arange = np.arange(*args)
            tensor_arange = Tensor.arange(*args)
            
            # Convert tensor data to numpy and ensure they have the same length
            tensor_as_np = np.array(tensor_arange.data)
            self.assertEqual(len(tensor_arange.data), len(np_arange), 
                           f"Length mismatch with float args {args}")
            
            # For float comparisons, we need to account for minor rounding differences
            # Check value by value with the appropriate precision
            for i, (np_val, tensor_val) in enumerate(zip(np_arange, tensor_arange.data)):
                self.assertAlmostEqual(np_val, tensor_val, 
                                     msg=f"Value mismatch at index {i} with args {args}")
            
    def test_transpose(self):
        """Test Tensor.transpose against np.transpose."""
        # Test default transpose (reverse dimensions)
        np_transpose_1d = np.transpose(self.np_1d)
        np_transpose_2d = np.transpose(self.np_2d)
        np_transpose_3d = np.transpose(self.np_3d)
        
        tensor_transpose_1d = self.tensor_1d.transpose()
        tensor_transpose_2d = self.tensor_2d.transpose()
        tensor_transpose_3d = self.tensor_3d.transpose()
        
        # Convert tensor data back to numpy for comparison
        tensor_as_np_1d = np.array(tensor_transpose_1d.data)
        tensor_as_np_2d = np.array(tensor_transpose_2d.data)
        tensor_as_np_3d = np.array(tensor_transpose_3d.data)
        
        self.assertTrue(np.array_equal(np_transpose_1d, tensor_as_np_1d))
        self.assertTrue(np.array_equal(np_transpose_2d, tensor_as_np_2d))
        self.assertTrue(np.array_equal(np_transpose_3d, tensor_as_np_3d))
        
        # Test custom axes for 3D array
        custom_axes = (1, 0, 2)
        np_transpose_custom = np.transpose(self.np_3d, axes=custom_axes)
        tensor_transpose_custom = self.tensor_3d.transpose(axes=custom_axes)
        
        # Convert tensor data back to numpy for comparison
        tensor_as_np_custom = np.array(tensor_transpose_custom.data)
        
        self.assertTrue(np.array_equal(np_transpose_custom, tensor_as_np_custom))
        self.assertEqual(tensor_transpose_custom.shape, np_transpose_custom.shape)
    
    def test_multi_processing_transpose(self):
        """Test multi_processing_transpose against numpy's transpose."""
        # Only testing 3D array as it's more likely to benefit from multi-processing
        np_transpose = np.transpose(self.np_3d)
        tensor_transpose = self.tensor_3d.multi_processing_transpose()
        
        # Convert tensor data back to numpy for comparison
        tensor_as_np = np.array(tensor_transpose.data)
        
        self.assertTrue(np.array_equal(np_transpose, tensor_as_np))
        self.assertEqual(tensor_transpose.shape, np_transpose.shape)
        
        # Test custom axes
        custom_axes = (1, 0, 2)
        np_transpose_custom = np.transpose(self.np_3d, axes=custom_axes)
        tensor_transpose_custom = self.tensor_3d.multi_processing_transpose(axes=custom_axes)
        
        # Convert tensor data back to numpy for comparison
        tensor_as_np_custom = np.array(tensor_transpose_custom.data)
        
        self.assertTrue(np.array_equal(np_transpose_custom, tensor_as_np_custom))
        self.assertEqual(tensor_transpose_custom.shape, np_transpose_custom.shape)
    
    def test_reshape(self):
        """Test Tensor.reshape against np.reshape."""
        test_shapes = [
            (5,),         # 1D to 1D
            (1, 5),       # 1D to 2D
            (5, 1),       # 1D to 2D alternative
            (3, 3),       # 2D to 2D with same elements
            (9,),         # 2D to 1D
            (1, 1, 9)     # 2D to 3D
        ]

        # Test with 1D tensor/array
        for shape in test_shapes[:3]:  # First three shapes are compatible with 1D
            np_reshaped = np.reshape(self.np_1d, shape)
            # print (self.tensor_1d.data)
            tensor_reshaped = Tensor.reshape(self.tensor_1d, shape)
            
            # Convert tensor data back to numpy for comparison
            tensor_as_np = np.array(tensor_reshaped.data)
            
            self.assertTrue(np.array_equal(np_reshaped, tensor_as_np))
            self.assertEqual(tensor_reshaped.shape, np_reshaped.shape)
        
        # Test with 2D tensor/array
        for shape in test_shapes[3:]:  
            np_reshaped = np.reshape(self.np_2d, shape)
            tensor_reshaped = Tensor.reshape(self.tensor_2d, shape)
            
            # Convert tensor data back to numpy for comparison
            tensor_as_np = np.array(tensor_reshaped.data)
            
            self.assertTrue(np.array_equal(np_reshaped, tensor_as_np))
            self.assertEqual(tensor_reshaped.shape, np_reshaped.shape)
        
        # Test order parameter
        orders = ['C', 'F']
        for order in orders:
            np_reshaped = np.reshape(self.np_2d, (9,), order=order)
            tensor_reshaped = Tensor.reshape(self.tensor_2d, (9,), order=order)
            
            # # Convert tensor data back to numpy for comparison
            tensor_as_np = np.array(tensor_reshaped.data)
            
            self.assertTrue(np.array_equal(np_reshaped, tensor_as_np))
    
    def test_flatten(self):
        """Test Tensor.flatten against np.flatten/np.ravel."""
        # Test C-order flatten (default)
        np_flat_c_1d = self.np_1d.flatten(order='C')
        np_flat_c_2d = self.np_2d.flatten(order='C')
        np_flat_c_3d = self.np_3d.flatten(order='C')
        
        tensor_flat_c_1d = self.tensor_1d.flatten(order='C')
        tensor_flat_c_2d = self.tensor_2d.flatten(order='C')
        tensor_flat_c_3d = self.tensor_3d.flatten(order='C')
        
        self.assertTrue(np.array_equal(np_flat_c_1d, tensor_flat_c_1d))
        self.assertTrue(np.array_equal(np_flat_c_2d, tensor_flat_c_2d))
        self.assertTrue(np.array_equal(np_flat_c_3d, tensor_flat_c_3d))
        
        # Test F-order flatten
        np_flat_f_1d = self.np_1d.flatten(order='F')
        np_flat_f_2d = self.np_2d.flatten(order='F')
        np_flat_f_3d = self.np_3d.flatten(order='F')
        
        tensor_flat_f_1d = self.tensor_1d.flatten(order='F')
        tensor_flat_f_2d = self.tensor_2d.flatten(order='F')
        tensor_flat_f_3d = self.tensor_3d.flatten(order='F')
        
        self.assertTrue(np.array_equal(np_flat_f_1d, tensor_flat_f_1d))
        self.assertTrue(np.array_equal(np_flat_f_2d, tensor_flat_f_2d))
        self.assertTrue(np.array_equal(np_flat_f_3d, tensor_flat_f_3d))
    
    def test_multi_processing_flatten(self):
        """Test multi_processing_flatten against numpy's flatten."""
        # Test C-order flatten (default)
        np_flat_c_3d = self.np_3d.flatten(order='C')
        tensor_flat_c_3d = self.tensor_3d.multi_processing_flatten(order='C')
        
        self.assertTrue(np.array_equal(np_flat_c_3d, tensor_flat_c_3d))
        
        # Test F-order flatten
        np_flat_f_3d = self.np_3d.flatten(order='F')
        tensor_flat_f_3d = self.tensor_3d.multi_processing_flatten(order='F')
        
        self.assertTrue(np.array_equal(np_flat_f_3d, tensor_flat_f_3d))
    
    def test_dot(self):
        """Test Tensor.dot against np.dot."""
        # Test scalar dot scalar
        scalar1, scalar2 = 5, 7
        np_dot_scalar = np.dot(scalar1, scalar2)
        tensor_dot_scalar = Tensor.dot(scalar1, scalar2)
        
        self.assertEqual(np_dot_scalar, tensor_dot_scalar)
        
        # Test scalar dot array
        np_dot_scalar_array = np.dot(scalar1, self.np_1d)
        tensor_dot_scalar_array = Tensor.dot(scalar1, self.tensor_1d)
        
        # Convert tensor data back to numpy for comparison
        tensor_as_np = np.array(tensor_dot_scalar_array.data)
        
        self.assertTrue(np.array_equal(np_dot_scalar_array, tensor_as_np))
        
        # Test 1D dot 1D (inner product)
        np_dot_1d = np.dot(self.np_1d, self.np_1d)
        tensor_dot_1d = Tensor.dot(self.tensor_1d, self.tensor_1d)
            
        self.assertEqual(np_dot_1d, tensor_dot_1d)
        
        # Test 2D dot 2D (matrix multiplication)
        np_dot_2d = np.dot(self.np_2d, self.np_2d.T)  # Ensure compatible shapes
        tensor_dot_2d = Tensor.dot(self.tensor_2d, Tensor.transpose(self.tensor_2d))
        
        # Convert tensor data back to numpy for comparison
        tensor_as_np = np.array(tensor_dot_2d.data)
        
        self.assertTrue(np.allclose(np_dot_2d, tensor_as_np))
        
        # Test more complex dot operations if your implementation supports them
        # For example, testing a batch matrix multiplication scenario
        A = [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]]
        ]
        B = [
            [[9, 10], [11, 12]],
            [[13, 14], [15, 16]]
        ]
        
        np_A = np.array(A)
        np_B = np.array(B)
        tensor_A = Tensor(A)
        tensor_B = Tensor(B)
        
        # For more complex cases, it depends on how your dot implementation handles batched operations
        # This is a simplified test that assumes your implementation handles it similarly to np.matmul
        try:
            # Using matmul for batch matrix multiplication
            np_dot_complex = np.dot(np_A, np_B)
            tensor_dot_complex = Tensor.dot(tensor_A, tensor_B)
            
            # Convert tensor data back to numpy for comparison
            tensor_as_np = np.array(tensor_dot_complex.data)
            
            self.assertTrue(np.allclose(np_dot_complex, tensor_as_np))
        except Exception as e:
            # If your implementation doesn't support this operation, just skip this test
            print(f"Skipping complex dot test: {e}")
    
    def test_prod(self):
        """Test Tensor.prod against np.prod."""
        # Test prod with no axis (all elements)
        np_prod_1d_all = np.prod(self.np_1d)
        tensor_prod_1d_all = Tensor.prod(self.tensor_1d)
        self.assertEqual(np_prod_1d_all, tensor_prod_1d_all)
        
        np_prod_2d_all = np.prod(self.np_2d)
        tensor_prod_2d_all = Tensor.prod(self.tensor_2d)
        self.assertEqual(np_prod_2d_all, tensor_prod_2d_all)
        
        np_prod_3d_all = np.prod(self.np_3d)
        tensor_prod_3d_all = Tensor.prod(self.tensor_3d)
        self.assertEqual(np_prod_3d_all, tensor_prod_3d_all)
        
        # Test prod along specific axes
        # 2D array tests
        for axis in [0, 1]:
            np_prod_2d_axis = np.prod(self.np_2d, axis=axis)
            tensor_prod_2d_axis = Tensor.prod(self.tensor_2d, axis=axis)
            
            # Convert tensor result to numpy
            tensor_as_np = np.array(tensor_prod_2d_axis.data)
            
            self.assertTrue(np.array_equal(np_prod_2d_axis, tensor_as_np),
                          f"Product along axis {axis} failed for 2D array")
        
        # 3D array tests
        for axis in [0, 1, 2]:
            np_prod_3d_axis = np.prod(self.np_3d, axis=axis)
            tensor_prod_3d_axis = Tensor.prod(self.tensor_3d, axis=axis)
            
            # Convert tensor result to numpy
            tensor_as_np = np.array(tensor_prod_3d_axis.data)
            
            self.assertTrue(np.array_equal(np_prod_3d_axis, tensor_as_np),
                          f"Product along axis {axis} failed for 3D array")
        
        # Test with larger test case
        test_4d = [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]]
        np_4d = np.array(test_4d)
        tensor_4d = Tensor(test_4d)
        
        # Test axis=2 for 4D array
        np_prod_4d_axis2 = np.prod(np_4d, axis=2)
        tensor_prod_4d_axis2 = Tensor.prod(tensor_4d, axis=2)
        
        tensor_as_np = np.array(tensor_prod_4d_axis2.data)
        self.assertTrue(np.array_equal(np_prod_4d_axis2, tensor_as_np))

    # def test_recursive_prod(self):
    #     """Test Tensor.recursive_prod against np.prod."""
    #     # Test prod with no axis (all elements)
    #     np_prod_1d_all = np.prod(self.np_1d)
    #     tensor_prod_1d_all = Tensor.recursive_prod(self.tensor_1d)
    #     self.assertEqual(np_prod_1d_all, tensor_prod_1d_all)
        
    #     np_prod_2d_all = np.prod(self.np_2d)
    #     tensor_prod_2d_all = Tensor.recursive_prod(self.tensor_2d)
    #     self.assertEqual(np_prod_2d_all, tensor_prod_2d_all)
        
    #     # Test prod along specific axes
    #     for axis in [0, 1]:
    #         np_prod_2d_axis = np.prod(self.np_2d, axis=axis)
    #         tensor_prod_2d_axis = Tensor.recursive_prod(self.tensor_2d, axis=axis)
    #         print (np_prod_2d_axis, tensor_prod_2d_axis)
            
    #         # Convert tensor result to numpy
    #         tensor_as_np = np.array(tensor_prod_2d_axis.data) if isinstance(tensor_prod_2d_axis, Tensor) else tensor_prod_2d_axis
            
    #         self.assertTrue(np.array_equal(np_prod_2d_axis, tensor_as_np),
    #                       f"Recursive product along axis {axis} failed for 2D array")

    def test_sum(self):
        """Test Tensor_sum against np.sum."""
        # Test sum with no axis (all elements)
        np_sum_1d_all = np.sum(self.np_1d)
        tensor_sum_1d_all = Tensor.sum(self.tensor_1d)
        self.assertEqual(np_sum_1d_all, tensor_sum_1d_all)
        
        np_sum_2d_all = np.sum(self.np_2d)
        tensor_sum_2d_all = Tensor.sum(self.tensor_2d)
        self.assertEqual(np_sum_2d_all, tensor_sum_2d_all)
        
        np_sum_3d_all = np.sum(self.np_3d)
        tensor_sum_3d_all = Tensor.sum(self.tensor_3d)
        self.assertEqual(np_sum_3d_all, tensor_sum_3d_all)
        
        # Test sum along specific axes
        # 2D array tests
        for axis in [0, 1]:
            np_sum_2d_axis = np.sum(self.np_2d, axis=axis)
            tensor_sum_2d_axis = Tensor.sum(self.tensor_2d, axis=axis)
            
            # Convert tensor result to numpy
            tensor_as_np = np.array(tensor_sum_2d_axis.data)
            
            self.assertTrue(np.array_equal(np_sum_2d_axis, tensor_as_np),
                          f"Sum along axis {axis} failed for 2D array")
        
        # 3D array tests
        for axis in [0, 1, 2]:
            np_sum_3d_axis = np.sum(self.np_3d, axis=axis)
            tensor_sum_3d_axis = Tensor.sum(self.tensor_3d, axis=axis)
            
            # Convert tensor result to numpy
            tensor_as_np = np.array(tensor_sum_3d_axis.data)
            
            self.assertTrue(np.array_equal(np_sum_3d_axis, tensor_as_np),
                          f"Sum along axis {axis} failed for 3D array")

    def test_add(self):
        """Test Tensor.add against np.add and broadcasting."""
        # Test scalar + scalar
        self.assertEqual(Tensor.add(5, 3), 8)
        
        # Test scalar + array
        scalar = 10
        np_scalar_add = self.np_1d + scalar
        tensor_scalar_add = Tensor.add(scalar, self.tensor_1d)
        tensor_as_np = np.array(tensor_scalar_add.data)
        self.assertTrue(np.array_equal(np_scalar_add, tensor_as_np))
        
        # Test array + array (same shape)
        np_add_same = self.np_1d + self.np_1d
        tensor_add_same = Tensor.add(self.tensor_1d, self.tensor_1d)
        tensor_as_np = np.array(tensor_add_same.data)
        self.assertTrue(np.array_equal(np_add_same, tensor_as_np))
        
        # Test broadcasting cases
        # Case 1: (3, 1) + (1, 3) -> (3, 3)
        A = [[1], [2], [3]]
        B = [[4, 5, 6]]
        np_A = np.array(A)
        np_B = np.array(B)
        
        np_broadcast_add = np_A + np_B
        tensor_broadcast_add = Tensor.add(A, B)
        tensor_as_np = np.array(tensor_broadcast_add.data)
        
        self.assertTrue(np.array_equal(np_broadcast_add, tensor_as_np))
        
        # Case 2: More complex broadcasting (1, 1, 3) + (1, 2, 1) -> (1, 2, 3)
        A = [[[1, 2, 3]]]
        B = [[[4], [5]]]
        np_A = np.array(A)
        np_B = np.array(B)
        
        np_broadcast_add2 = np_A + np_B
        tensor_broadcast_add2 = Tensor.add(A, B)
        tensor_as_np2 = np.array(tensor_broadcast_add2.data)
        
        self.assertTrue(np.array_equal(np_broadcast_add2, tensor_as_np2))
        
        # Test incompatible shapes (should raise ValueError)
        A = [[1, 2, 3], [4, 5, 6]]  # (2, 3)
        B = [[1, 2], [3, 4]]        # (2, 2)
        
        with self.assertRaises(ValueError):
            Tensor.add(A, B)

    def test_lcm(self):
        """Test Tensor.lcm against math.lcm and broadcasting."""
        # Test scalar LCM
        self.assertEqual(Tensor.lcm(4, 6), math.lcm(4, 6))  # Should be 12
        self.assertEqual(Tensor.lcm(12, 18), math.lcm(12, 18))  # Should be 36
        
        # Test scalar + array LCM
        scalar = 6
        A = [2, 3, 4, 5]
        expected = [math.lcm(scalar, x) for x in A]
        
        tensor_lcm = Tensor.lcm(scalar, A)
        result = tensor_lcm.data if isinstance(tensor_lcm, Tensor) else [tensor_lcm]
        
        self.assertEqual(result, expected)
        
        # Test array + array LCM (same shape)
        A = [2, 4, 6, 8]
        B = [3, 6, 9, 12]
        expected = [math.lcm(a, b) for a, b in zip(A, B)]
        
        tensor_lcm = Tensor.lcm(A, B)
        tensor_as_list = tensor_lcm.data if isinstance(tensor_lcm, Tensor) else [tensor_lcm]
        
        self.assertEqual(tensor_as_list, expected)
        
        # Test broadcasting with LCM
        A = [[2], [3], [4]]  # (3, 1)
        B = [[6, 9, 12]]     # (1, 3)
        
        expected = []
        for i in range(3):
            row = []
            for j in range(3):
                row.append(math.lcm(A[i][0], B[0][j]))
            expected.append(row)
        
        tensor_lcm = Tensor.lcm(A, B)
        tensor_as_np = np.array(tensor_lcm.data)
        expected_np = np.array(expected)
        
        self.assertTrue(np.array_equal(tensor_as_np, expected_np))

    def test_gt(self):
        """Test Tensor.__gt__ (greater than) operator."""
        # Test scalar comparison
        self.assertTrue(5 > 3)
        
        # Test array comparison
        threshold = 5
        A = [1, 3, 5, 7, 9]
        expected = [x > threshold for x in A]
        
        tensor_gt = Tensor(A) > threshold
        tensor_result = tensor_gt.data if isinstance(tensor_gt, Tensor) else [tensor_gt]
        
        self.assertEqual(tensor_result, expected)
        
        # Test 2D array comparison
        A_2d = [[1, 6, 3], [8, 2, 9]]
        threshold = 4
        expected_2d = [[x > threshold for x in row] for row in A_2d]
        
        tensor_gt_2d = Tensor(A_2d) > threshold
        tensor_result_2d = tensor_gt_2d.data
        
        self.assertEqual(tensor_result_2d, expected_2d)

    def test_lt(self):
        """Test Tensor.__lt__ (less than) operator."""
        # Test scalar comparison
        self.assertTrue(3 < 5)
        
        # Test array comparison
        threshold = 5
        A = [1, 3, 5, 7, 9]
        expected = [x < threshold for x in A]
        
        tensor_lt = Tensor(A) < threshold
        tensor_result = tensor_lt.data if isinstance(tensor_lt, Tensor) else [tensor_lt]
        
        self.assertEqual(tensor_result, expected)
        
        # Test 2D array comparison
        A_2d = [[1, 6, 3], [8, 2, 9]]
        threshold = 4
        expected_2d = [[x < threshold for x in row] for row in A_2d]
        
        tensor_lt_2d = Tensor(A_2d) < threshold
        tensor_result_2d = tensor_lt_2d.data
        
        self.assertEqual(tensor_result_2d, expected_2d)

    def test_logical_and(self):
        """Test Tensor.__and__ (logical AND) operator."""
        # Test scalar boolean AND
        self.assertEqual(True & True, True)
        self.assertEqual(True & False, False)
        self.assertEqual(False & True, False)
        self.assertEqual(False & False, False)
        
        # Test boolean + array
        A = [True, False, True, False]
        scalar_bool = True
        expected = [x and scalar_bool for x in A]
        
        tensor_and = Tensor(A) & scalar_bool
        tensor_result = tensor_and
        
        self.assertEqual(tensor_result, expected)
        
        # Test array + array logical AND (same shape)
        A = [True, False, True, False]
        B = [True, True, False, False]
        expected = [a and b for a, b in zip(A, B)]
        
        tensor_and = Tensor(A) & Tensor(B)
        tensor_result = tensor_and
        
        self.assertEqual(tensor_result, expected)
        
        # Test broadcasting with logical AND
        A = [[True], [False]]    # (2, 1)
        B = [[True, False]]      # (1, 2)
        
        
        tensor_and = Tensor(A) & Tensor(B)
        tensor_result = tensor_and
        
        # Test combining comparison operations with logical AND
        A = [1, 2, 3, 4]
        B = [1, 2, 3, 4]
        tensor_a = Tensor(A)
        tensor_b = Tensor(B)
        
        gt_result = tensor_a > 1  # [False, True, True, True]
        lt_result = tensor_b < 4  # [True, True, True, False]

        combined = gt_result & lt_result  # [False, True, True, False]
        expected = [False, True, True, False]
        self.assertEqual(combined, expected)
        
if __name__ == "__main__":
    unittest.main()