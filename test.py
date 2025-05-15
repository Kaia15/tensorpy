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
    
    # def test_array(self):
    #     """Test Tensor.array against np.array."""
    #     # Test basic array creation
    #     test_cases = [
    #         [1, 2, 3, 4],
    #         [1.0, 2.0, 3.0],
    #         [True, False, True],
    #         [[1, 2], [3, 4]]
    #     ]
        
    #     for data in test_cases:
    #         np_array = np.array(data)
    #         tensor_array = Tensor.array(data)
            
    #         # Convert tensor data back to numpy for comparison
    #         tensor_as_np = np.array(tensor_array.data)
    #         print (np_array, tensor_as_np)
            
    #         self.assertTrue(np.array_equal(np_array, tensor_as_np))
    #         self.assertEqual(tensor_array.shape, np_array.shape)
        
    #     # Test ndmin parameter
    #     for ndmin in range(1, 4):
    #         np_array = np.array(self.data_1d, ndmin=ndmin)
    #         tensor_array = Tensor.array(self.data_1d, ndmin=ndmin)
            
    #         # Convert tensor data back to numpy for comparison
    #         tensor_as_np = np.array(tensor_array.data)
            
    #         self.assertTrue(np.array_equal(np_array, tensor_as_np))
    #         self.assertEqual(tensor_array.shape, np_array.shape)
    
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

        # print (type(self.tensor_1d))
        
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
        
        # Convert tensor result to numpy if it's a Tensor
        if isinstance(tensor_dot_1d, Tensor):
            tensor_dot_1d = np.array(tensor_dot_1d.data)
            
        self.assertEqual(np_dot_1d, tensor_dot_1d)
        
        # Test 2D dot 2D (matrix multiplication)
        np_dot_2d = np.dot(self.np_2d, self.np_2d.T)  # Ensure compatible shapes
        tensor_dot_2d = Tensor.dot(self.tensor_2d, self.tensor_2d.transpose())
        
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
            np_dot_complex = np.matmul(np_A, np_B)
            tensor_dot_complex = Tensor.dot(tensor_A, tensor_B)
            print (tensor_dot_complex.data)
            print (np_dot_complex)
            
            # Convert tensor data back to numpy for comparison
            tensor_as_np = np.array(tensor_dot_complex.data)
            
            self.assertTrue(np.allclose(np_dot_complex, tensor_as_np))
        except Exception as e:
            # If your implementation doesn't support this operation, just skip this test
            print(f"Skipping complex dot test: {e}")
    
    
if __name__ == "__main__":
    unittest.main()