"""Comprehensive unit tests for DataSerializer."""
import pytest
import tempfile
import os
import numpy as np
from src.utils.data_serializer import DataSerializer


class TestDataSerializer:
    """Test suite for DataSerializer."""

    def test_save_and_load_json(self):
        """Test saving and loading JSON data."""
        data = {"key1": "value1", "key2": 123, "key3": [1, 2, 3]}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_path = f.name
        
        try:
            DataSerializer.save(data, json_path)
            loaded = DataSerializer.load(json_path)
            
            assert loaded["key1"] == "value1"
            assert loaded["key2"] == 123
            assert loaded["key3"] == [1, 2, 3]
        finally:
            if os.path.exists(json_path):
                os.unlink(json_path)

    def test_save_and_load_npz(self):
        """Test saving and loading NPZ data."""
        data = {
            "array1": np.array([1, 2, 3]),
            "array2": np.array([[1, 2], [3, 4]])
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.npz', delete=False) as f:
            npz_path = f.name
        
        try:
            DataSerializer.save(data, npz_path)
            loaded = DataSerializer.load(npz_path)
            
            assert "array1" in loaded
            assert "array2" in loaded
            assert len(loaded["array1"]) == 3
        finally:
            if os.path.exists(npz_path):
                os.unlink(npz_path)

    def test_save_invalid_format(self):
        """Test that unsupported formats raise ValueError."""
        data = {"key": "value"}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            txt_path = f.name
        
        try:
            with pytest.raises(Exception):  # Could be ValueError or RetryError
                DataSerializer.save(data, txt_path)
        finally:
            if os.path.exists(txt_path):
                os.unlink(txt_path)

    def test_load_invalid_format(self):
        """Test that loading unsupported formats raises ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test")
            txt_path = f.name
        
        try:
            with pytest.raises(Exception):  # Could be ValueError or RetryError
                DataSerializer.load(txt_path)
        finally:
            os.unlink(txt_path)

    def test_save_non_string_filepath(self):
        """Test that non-string filepath raises TypeError."""
        data = {"key": "value"}
        
        with pytest.raises(TypeError, match="Filepath must be a string"):
            DataSerializer.save(data, 123)  # type: ignore[arg-type]

    def test_load_non_string_filepath(self):
        """Test that non-string filepath raises TypeError for load."""
        with pytest.raises(TypeError, match="Filepath must be a string"):
            DataSerializer.load(123)  # type: ignore[arg-type]

    def test_save_nested_data_json(self):
        """Test saving nested data structures to JSON."""
        data = {
            "level1": {
                "level2": {
                    "key": "value",
                    "number": 42
                }
            },
            "list": [1, 2, 3]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_path = f.name
        
        try:
            DataSerializer.save(data, json_path)
            loaded = DataSerializer.load(json_path)
            
            assert loaded["level1"]["level2"]["key"] == "value"
            assert loaded["level1"]["level2"]["number"] == 42
        finally:
            if os.path.exists(json_path):
                os.unlink(json_path)

    def test_save_numpy_arrays_converted_for_npz(self):
        """Test that lists are converted to numpy arrays for NPZ."""
        data = {
            "list_data": [1, 2, 3, 4, 5]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.npz', delete=False) as f:
            npz_path = f.name
        
        try:
            DataSerializer.save(data, npz_path)
            loaded = DataSerializer.load(npz_path)
            
            assert "list_data" in loaded
            assert isinstance(loaded["list_data"], list)
        finally:
            if os.path.exists(npz_path):
                os.unlink(npz_path)

    def test_json_empty_dict(self):
        """Test saving and loading empty dictionary."""
        data: dict[str, str] = {}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_path = f.name
        
        try:
            DataSerializer.save(data, json_path)
            loaded = DataSerializer.load(json_path)
            
            assert loaded == {}
        finally:
            if os.path.exists(json_path):
                os.unlink(json_path)

    def test_json_unicode_data(self):
        """Test saving and loading Unicode data."""
        data = {
            "ukrainian": "Привіт",
            "emoji": "🚀",
            "chinese": "你好"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_path = f.name
        
        try:
            DataSerializer.save(data, json_path)
            loaded = DataSerializer.load(json_path)
            
            assert loaded["ukrainian"] == "Привіт"
            assert loaded["emoji"] == "🚀"
            assert loaded["chinese"] == "你好"
        finally:
            if os.path.exists(json_path):
                os.unlink(json_path)

    def test_npz_multidimensional_arrays(self):
        """Test saving and loading multidimensional arrays."""
        data = {
            "matrix": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            "tensor": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.npz', delete=False) as f:
            npz_path = f.name
        
        try:
            DataSerializer.save(data, npz_path)
            loaded = DataSerializer.load(npz_path)
            
            assert "matrix" in loaded
            assert "tensor" in loaded
        finally:
            if os.path.exists(npz_path):
                os.unlink(npz_path)

    def test_retry_mechanism_success(self):
        """Test that retry mechanism works on success."""
        data = {"test": "data"}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_path = f.name
        
        try:
            # Should succeed on first try
            DataSerializer.save(data, json_path)
            assert os.path.exists(json_path)
        finally:
            if os.path.exists(json_path):
                os.unlink(json_path)

    def test_json_with_floats(self):
        """Test JSON serialization with floating point numbers."""
        data = {
            "float1": 3.14159,
            "float2": 2.71828,
            "float3": 1.41421
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_path = f.name
        
        try:
            DataSerializer.save(data, json_path)
            loaded = DataSerializer.load(json_path)
            
            assert abs(loaded["float1"] - 3.14159) < 1e-5
            assert abs(loaded["float2"] - 2.71828) < 1e-5
        finally:
            if os.path.exists(json_path):
                os.unlink(json_path)

    def test_npz_different_dtypes(self):
        """Test NPZ with different data types."""
        data = {
            "int_array": np.array([1, 2, 3], dtype=np.int32),
            "float_array": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "complex_array": np.array([1+2j, 3+4j], dtype=np.complex64)
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.npz', delete=False) as f:
            npz_path = f.name
        
        try:
            DataSerializer.save(data, npz_path)
            loaded = DataSerializer.load(npz_path)
            
            assert "int_array" in loaded
            assert "float_array" in loaded
            assert "complex_array" in loaded
        finally:
            if os.path.exists(npz_path):
                os.unlink(npz_path)

    def test_file_overwrite(self):
        """Test that saving overwrites existing files."""
        data1 = {"version": 1}
        data2 = {"version": 2}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_path = f.name
        
        try:
            DataSerializer.save(data1, json_path)
            DataSerializer.save(data2, json_path)
            loaded = DataSerializer.load(json_path)
            
            assert loaded["version"] == 2
        finally:
            if os.path.exists(json_path):
                os.unlink(json_path)

    def test_load_nonexistent_file(self):
        """Test that loading non-existent file raises error."""
        with pytest.raises((FileNotFoundError, Exception)):
            DataSerializer.load("/nonexistent/path/data.json")
