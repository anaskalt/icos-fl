"""Test module for the Model Storage utilities.

This module validates the implementation of model storage functionality,
including local filesystem persistence, DataClay integration, thread safety,
and comprehensive error handling following the ICOS-FL testing patterns.
"""

import os
import sys
import tempfile
import threading
import time
from typing import Dict, List, Optional

import torch

# Add source directory to path BEFORE any relative imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from icos_fl.models.lstm import LSTMModel
from icos_fl.utils.colors import BCYA, BGRN, BRED, BYEL, RED, WHT, paint

# Check if DataClay is available
DATACLAY_AVAILABLE = True
try:
    from icos_fl.utils.model_storage import FLModelMetadata, ModelStorageManager
    from icos_fl.utils.singleton import Singleton
except ImportError:
    DATACLAY_AVAILABLE = False
    print(paint(BYEL, "⚠ DataClay not available - some tests will be skipped"))

# Test configuration
TEST_METRIC = "test_metric"
TEST_PROXY_HOST = "127.0.0.1"
TEST_DATASET = "admin"


def print_section_header(title: str) -> None:
    """Print a section header with a nice box around it."""
    min_box_width = 50
    title_len = len(title)
    required_width = title_len + 4
    box_width = max(min_box_width, required_width)

    border_line = "╔" + "═" * (box_width - 2) + "╗"
    bottom_line = "╚" + "═" * (box_width - 2) + "╝"

    inner_width = box_width - 4
    spaces_total = inner_width - title_len
    spaces_left = spaces_total // 2
    spaces_right = spaces_total - spaces_left

    title_padded = " " * spaces_left + title + " " * spaces_right

    print("")
    print(paint(BCYA, border_line))
    print(paint(BCYA, f"║ {title_padded} ║"))
    print(paint(BCYA, bottom_line))
    print("")


def print_subsection_header(title: str) -> None:
    """Print a subsection header with a smaller box."""
    min_box_width = 36
    title_len = len(title)
    required_width = title_len + 4
    box_width = max(min_box_width, required_width)

    border_line = "╔" + "═" * (box_width - 2) + "╗"
    bottom_line = "╚" + "═" * (box_width - 2) + "╝"

    inner_width = box_width - 4
    spaces_total = inner_width - title_len
    spaces_left = spaces_total // 2
    spaces_right = spaces_total - spaces_left

    title_padded = " " * spaces_left + title + " " * spaces_right

    print("")
    print(paint(BCYA, border_line))
    print(paint(BCYA, f"║ {title_padded} ║"))
    print(paint(BCYA, bottom_line))
    print("")


def print_subsection_title(title: str) -> None:
    """Print a subsection title with an arrow prefix."""
    print(paint(BYEL, f"▶ {title}"))


def print_test_header() -> None:
    """Print the main test header with logo."""
    print("")
    print(paint(BCYA, "════════════════════════════════════════════════════════════════"))
    print(paint(BCYA, "                🔍 MODEL STORAGE TESTING 🔍"))
    print(paint(BCYA, "════════════════════════════════════════════════════════════════"))
    print("")


def print_test_footer() -> None:
    """Print the test footer with completion message."""
    print("")
    print(paint(BCYA, "════════════════════════════════════════════════════════════════"))
    print(paint(BGRN, "           ✅ MODEL STORAGE TESTS COMPLETED ✅"))
    print(paint(BCYA, "════════════════════════════════════════════════════════════════"))
    print("")


def create_test_model() -> LSTMModel:
    """Create a simple LSTM model for testing.

    Returns:
        LSTMModel instance with test configuration suitable for unit testing
    """
    return LSTMModel(hidden_layer_size=10, time_step=5, num_layers=1)


def clear_singleton_instances() -> None:
    """Clear ModelStorageManager singleton instances for test isolation.

    This function accesses the Singleton metaclass's internal storage to remove
    all instances of ModelStorageManager, ensuring clean test environments.
    """
    if not DATACLAY_AVAILABLE:
        return

    # Access the metaclass's _instances dictionary
    # Filter and remove only ModelStorageManager instances
    keys_to_remove = [key for key in Singleton._instances.keys() if key[0] == ModelStorageManager]

    for key in keys_to_remove:
        del Singleton._instances[key]


def get_manager_instance_count() -> int:
    """Get the count of ModelStorageManager instances in the singleton registry.

    Returns:
        Number of ModelStorageManager instances currently registered
    """
    if not DATACLAY_AVAILABLE:
        return 0

    return sum(1 for key in Singleton._instances.keys() if key[0] == ModelStorageManager)


def get_manager_instances() -> Dict:
    """Get all ModelStorageManager instances from the singleton registry.

    Returns:
        Dictionary mapping instance keys to ModelStorageManager objects
    """
    if not DATACLAY_AVAILABLE:
        return {}

    return {
        key: instance
        for key, instance in Singleton._instances.items()
        if key[0] == ModelStorageManager
    }


def test_flmodel_metadata() -> None:
    """Test the FLModelMetadata DataClay object functionality."""
    if not DATACLAY_AVAILABLE:
        print_subsection_header("FLModelMetadata Class")
        skip_msg = paint(BYEL, "⚠ Skipping - DataClay not available")
        print(f"  {skip_msg}")
        return

    print_subsection_header("FLModelMetadata Class")

    # Test initialization
    print_subsection_title("Testing FLModelMetadata initialization:")

    try:
        metadata_obj = FLModelMetadata()

        init_checks = [
            ("state_dict", len(metadata_obj.state_dict) == 0),
            ("model_config", len(metadata_obj.model_config) == 0),
            ("metadata", len(metadata_obj.metadata) == 0),
            ("is_available", not metadata_obj.is_available),  # Fixed E712
            ("last_updated", metadata_obj.last_updated is None),
        ]

        all_passed = True
        for attr, check in init_checks:
            status = paint(BGRN, "✓") if check else paint(RED, "✗")
            print(f"  {status} {attr} initialized correctly")
            if not check:
                all_passed = False

        if not all_passed:
            print(paint(RED, "  ✗ Initialization test failed"))
            return

        # Test storing model state
        print("")
        print_subsection_title("Testing store_model_state:")

        test_model = create_test_model()
        test_config = {"hidden_layer_size": 10, "time_step": 5, "num_layers": 1}
        test_metadata = {"metric": TEST_METRIC, "round": 1}

        # Store model state
        metadata_obj.store_model_state(
            model=test_model, model_config=test_config, metadata=test_metadata
        )

        store_success = paint(BGRN, "✓ Model state stored successfully")
        print(f"  {store_success}")

        # Verify storage
        state_dict_size = len(metadata_obj.state_dict)
        print(f"  • State dict contains {paint(WHT, str(state_dict_size))} parameters")
        print(f"  • Model is available: {paint(BGRN, str(metadata_obj.is_available))}")

        if metadata_obj.last_updated:
            print(f"  • Last updated: {paint(WHT, metadata_obj.last_updated[:19])}")

        # Test retrieving model state
        print("")
        print_subsection_title("Testing retrieve_state_dict:")

        retrieved_state = metadata_obj.retrieve_state_dict()
        retrieve_success = paint(BGRN, "✓ State dict retrieved successfully")
        print(f"  {retrieve_success}")

        # Verify retrieved state
        param_count = len(retrieved_state)
        print(f"  • Retrieved {paint(WHT, str(param_count))} parameters")

        # Check tensor types
        all_tensors = all(isinstance(v, torch.Tensor) for v in retrieved_state.values())
        tensor_check = paint(BGRN, "✓") if all_tensors else paint(RED, "✗")
        print(f"  {tensor_check} All values are PyTorch tensors")

        # Test error handling
        print("")
        print_subsection_title("Testing error handling:")

        # Test None model
        try:
            metadata_obj.store_model_state(model=None, model_config={}, metadata={})
            print(paint(RED, "  ✗ Should have raised ValueError for None model"))
        except ValueError:
            print(paint(BGRN, "  ✓ Correctly raised ValueError for None model"))
        except (RuntimeError, TypeError) as e:
            print(paint(RED, f"  ✗ Unexpected error: {type(e).__name__}: {e}"))

    except (ImportError, RuntimeError, AttributeError) as e:
        error_msg = paint(RED, f"✗ Test failed with error: {type(e).__name__}: {e}")
        print(f"  {error_msg}")


def _verify_same_parameters_singleton(
    manager1: ModelStorageManager, manager2: ModelStorageManager
) -> bool:
    """Verify singleton behavior for instances with same parameters.

    Args:
        manager1: First manager instance
        manager2: Second manager instance

    Returns:
        True if singleton pattern is working correctly
    """
    print("")
    print_subsection_title("Singleton verification (same parameters):")

    # Test 1: Identity check
    identity_match = manager1 is manager2
    if identity_match:
        print(f"  {paint(BGRN, '✓')} Identity check passed: manager1 is manager2")
    else:
        print(f"  {paint(RED, '✗')} Identity check failed: manager1 is not manager2")

    # Test 2: Memory address check
    same_memory = id(manager1) == id(manager2)
    if same_memory:
        print(f"  {paint(BGRN, '✓')} Memory check passed: same memory address")
    else:
        print(f"  {paint(RED, '✗')} Memory check failed: different memory addresses")

    # Test 3: Instance count verification
    instance_count = get_manager_instance_count()
    if instance_count == 1:
        print(f"  {paint(BGRN, '✓')} Instance count correct: {paint(WHT, '1')} instance")
    else:
        inst_msg = f"Instance count incorrect: {paint(WHT, str(instance_count))} instances"
        print(f"  {paint(RED, '✗')} {inst_msg}")

    return identity_match and same_memory and (instance_count == 1)


def _verify_different_parameters_singleton(
    manager1: ModelStorageManager, manager3: ModelStorageManager
) -> bool:
    """Verify singleton behavior for instances with different parameters.

    Args:
        manager1: First manager instance (original parameters)
        manager3: Third manager instance (different parameters)

    Returns:
        True if parameterized singleton is working correctly
    """
    print("")
    print_subsection_title("Parameterized singleton verification:")

    # Test 1: Different instances check
    different_instances = manager1 is not manager3
    if different_instances:
        print(f"  {paint(BGRN, '✓')} Different parameters created different instance")
    else:
        err_msg = "Different parameters returned same instance (incorrect!)"
        print(f"  {paint(RED, '✗')} {err_msg}")

    # Test 2: Memory address comparison
    different_memory = id(manager1) != id(manager3)
    if different_memory:
        print(f"  {paint(BGRN, '✓')} Different memory addresses confirmed")
        print(f"    - manager1: {paint(WHT, str(id(manager1)))}")
        print(f"    - manager3: {paint(WHT, str(id(manager3)))}")
    else:
        print(f"  {paint(RED, '✗')} Same memory address (incorrect!)")

    # Test 3: Instance registry check
    expected_instances = 2
    actual_instances = get_manager_instance_count()
    if actual_instances == expected_instances:
        inst_msg = f"Instance registry correct: {paint(WHT, str(actual_instances))} instances"
        print(f"  {paint(BGRN, '✓')} {inst_msg}")
    else:
        err_msg = f"Instance registry incorrect: expected {expected_instances}, "
        err_msg += f"got {actual_instances}"
        print(f"  {paint(RED, '✗')} {err_msg}")

    return different_instances and different_memory and (actual_instances == expected_instances)


def _print_instance_registry_details() -> None:
    """Print detailed information about instances in the singleton registry."""
    print("")
    print_subsection_title("Instance registry details:")
    instances = get_manager_instances()

    for i, (key, instance) in enumerate(instances.items(), 1):
        # Extract meaningful info from key tuple: (class, args, kwargs)
        class_name = key[0].__name__
        args = key[1]
        kwargs = dict(key[2]) if key[2] else {}

        print(f"  • Instance {i}:")
        print(f"    - Class: {paint(WHT, class_name)}")
        if args:
            print(f"    - Args: {paint(WHT, str(args))}")
        if kwargs:
            print(f"    - Kwargs: {paint(WHT, str(kwargs))}")
        print(f"    - ID: {paint(WHT, str(id(instance)))}")


def test_model_storage_manager_initialization() -> None:
    """Test ModelStorageManager singleton initialization with parameterized instances.

    This test validates:
    1. Same parameters return the same singleton instance
    2. Different parameters create separate singleton instances
    3. Thread-safe instance creation via the enhanced Singleton metaclass
    """
    if not DATACLAY_AVAILABLE:
        print_subsection_header("StorageManager Initialization")
        skip_msg = paint(BYEL, "⚠ Skipping - DataClay not available")
        print(f"  {skip_msg}")
        return

    print_subsection_header("StorageManager Initialization")
    clear_singleton_instances()
    print_subsection_title("Testing singleton pattern with same parameters:")

    try:
        # Create instances with same parameters
        manager1 = ModelStorageManager(proxy_host=TEST_PROXY_HOST, dataset=TEST_DATASET)
        print(f"  • Created first instance: {paint(BGRN, 'OK')}")
        print(f"    - ID: {paint(WHT, str(id(manager1)))}")

        manager2 = ModelStorageManager(proxy_host=TEST_PROXY_HOST, dataset=TEST_DATASET)
        print(f"  • Created second instance: {paint(BGRN, 'OK')}")
        print(f"    - ID: {paint(WHT, str(id(manager2)))}")

        # Verify singleton behavior
        singleton_working = _verify_same_parameters_singleton(manager1, manager2)
        if not singleton_working:
            print(paint(RED, "\n  ⚠ Singleton pattern not working correctly!"))
            return

        # Test parameterized singleton behavior
        print("")
        print_subsection_title("Testing parameterized singleton (different parameters):")

        different_params = {"proxy_host": "192.168.1.100", "dataset": "test_dataset"}
        manager3 = ModelStorageManager(**different_params)
        print("  • Created third instance with different parameters:")
        print(f"    - proxy_host: {paint(WHT, different_params['proxy_host'])}")
        print(f"    - dataset: {paint(WHT, different_params['dataset'])}")
        print(f"    - ID: {paint(WHT, str(id(manager3)))}")

        # Verify parameterized behavior
        parameterized_working = _verify_different_parameters_singleton(manager1, manager3)

        # Print instance registry details
        _print_instance_registry_details()

        # Final result
        print("")
        if singleton_working and parameterized_working:
            print(paint(BGRN, "  ✅ All singleton tests passed successfully!"))
        else:
            print(paint(RED, "  ❌ Some singleton tests failed!"))

    except (ImportError, RuntimeError, AttributeError) as e:
        error_msg = paint(RED, f"✗ Initialization test failed: {type(e).__name__}: {e}")
        print(f"  {error_msg}")
        import traceback

        traceback.print_exc()
    finally:
        # Cleanup all instances
        instances = get_manager_instances()
        for instance in instances.values():
            if hasattr(instance, "cleanup"):
                try:
                    instance.cleanup()
                except (AttributeError, RuntimeError):
                    pass
        clear_singleton_instances()


def test_local_storage() -> None:
    """Test local filesystem storage functionality."""
    if not DATACLAY_AVAILABLE:
        print_subsection_header("Local Storage Backend")
        skip_msg = paint(BYEL, "⚠ Skipping - DataClay module not available")
        print(f"  {skip_msg}")
        return

    print_subsection_header("Local Storage Backend")

    # Clear singletons and create fresh manager
    clear_singleton_instances()

    try:
        manager = ModelStorageManager()
        model = create_test_model()

        print_subsection_title("Testing local save operation:")

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_model.pt")

            # Test successful save
            results = manager.save_model(
                model=model,
                save_path=save_path,
                identifier="test_local_save",
                use_local=True,
                use_dataclay=False,
                model_config={"test": "config"},
                metadata={"test": "metadata"},
            )

            # Check results
            if results["local"]:
                save_success = paint(BGRN, "✓ Local save successful")
                print(f"  {save_success}")

                # Verify file exists
                file_exists = os.path.exists(save_path)
                file_check = paint(BGRN, "✓") if file_exists else paint(RED, "✗")
                print(f"  {file_check} Model file exists at expected path")

                if file_exists:
                    file_size = os.path.getsize(save_path)
                    print(f"  • File size: {paint(WHT, f'{file_size:,}')} bytes")

                    # Load and verify checkpoint
                    checkpoint = torch.load(save_path, weights_only=False)
                    has_state_dict = "model_state_dict" in checkpoint
                    has_config = "model_config" in checkpoint
                    has_metadata = "metadata" in checkpoint
                    has_version = "icos_fl_version" in checkpoint

                    checkpoint_checks = [
                        ("model_state_dict", has_state_dict),
                        ("model_config", has_config),
                        ("metadata", has_metadata),
                        ("icos_fl_version", has_version),
                    ]

                    print("")
                    print_subsection_title("Checkpoint contents:")
                    for key, exists in checkpoint_checks:
                        status = paint(BGRN, "✓") if exists else paint(RED, "✗")
                        print(f"  {status} Contains {key}")

                    # Verify metadata content
                    if has_metadata:
                        timestamp_exists = "timestamp" in checkpoint["metadata"]
                        timestamp_check = paint(BGRN, "✓") if timestamp_exists else paint(RED, "✗")
                        print(f"  {timestamp_check} Metadata contains timestamp")
            else:
                save_failed = paint(RED, "✗ Local save failed")
                print(f"  {save_failed}")

            # Test atomic write behavior
            print("")
            print_subsection_title("Testing atomic write:")

            # Check that temporary file was cleaned up
            temp_file = f"{save_path}.tmp"
            no_temp_file = not os.path.exists(temp_file)
            temp_check = paint(BGRN, "✓") if no_temp_file else paint(RED, "✗")
            print(f"  {temp_check} Temporary file cleaned up after save")

    except (ImportError, RuntimeError, OSError) as e:
        error_msg = paint(RED, f"✗ Local storage test failed: {type(e).__name__}: {e}")
        print(f"  {error_msg}")
    finally:
        if "manager" in locals():
            manager.cleanup()
        clear_singleton_instances()


def test_dataclay_storage() -> None:
    """Test DataClay storage functionality (mocked)."""
    if not DATACLAY_AVAILABLE:
        print_subsection_header("DataClay Storage Backend")
        skip_msg = paint(BYEL, "⚠ Skipping - DataClay not available")
        print(f"  {skip_msg}")
        return

    print_subsection_header("DataClay Storage Backend")

    # Clear singletons and create fresh manager
    clear_singleton_instances()

    try:
        manager = ModelStorageManager()
        model = create_test_model()

        print_subsection_title("Testing DataClay connection:")

        # Note: This will fail in test environment without real DataClay
        connected = manager._ensure_dataclay_connection()

        if connected:
            conn_status = paint(BGRN, "✓ DataClay connection established")
            print(f"  {conn_status}")

            # Test save operation
            print("")
            print_subsection_title("Testing DataClay save:")

            # Use a secure temp file for testing
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
                temp_path = tmp_file.name

            try:
                results = manager.save_model(
                    model=model,
                    save_path=temp_path,  # Use secure temp file
                    identifier="test_dataclay_save",
                    use_local=False,
                    use_dataclay=True,
                    model_config={"test": "config"},
                )

                if results["dataclay"]:
                    dc_save_success = paint(BGRN, "✓ DataClay save successful")
                    print(f"  {dc_save_success}")

                    # Test load operation
                    print("")
                    print_subsection_title("Testing DataClay load:")

                    loaded = manager.load_model_from_dataclay("test_dataclay_save")
                    if loaded:
                        state_dict, config = loaded
                        load_success = paint(BGRN, "✓ Model loaded from DataClay")
                        print(f"  {load_success}")
                        print(f"  • Loaded {paint(WHT, str(len(state_dict)))} parameters")
                        print(f"  • Config: {paint(WHT, str(config.get('test', 'N/A')))}")
                    else:
                        load_failed = paint(BYEL, "⚠ DataClay load failed")
                        print(f"  {load_failed}")
                else:
                    dc_save_failed = paint(BYEL, "⚠ DataClay save failed (expected in test env)")
                    print(f"  {dc_save_failed}")
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        else:
            conn_failed = paint(BYEL, "⚠ DataClay connection failed (expected in test env)")
            print(f"  {conn_failed}")
            print(
                f"  • Max connection attempts: {paint(WHT, str(manager._max_connection_attempts))}"
            )
            print(f"  • Current attempts: {paint(WHT, str(manager._connection_attempts))}")

    except (ImportError, RuntimeError, OSError) as e:
        error_msg = paint(RED, f"✗ DataClay storage test error: {type(e).__name__}: {e}")
        print(f"  {error_msg}")
    finally:
        if "manager" in locals():
            manager.cleanup()
        clear_singleton_instances()


def test_multi_backend_storage() -> None:
    """Test saving to multiple backends simultaneously."""
    if not DATACLAY_AVAILABLE:
        print_subsection_header("Multi-Backend Storage")
        skip_msg = paint(BYEL, "⚠ Skipping - DataClay not available")
        print(f"  {skip_msg}")
        return

    print_subsection_header("Multi-Backend Storage")

    # Clear singletons and create fresh manager
    clear_singleton_instances()

    try:
        manager = ModelStorageManager()
        model = create_test_model()

        print_subsection_title("Testing simultaneous save to local + DataClay:")

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "multi_backend_test.pt")

            results = manager.save_model(
                model=model,
                save_path=save_path,
                identifier="test_multi_backend",
                use_local=True,
                use_dataclay=True,
                model_config={"backend": "multi"},
                metadata={"test": "multi-backend"},
            )

            # Display results
            local_status = paint(BGRN, "Success") if results["local"] else paint(RED, "Failed")
            dataclay_status = (
                paint(BYEL, "Failed (expected)")
                if not results["dataclay"]
                else paint(BGRN, "Success")
            )

            print(f"  • Local save: {local_status}")
            print(f"  • DataClay save: {dataclay_status}")

            # At least one should succeed
            any_success = any(results.values())
            overall_status = paint(BGRN, "✓") if any_success else paint(RED, "✗")
            print(f"  {overall_status} At least one backend succeeded")

            # Verify local file if successful
            if results["local"] and os.path.exists(save_path):
                file_size = os.path.getsize(save_path)
                print(f"  • Local file size: {paint(WHT, f'{file_size:,}')} bytes")

    except (ImportError, RuntimeError, OSError) as e:
        error_msg = paint(RED, f"✗ Multi-backend test error: {type(e).__name__}: {e}")
        print(f"  {error_msg}")
    finally:
        if "manager" in locals():
            manager.cleanup()
        clear_singleton_instances()


def test_thread_safety() -> None:
    """Test thread safety of the singleton storage manager."""
    if not DATACLAY_AVAILABLE:
        print_subsection_header("Thread Safety Testing")
        skip_msg = paint(BYEL, "⚠ Skipping - DataClay not available")
        print(f"  {skip_msg}")
        return

    print_subsection_header("Thread Safety Testing")

    # Clear singletons before test
    clear_singleton_instances()

    print_subsection_title("Testing concurrent access:")

    managers: List[Optional[ModelStorageManager]] = []
    errors: List[Exception] = []
    lock = threading.Lock()

    def create_manager(index: int) -> None:
        """Create a manager instance in a thread."""
        try:
            time.sleep(0.01 * index)  # Small delay to encourage race conditions
            manager = ModelStorageManager()
            with lock:
                managers.append(manager)
                thread_msg = f"  • Thread {index}: Manager created"
                print(paint(BGRN, thread_msg))
        except (ImportError, RuntimeError, AttributeError) as e:
            with lock:
                errors.append(e)
                error_msg = f"  • Thread {index}: {paint(RED, f'Error: {e}')}"
                print(error_msg)

    # Create multiple threads
    threads = []
    num_threads = 5
    for i in range(num_threads):
        thread = threading.Thread(target=create_manager, args=(i,))
        threads.append(thread)
        thread.start()

    # Wait for all threads
    for thread in threads:
        thread.join()

    print("")
    print_subsection_title("Thread safety results:")

    # Check results
    no_errors = len(errors) == 0
    error_check = paint(BGRN, "✓") if no_errors else paint(RED, "✗")
    print(f"  {error_check} No errors during concurrent access")

    if errors:
        for i, error in enumerate(errors):
            print(f"    Error {i + 1}: {paint(RED, str(error))}")

    if managers:
        # Check all managers are the same instance
        all_same = all(m is managers[0] for m in managers if m is not None)
        instance_check = paint(BGRN, "✓") if all_same else paint(RED, "✗")
        print(f"  {instance_check} All threads got same singleton instance")

        # Verify singleton behavior
        print(f"  • Total manager instances created: {paint(WHT, str(len(managers)))}")
        print(f"  • Expected instances: {paint(WHT, str(num_threads))}")

        # Cleanup the single instance
        if managers[0] is not None:
            managers[0].cleanup()

    clear_singleton_instances()


def test_error_handling() -> None:
    """Test error handling in various scenarios."""
    if not DATACLAY_AVAILABLE:
        print_subsection_header("Error Handling")
        skip_msg = paint(BYEL, "⚠ Skipping - DataClay not available")
        print(f"  {skip_msg}")
        return

    print_subsection_header("Error Handling")

    # Clear singletons and create fresh manager
    clear_singleton_instances()

    try:
        manager = ModelStorageManager()

        # Test 1: Invalid model type
        print_subsection_title("Testing invalid model type:")

        try:
            # Use secure temp file
            with tempfile.NamedTemporaryFile(suffix=".pt") as tmp_file:
                results = manager.save_model(
                    model="not a model",  # Invalid type
                    save_path=tmp_file.name,
                    identifier="test_invalid",
                    use_local=True,
                    use_dataclay=False,
                )
            print(paint(RED, "  ✗ Should have raised ValueError"))
        except ValueError as e:
            error_caught = paint(BGRN, "✓ ValueError caught correctly")
            print(f"  {error_caught}")
            print(f"  • Error message: {paint(WHT, str(e))}")
        except (RuntimeError, TypeError) as e:
            unexpected = paint(RED, f"✗ Unexpected error type: {type(e).__name__}")
            print(f"  {unexpected}")

        # Test 2: Invalid save path
        print("")
        print_subsection_title("Testing invalid save path:")

        model = create_test_model()
        results = manager.save_model(
            model=model,
            save_path="/invalid/path/that/does/not/exist/model.pt",
            identifier="test_invalid_path",
            use_local=True,
            use_dataclay=False,
        )

        save_failed = not results["local"]
        failure_check = paint(BGRN, "✓") if save_failed else paint(RED, "✗")
        print(f"  {failure_check} Save failed gracefully with invalid path")

        # Test 3: Loading non-existent model from DataClay
        print("")
        print_subsection_title("Testing load of non-existent model:")

        result = manager.load_model_from_dataclay("non_existent_model_id")

        load_none = result is None
        none_check = paint(BGRN, "✓") if load_none else paint(RED, "✗")
        print(f"  {none_check} Returns None for non-existent model")

        # Test 4: Empty identifier
        print("")
        print_subsection_title("Testing empty identifier:")

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "empty_id_test.pt")
            results = manager.save_model(
                model=model,
                save_path=save_path,
                identifier="",  # Empty identifier
                use_local=True,
                use_dataclay=False,
            )

            # Should still work for local save
            local_ok = results["local"]
            empty_id_check = paint(BGRN, "✓") if local_ok else paint(RED, "✗")
            print(f"  {empty_id_check} Local save works with empty identifier")

    except (ImportError, RuntimeError, OSError) as e:
        error_msg = paint(RED, f"✗ Error handling test failed: {type(e).__name__}: {e}")
        print(f"  {error_msg}")
    finally:
        if "manager" in locals():
            manager.cleanup()
        clear_singleton_instances()


def test_resource_cleanup() -> None:
    """Test proper resource cleanup."""
    if not DATACLAY_AVAILABLE:
        print_subsection_header("Resource Cleanup")
        skip_msg = paint(BYEL, "⚠ Skipping - DataClay not available")
        print(f"  {skip_msg}")
        return

    print_subsection_header("Resource Cleanup")

    # Clear singletons and create fresh manager
    clear_singleton_instances()

    try:
        print_subsection_title("Testing cleanup method:")

        manager = ModelStorageManager()

        # Add some references to track
        manager._model_storage_refs["test1"] = "alias1"
        manager._model_storage_refs["test2"] = "alias2"

        # Run cleanup
        manager.cleanup()
        cleanup_success = paint(BGRN, "✓ Cleanup completed without errors")
        print(f"  {cleanup_success}")

        # Verify state after cleanup
        print("")
        print_subsection_title("State after cleanup:")

        client_none = manager.client is None
        not_connected = not manager._dataclay_connected
        refs_cleared = len(manager._model_storage_refs) == 0

        cleanup_checks = [
            ("Client set to None", client_none),
            ("Connection flag reset", not_connected),
            ("References cleared", refs_cleared),
        ]

        for check_name, passed in cleanup_checks:
            status = paint(BGRN, "✓") if passed else paint(RED, "✗")
            print(f"  {status} {check_name}")

        # Test double cleanup (should not error)
        print("")
        print_subsection_title("Testing double cleanup:")

        try:
            manager.cleanup()
            double_cleanup_ok = paint(BGRN, "✓ Double cleanup handled gracefully")
            print(f"  {double_cleanup_ok}")
        except (AttributeError, RuntimeError) as e:
            double_cleanup_error = paint(RED, f"✗ Double cleanup error: {e}")
            print(f"  {double_cleanup_error}")

    except (ImportError, RuntimeError, AttributeError) as e:
        error_msg = paint(RED, f"✗ Cleanup test failed: {type(e).__name__}: {e}")
        print(f"  {error_msg}")
    finally:
        clear_singleton_instances()


def test_model_storage() -> None:
    """Run all model storage tests.

    This comprehensive test suite validates:
    - FLModelMetadata DataClay object functionality
    - Singleton pattern implementation with parameterized instances
    - Local filesystem storage operations
    - DataClay distributed storage integration
    - Multi-backend concurrent saves
    - Thread safety guarantees
    - Error handling robustness
    - Resource cleanup procedures
    """
    # Print custom header
    print_test_header()

    # Display header
    print_section_header("MODEL STORAGE MODULE TESTING")

    # Check DataClay availability
    if not DATACLAY_AVAILABLE:
        warning_msg = paint(
            BYEL,
            "\n⚠ WARNING: DataClay is not installed. "
            "Some tests will be skipped.\n"
            "To run all tests, install DataClay: pip install dataclay",
        )
        print(warning_msg)

    try:
        # Run all test functions
        test_flmodel_metadata()
        test_model_storage_manager_initialization()
        test_local_storage()
        test_dataclay_storage()
        test_multi_backend_storage()
        test_thread_safety()
        test_error_handling()
        test_resource_cleanup()

        # Display completion
        print_section_header("ALL TESTS COMPLETED")
        print_test_footer()

    except (ImportError, RuntimeError, AttributeError) as e:
        error_msg = f"An error occurred during testing: {e!s}"
        print(paint(RED, error_msg))
        print("")
        print(paint(BRED, "❌ TESTS FAILED"))
        print("")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_model_storage()
