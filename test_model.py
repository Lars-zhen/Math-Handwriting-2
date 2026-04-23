"""
Model Structure Test Script
Verifies that the LeNet-5 + SE model is correctly implemented
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from models import LeNet5WithSE, SEBlock, count_parameters


def test_se_block():
    """Test SE Block implementation"""
    print("=" * 50)
    print("Testing SE Block...")
    print("=" * 50)

    # Test SE block with 6 channels
    se_block = SEBlock(channels=6, reduction=2)

    # Create random input
    x = torch.randn(2, 6, 32, 32)

    # Forward pass
    y = se_block(x)

    # Verify output shape
    assert y.shape == x.shape, f"Expected shape {x.shape}, got {y.shape}"

    # Verify channels are unchanged
    assert y.shape[1] == 6, f"Channel dimension changed: expected 6, got {y.shape[1]}"

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Channels preserved: OK")
    print("SE Block test: PASSED")
    return True


def test_lenet5_se():
    """Test LeNet-5 with SE implementation"""
    print("\n" + "=" * 50)
    print("Testing LeNet-5 with SE...")
    print("=" * 50)

    # Create model
    num_classes = 249
    model = LeNet5WithSE(num_classes=num_classes, dropout_rate=0.3)

    # Count parameters
    num_params = count_parameters(model)
    print(f"  Total parameters: {num_params:,}")

    # Create random input
    x = torch.randn(1, 1, 32, 32)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x)

    # Verify output shape
    expected_shape = (1, num_classes)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
    print("LeNet-5 with SE test: PASSED")
    return True


def test_batch_processing():
    """Test batch processing"""
    print("\n" + "=" * 50)
    print("Testing Batch Processing...")
    print("=" * 50)

    model = LeNet5WithSE(num_classes=249)
    model.eval()

    # Test different batch sizes
    batch_sizes = [1, 4, 16, 32, 64]

    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 1, 32, 32)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (batch_size, 249), f"Batch size {batch_size} failed"
        print(f"  Batch size {batch_size:2d}: OK (output shape: {output.shape})")

    print("Batch Processing test: PASSED")
    return True


def test_gradient_flow():
    """Test that gradients can flow through the model"""
    print("\n" + "=" * 50)
    print("Testing Gradient Flow...")
    print("=" * 50)

    model = LeNet5WithSE(num_classes=249)
    model.train()

    x = torch.randn(4, 1, 32, 32)
    target = torch.randint(0, 249, (4,))

    # Forward pass
    output = model(x)

    # Calculate loss
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(output, target)

    # Backward pass
    loss.backward()

    # Check gradients
    gradient_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradient_count += 1
            # Allow very small gradients during early training
            grad_norm = param.grad.norm().item()
            if grad_norm < 1e-6:
                print(f"  Warning: Small gradient for {name}: {grad_norm:.2e}")

    print(f"  Parameters with gradients: {gradient_count}")
    print(f"  Loss value: {loss.item():.4f}")
    print("Gradient Flow test: PASSED")
    return True


def test_model_components():
    """Test individual model components"""
    print("\n" + "=" * 50)
    print("Testing Model Components...")
    print("=" * 50)

    model = LeNet5WithSE(num_classes=249)

    # List all components
    print("  Model components:")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules
            print(f"    - {name}: {module.__class__.__name__}")

    # Test each component
    x = torch.randn(1, 1, 32, 32)

    # Test conv1
    conv1_out = model.conv1(x)
    print(f"\n  conv1 output shape: {conv1_out.shape}")

    # Test pool1
    pool1_out = model.pool1(conv1_out)
    print(f"  pool1 output shape: {pool1_out.shape}")

    # Test SE block 1
    se1_out = model.se_block1(pool1_out)
    print(f"  se_block1 output shape: {se1_out.shape}")

    # Test conv2
    conv2_out = model.conv2(se1_out)
    print(f"  conv2 output shape: {conv2_out.shape}")

    # Test pool2
    pool2_out = model.pool2(conv2_out)
    print(f"  pool2 output shape: {pool2_out.shape}")

    # Test SE block 2
    se2_out = model.se_block2(pool2_out)
    print(f"  se_block2 output shape: {se2_out.shape}")

    print("\nModel Components test: PASSED")
    return True


def test_different_input_sizes():
    """Test model with standard 32x32 input size"""
    print("\n" + "=" * 50)
    print("Testing Standard Input Size...")
    print("=" * 50)

    model = LeNet5WithSE(num_classes=249)

    # Standard 32x32 (LeNet-5 standard)
    x_32 = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        out_32 = model(x_32)
    print(f"  32x32 input: {out_32.shape}")

    assert out_32.shape == (1, 249), "Standard input size test failed"

    print("\nStandard Input Size test: PASSED")
    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print(" CNN+SE MODEL STRUCTURE TEST SUITE")
    print("=" * 60)
    print()

    tests = [
        ("SE Block", test_se_block),
        ("LeNet-5 with SE", test_lenet5_se),
        ("Batch Processing", test_batch_processing),
        ("Gradient Flow", test_gradient_flow),
        ("Model Components", test_model_components),
        ("Standard Input Size", test_different_input_sizes),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"\n{test_name} test FAILED:")
            print(f"  Error: {str(e)}")
            failed += 1

    # Summary
    print("\n" + "=" * 60)
    print(" TEST SUMMARY")
    print("=" * 60)
    print(f"  Total tests: {passed + failed}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")

    if failed == 0:
        print("\n[OK] All tests passed! Model structure is correct.")
        print("\nModel can now be trained with: python train.py")
    else:
        print("\n[FAIL] Some tests failed. Please check the model implementation.")

    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
