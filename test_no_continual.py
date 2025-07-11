#!/usr/bin/env python3
"""
Test script to verify that continual_method='no' works without DST functionality
"""

import os
import sys
import subprocess

def test_no_continual_method():
    """Test training with continual_method='no'"""
    
    print("Testing continual_method='no' configuration...")
    
    # Test command with continual_method='no'
    cmd = [
        "python", "main.py",
        "default_setting=./configs/default/mvtecad_35.yaml",
        "model_setting=./configs/model/cfgcad.yaml", 
        "DEFAULT.exp_name=test_no_continual",
        "CONTINUAL.continual=false",
        "CONTINUAL.method.name=no",
        "TRAIN.epochs=1",  # Just 1 epoch for testing
        "TRAIN.wandb.use=false"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run the command and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        if result.returncode == 0:
            print("✓ Test passed: Training completed successfully with continual_method='no'")
            
            # Check for expected log messages
            output = result.stdout + result.stderr
            if "Continual method is 'no' - using standard training without DST" in output:
                print("✓ Confirmed: DST disabled correctly")
            else:
                print("⚠ Warning: DST disable message not found in output")
                
            if "Standard training mode - no continual learning features" in output:
                print("✓ Confirmed: Standard training mode activated")
            else:
                print("⚠ Warning: Standard training mode message not found")
                
        else:
            print(f"✗ Test failed: Command returned exit code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Test failed: Command timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"✗ Test failed: Exception occurred: {e}")
        return False
        
    return True

if __name__ == "__main__":
    success = test_no_continual_method()
    sys.exit(0 if success else 1) 