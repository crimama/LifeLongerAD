# PGPT (Prototype-Guided Prompt Tuning) for Continual Learning

## Overview

PGPT implements a continual learning approach using **class-level prompts** that allows fine-grained control over different classes within the same task. Unlike traditional task-based approaches, PGPT can handle multiple classes per task and apply different prompts for each class even within the same batch.

## Key Features

### 1. **Class-Level Prompt Management**
- **Individual Class Prompts**: Each class gets its own set of prompts across all decoder layers
- **Batch-Level Adaptation**: Different samples in the same batch can use different prompts based on their class labels
- **Dynamic Prompt Selection**: Prompts are selected and applied dynamically during training and inference

### 2. **Multi-Class Task Support**
- **Flexible Task Structure**: A single task can contain multiple classes
- **Concurrent Learning**: Learn prompts for multiple classes simultaneously
- **Class-Specific Prototypes**: Each class maintains its own prototype for K-NN selection

### 3. **Enhanced Architecture**
- **BatchPromptEmbedding**: Custom embedding layer that handles batch-wise prompts
- **Automatic Device Management**: Prompts are automatically moved to the correct device
- **Memory Efficient**: Only stores prompts for learned classes

## Architecture Components

### Core Classes

1. **CL_PromptInput**: Main continual learning manager with class-level prompt support
2. **BatchPromptEmbedding**: Custom embedding layer for batch-wise prompt application

### Key Methods

- `initialize_prompt_for_class()`: Creates prompts for a new class
- `add_class_to_task()`: Adds a class to the current task
- `inject_prompts_for_batch()`: Applies class-specific prompts for each sample in batch
- `calculate_prototype()`: Computes class prototypes from training data
- `select_prompt_by_knn()`: K-NN selection during cross-class inference
- `get_class_from_label()`: Maps class labels to class names

### Training Workflow

#### Phase 1: Task Setup
1. **Multi-Class Initialization**: Initialize prompts for all classes in the task
2. **Active Class Tracking**: Track which classes are active in current task
3. **Primary Class Selection**: Set primary class for logging and checkpointing

#### Phase 2: Class-Aware Training
1. **Batch Processing**: For each batch, identify class labels
2. **Dynamic Prompt Injection**: Apply appropriate prompts for each sample
3. **Concurrent Learning**: Train prompts for multiple classes simultaneously
4. **Prototype Calculation**: Calculate prototypes for all classes after training

#### Phase 3: Cross-Class Inference
1. **Class-Specific Application**: Use class labels when available
2. **K-NN Fallback**: Use prototype-based selection for unknown classes
3. **Prompt Injection**: Apply selected prompts dynamically

## Configuration

### Basic Class-Level PGPT Configuration
```yaml
CONTINUAL:
  use_pgpt: true                    # Enable PGPT
  prompt_dim: 256                   # Prompt vector dimension
  k_neighbors: 1                    # K-NN neighbors for cross-class inference
  
  method:
    name: "spacenet"
    params:
      use_pgpt: true               # Enable in method params
      prompt_dim: 256              # Must match main config
```

### Multi-Class Task Configuration
```python
# Example: 3 tasks with different numbers of classes
loader_dict = {
    ["bottle", "cable"]: {           # Task 1: 2 classes
        "train": dataloader1, 
        "test": testloader1
    },
    ["capsule"]: {                   # Task 2: 1 class
        "train": dataloader2, 
        "test": testloader2
    },
    ["hazelnut", "metal_nut", "pill"]: {  # Task 3: 3 classes
        "train": dataloader3, 
        "test": testloader3
    }
}
```

## Usage Examples

### 1. **Single-Class Task (Traditional)**
```python
# Configuration for single-class tasks
loader_dict = {
    "bottle": {"train": train_loader, "test": test_loader},
    "cable": {"train": train_loader, "test": test_loader},
    # Each class is a separate task
}

cl_manager = CL_PromptInput(
    model=model,
    device=device,
    use_pgpt=True,
    prompt_dim=256
)
```

### 2. **Multi-Class Task**
```python
# Configuration for multi-class tasks
loader_dict = {
    ["bottle", "cable", "capsule"]: {  # 3 classes in one task
        "train": mixed_train_loader,    # Contains samples from all 3 classes
        "test": mixed_test_loader
    },
    ["pill", "screw"]: {               # 2 classes in another task
        "train": mixed_train_loader2,
        "test": mixed_test_loader2
    }
}

# Training automatically handles class-specific prompts
for task_classes, loaders in loader_dict.items():
    # All classes in task_classes get their own prompts
    # Batch processing applies appropriate prompts per sample
```

### 3. **Training Loop with Class-Level Prompts**
```python
for images, labels, class_labels in dataloader:
    # class_labels: [0, 1, 0, 2, 1, ...]  # Class IDs for each sample
    
    # Automatically inject appropriate prompts for each sample
    cl_manager.inject_prompts_for_batch(class_labels)
    
    # Forward pass - each sample uses its class-specific prompt
    outputs = model({'image': images, 'clslabel': class_labels})
    
    # Loss calculation and backpropagation
    loss = model.criterion(outputs, inputs)
    loss.backward()
    optimizer.step()
```

### 4. **Cross-Class Inference**
```python
# During evaluation on a different class
for images, labels, class_labels in test_dataloader:
    with torch.no_grad():
        # Extract backbone features first
        backbone_features = model.backbone({'image': images, 'clslabel': class_labels})
        backbone_features = model.neck(backbone_features).get('feature_align', None)
        
        # Enhanced class-aware prompt injection with K-NN fallback
        cl_manager.inject_prompts_for_batch(class_labels, backbone_features)
        
        outputs = model({'image': images, 'clslabel': class_labels})
```

### 5. **Enhanced Cross-Class Evaluation**

The system now provides sophisticated cross-class evaluation capabilities:

```python
# The inject_prompts_for_batch function now handles:
# 1. Direct class-specific prompt usage when available
# 2. K-NN based prompt selection using learned prototypes 
# 3. Fallback to current class prompts
# 4. Zero prompt as last resort

cl_manager.inject_prompts_for_batch(
    class_labels=batch_class_labels,          # [B] tensor of class IDs
    backbone_features=extracted_features      # [B, C, H, W] backbone features
)

# During cross-class evaluation:
# - If class has learned prompts → use them directly
# - If no learned prompts → use K-NN to find closest learned class
# - If no prototypes → fallback to current training class prompts
# - If nothing available → use zero prompts with warning
```

## Enhanced Cross-Class Evaluation

### Intelligent Fallback Mechanism

The PGPT system now includes a sophisticated fallback mechanism for cross-class evaluation:

1. **Primary Path**: Use class-specific prompts if available
2. **K-NN Fallback**: Select prompts from the most similar learned class using prototype similarity
3. **Current Class Fallback**: Use the current training class prompts
4. **Zero Prompt Fallback**: Create zero prompts as last resort

### How K-NN Selection Works

```python
# Example of K-NN selection process
def evaluate_cross_class(model, cl_manager, test_data):
    for images, labels, class_labels in test_data:
        # Extract features
        features = model.backbone({'image': images, 'clslabel': class_labels})
        backbone_features = model.neck(features).get('feature_align', None)
        
        # Intelligent prompt selection happens automatically
        cl_manager.inject_prompts_for_batch(class_labels, backbone_features)
        
        # For each sample without direct class prompts:
        # 1. Extract sample features: [C, H, W] → [C*H*W]  
        # 2. Compare with all learned prototypes using cosine similarity
        # 3. Select prompts from the most similar class
        # 4. Log selection for debugging
        
        outputs = model.reconstruction({'image': images, 'clslabel': class_labels})
```

### Debugging Cross-Class Evaluation

The system provides detailed logging for cross-class scenarios:

```python
# Example debug output
K-NN fallback: selected 'bottle' for class 'cable' (layer 0)
Cross-class evaluation: evaluating ['cable', 'capsule'] with model trained on bottle
Warning: Using zero prompt for class 'unknown_class' (no learned prompt available)
```

## Advantages

### 1. **Fine-Grained Control**
- **Class-Level Precision**: Each class can have specialized prompts
- **Batch Flexibility**: Different samples in same batch can use different prompts
- **Dynamic Adaptation**: Prompts are selected based on actual content

### 2. **Multi-Class Task Support**
- **Task Efficiency**: Train multiple related classes together
- **Resource Sharing**: Share backbone while maintaining class-specific prompts
- **Flexible Organization**: Organize classes into tasks as needed

### 3. **Memory and Computation Efficiency**
- **Shared Parameters**: Main model parameters shared across all classes
- **Minimal Overhead**: Only class-specific prompts stored per class
- **Efficient Inference**: Fast prompt selection and application

### 4. **Robust Generalization**
- **Cross-Class Transfer**: K-NN mechanism enables cross-class inference
- **Prototype Guidance**: Learned prototypes guide prompt selection
- **Continual Adaptation**: Easy to add new classes without retraining

## Performance Considerations

### 1. **Memory Usage**
- **Prompt Storage**: O(num_classes × num_layers × prompt_dim)
- **Prototype Storage**: O(num_classes × feature_dim)
- **Batch Processing**: Minimal additional memory for batch-wise prompts

### 2. **Computational Overhead**
- **Class Lookup**: Fast O(1) class label to prompt mapping
- **Batch Assembly**: Linear O(batch_size) prompt assembly
- **K-NN Selection**: O(num_classes) similarity computation when needed

## Troubleshooting

### Common Issues

1. **Class Label Mapping**: Ensure `get_class_from_label()` correctly maps your label IDs
2. **Device Placement**: Prompts are automatically moved to correct device
3. **Batch Size**: Works with any batch size, including mixed-class batches
4. **Memory**: Monitor GPU memory with large numbers of classes

### Best Practices

1. **Group Related Classes**: Put related classes in the same task for better learning
2. **Balance Class Distribution**: Ensure roughly balanced class distributions in mixed batches
3. **Monitor Prompt Usage**: Check which prompts are being used during training
4. **Prototype Quality**: Ensure sufficient data for meaningful prototype calculation

## Migration from Task-Level to Class-Level

### Configuration Changes
```yaml
# OLD: Task-level configuration
CONTINUAL:
  use_pgpt: true
  num_prompts_per_class: 1  # Remove this

# NEW: Class-level configuration  
CONTINUAL:
  use_pgpt: true
  # num_prompts_per_class removed - automatic class management
```

### Code Changes
```python
# OLD: Manual prompt injection
cl_manager.set_current_class(class_name)
outputs = model(inputs)

# NEW: Automatic batch-wise prompt injection
cl_manager.inject_prompts_for_batch(class_labels)  # Handles mixed batches
outputs = model(inputs)
``` 

## Loading PGPT Models from Checkpoints

### Issue with learned_embed Parameters

PGPT models dynamically modify `learned_embed` parameters during runtime. These parameters are not saved in the standard model `state_dict()` because they are created and modified dynamically. When loading a model, you might encounter errors like:

```
RuntimeError: Error(s) in loading state_dict for CFGCAD:
        Missing key(s) in state_dict: "reconstruction.transformer.decoder.layers.0.learned_embed.weight", ...
```

### Solution 1: Using the Helper Function (Recommended)

Use the provided helper function for safe loading:

```python
from train.train_cfgcad import load_pgpt_model_from_checkpoint
from CL import CL_PromptInput

# Initialize your model with the same architecture as during training
model = CFGCAD(
    backbone='resnet18',
    **your_model_params
)

# Load the checkpoint safely
model, cl_manager_state = load_pgpt_model_from_checkpoint(
    model, 
    'path/to/your/checkpoint.pth', 
    device='cuda'
)

# If you need the CL manager (for PGPT functionality), restore it:
cl_manager = CL_PromptInput(model, device='cuda', use_pgpt=True, prompt_dim=256)
if cl_manager_state:
    cl_manager.load_state_dict(cl_manager_state)
    print("CL manager restored with prompts and prototypes")
```

### Solution 2: Manual Loading with strict=False

If you prefer manual control:

```python
import torch

# Load checkpoint
checkpoint = torch.load('path/to/checkpoint.pth', map_location='cpu')

# Load model state with missing key tolerance
missing_keys, unexpected_keys = model.load_state_dict(
    checkpoint['model_state_dict'], 
    strict=False  # This allows missing learned_embed keys
)

# Move to device and set to eval mode
model = model.to('cuda')
model.eval()
```

### Solution 3: Standard PyTorch Loading (Advanced)

For advanced users who want full control:

```python
import torch

# Load checkpoint
checkpoint = torch.load('path/to/checkpoint.pth', map_location='cpu')

# Load with strict=False to ignore missing PGPT parameters
missing_keys, unexpected_keys = model.load_state_dict(
    checkpoint['model_state_dict'], 
    strict=False
)

# Filter out expected PGPT missing keys
pgpt_missing = [k for k in missing_keys if 'learned_embed' in k]
other_missing = [k for k in missing_keys if 'learned_embed' not in k]

print(f"PGPT parameters missing (expected): {len(pgpt_missing)}")
if other_missing:
    print(f"WARNING: Other parameters missing: {other_missing}")
```

### Important Notes

1. **PGPT prompts are reinitialized**: When you load a model, PGPT prompts will be reinitialized when needed. The CL manager state contains the learned prompts and prototypes.

2. **CL Manager State**: Always check if `cl_manager_state` is available in the checkpoint if you want to restore the learned prompts and prototypes.

3. **Architecture Consistency**: Make sure your model architecture (especially `prompt_dim`, `num_decoder_layers`) matches the saved model.

4. **Device Compatibility**: The helper function handles device placement automatically. 