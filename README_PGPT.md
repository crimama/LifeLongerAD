# PGPT (Prototype-Guided Prompt Tuning) Implementation

## Overview

This implementation adds **Prototype-Guided Prompt Tuning (PGPT)** functionality to the existing continual learning framework. PGPT is a parameter-efficient approach that uses small learnable prompts to guide the reconstruction process for each class, while keeping the main model parameters shared across all classes.

## Key Features

### 1. **Shared-Specific Architecture**
- **Shared Body**: Backbone, Neck, Transformer Decoder are shared across all classes
- **Class-specific Prompts**: Each class has its own learnable prompt vectors
- **Prototype Repository**: Stores representative features for each class

### 2. **Prototype-Guided Prompt Selection**
- **K-NN Selection**: Uses cosine similarity to find the closest prototype
- **Dynamic Prompt Injection**: Automatically selects the best prompt during inference
- **Prototype Calculation**: Computes class prototypes after training completion

### 3. **Parameter Efficiency**
- **Minimal Parameters**: Only prompt vectors are class-specific
- **Shared Knowledge**: Main model learns general reconstruction capabilities
- **Scalable**: Easy to add new classes without retraining the entire model

## Implementation Details

### Core Components

#### 1. **CL_Transformer Class Enhancements**
```python
# PGPT initialization
cl_manager = CL_Transformer(
    model=model,
    device=device,
    sparsity_config=sparsity_config,
    use_pgpt=True,                    # Enable PGPT
    prompt_dim=256,                   # Prompt vector dimension
    num_prompts_per_class=1,          # Prompts per class
    k_neighbors=1                     # K-NN parameter
)
```

#### 2. **Prompt Management**
- `initialize_prompt_for_class()`: Creates prompts for new classes
- `set_current_class()`: Freezes/unfreezes prompts during training
- `get_prompts_for_class()`: Retrieves prompts for specific classes

#### 3. **Prototype System**
- `calculate_prototype()`: Computes class prototypes from training data
- `select_prompt_by_knn()`: K-NN selection during inference
- `inject_prompt_into_model()`: Integrates prompts into model input

### Training Workflow

#### Phase 1: Class Training
1. **Prompt Initialization**: Create random prompts for new class
2. **Class-Specific Training**: Train shared model + current class prompts
3. **Prototype Calculation**: Compute class prototype after training
4. **Prompt Freezing**: Freeze prompts for trained classes

#### Phase 2: Inference
1. **Feature Extraction**: Extract features from input image
2. **K-NN Selection**: Find closest prototype using cosine similarity
3. **Prompt Injection**: Inject selected prompt into model
4. **Reconstruction**: Generate reconstruction using guided model

## Configuration

### Basic PGPT Configuration
```yaml
CONTINUAL:
  use_pgpt: true                    # Enable PGPT
  prompt_dim: 256                   # Prompt vector dimension
  num_prompts_per_class: 1          # Number of prompts per class
  k_neighbors: 1                    # K-NN neighbors
```

### Advanced Configuration
```yaml
CONTINUAL:
  use_pgpt: true
  prompt_dim: 512                   # Larger prompts for complex classes
  num_prompts_per_class: 3          # Multiple prompts per class
  k_neighbors: 3                    # Ensemble multiple neighbors
  use_knowledge_distillation: true  # Combine with KD
  use_neighbor_mask: true           # Enhanced masking
```

## Usage Examples

### 1. **Basic PGPT Training**
```python
# Load configuration with PGPT enabled
config = load_config("configs/pgpt_example.yaml")

# Initialize model and CL manager
model = create_model(config)
cl_manager = CL_Transformer(
    model=model,
    device=device,
    sparsity_config=config.CONTINUAL.method.params,
    use_pgpt=True,
    prompt_dim=256
)

# Training loop
for class_name, dataloader in loader_dict.items():
    # Initialize prompts for new class
    cl_manager.initialize_prompt_for_class(class_name)
    cl_manager.set_current_class(class_name)
    
    # Train the model
    train_model(model, dataloader, cl_manager)
    
    # Calculate prototype after training
    cl_manager.calculate_prototype(dataloader, class_name)
```

### 2. **PGPT Inference**
```python
# During inference
for images, labels in test_dataloader:
    # Extract features for prompt selection
    features = model.backbone(images)
    
    # Select best prompt using K-NN
    prompt, selected_class = cl_manager.select_prompt_by_knn(features)
    
    # Inject prompt and get reconstruction
    model_input = {'image': images, 'prompt': prompt}
    outputs = model(model_input)
    
    # Calculate anomaly score
    anomaly_score = calculate_anomaly_score(outputs)
```

## Advantages

### 1. **Parameter Efficiency**
- **Shared Parameters**: Main model parameters are shared across all classes
- **Minimal Class-Specific Parameters**: Only small prompt vectors per class
- **Scalable**: Easy to add new classes without increasing model size significantly

### 2. **Knowledge Preservation**
- **Prototype-Guided Selection**: Uses learned prototypes for prompt selection
- **K-NN Mechanism**: Robust selection based on feature similarity
- **Continual Learning**: Maintains performance on previous classes

### 3. **Flexibility**
- **Configurable**: Easy to adjust prompt dimensions and selection parameters
- **Compatible**: Works with existing continual learning methods
- **Extensible**: Can be combined with other techniques (KD, masking, etc.)

## Performance Considerations

### 1. **Memory Usage**
- **Prompt Storage**: Minimal memory for prompt vectors
- **Prototype Storage**: Small memory footprint for prototypes
- **Shared Model**: Efficient memory usage for main model

### 2. **Computational Overhead**
- **K-NN Selection**: Fast cosine similarity computation
- **Prompt Injection**: Minimal computational cost
- **Prototype Calculation**: One-time computation per class

### 3. **Training Efficiency**
- **Focused Training**: Only relevant parameters are updated
- **Frozen Prompts**: Previous class prompts remain unchanged
- **Shared Learning**: General reconstruction capabilities improve over time

## Integration with Existing Code

The PGPT implementation is designed to be **minimally invasive** and **backward compatible**:

1. **Optional Feature**: PGPT is disabled by default
2. **Existing Functionality**: All existing features remain unchanged
3. **Configurable**: Can be enabled/disabled via configuration
4. **Modular Design**: PGPT components are self-contained

## Future Enhancements

### 1. **Advanced Prompt Strategies**
- **Multi-Prompt Ensembles**: Use multiple prompts per class
- **Adaptive Prompt Selection**: Dynamic prompt selection based on context
- **Prompt Evolution**: Gradually evolve prompts during training

### 2. **Enhanced Prototype Methods**
- **Hierarchical Prototypes**: Multi-level prototype representations
- **Online Prototype Updates**: Continuous prototype refinement
- **Prototype Clustering**: Group similar prototypes for efficiency

### 3. **Integration Improvements**
- **Attention-Based Injection**: Use attention mechanisms for prompt injection
- **Cross-Modal Prompts**: Support for multi-modal prompt representations
- **Meta-Learning**: Learn to generate prompts for new classes

## Troubleshooting

### Common Issues

1. **Prompt Dimension Mismatch**
   - Ensure `prompt_dim` matches model's feature dimension
   - Check that prompt injection is compatible with model architecture

2. **Prototype Calculation Failures**
   - Verify that feature extraction works correctly
   - Check that dataloader provides expected data format

3. **K-NN Selection Issues**
   - Ensure prototypes are calculated before inference
   - Check that feature dimensions match between input and prototypes

### Debug Tips

1. **Enable Debug Logging**
   ```python
   # Add debug prints in PGPT methods
   print(f"✓ PGPT: Prompt injected for class '{class_name}'")
   ```

2. **Monitor Prompt Selection**
   ```python
   # Track which prompts are selected during inference
   prompt, selected_class = cl_manager.select_prompt_by_knn(features)
   print(f"Selected prompt for class: {selected_class}")
   ```

3. **Verify Prototype Quality**
   ```python
   # Check prototype statistics
   for class_name, prototype in cl_manager.prototype_repository.items():
       print(f"Class {class_name}: prototype norm = {prototype.norm()}")
   ```

## Conclusion

The PGPT implementation provides a **parameter-efficient** and **scalable** approach to continual learning for anomaly detection. By using small learnable prompts to guide reconstruction, it maintains the benefits of shared model parameters while providing class-specific adaptation capabilities.

The implementation is **production-ready** and can be easily integrated into existing continual learning pipelines with minimal code changes. 