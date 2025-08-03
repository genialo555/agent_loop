# Gemma 3N Technical Reference

## Architecture Deep Dive

### MatFormer (Matryoshka Transformer) Architecture

The MatFormer architecture is the cornerstone of Gemma 3N's efficiency and flexibility. It implements a novel nested transformer design where smaller models are embedded within larger ones.

#### Core Concepts

1. **Nested Model Training**
   - During training, both E4B and E2B models are optimized simultaneously
   - The E2B model is not a separate entity but an integral part of E4B
   - This allows for zero-cost extraction of the smaller model

2. **Progressive Layer Embedding**
   ```
   Layer 1: FFN Hidden Dim = 16384 (Full E4B)
   Layer 2: FFN Hidden Dim = 14336 (Partially reduced)
   ...
   Layer N: FFN Hidden Dim = 8192  (E2B configuration)
   ```

3. **Mix-n-Match Capability**
   - Create custom model sizes by adjusting FFN dimensions
   - Skip layers selectively for further size reduction
   - Fine-grained control over model capacity vs. performance

#### Implementation Details

```python
# Example: Creating a custom-sized model between E2B and E4B
def create_mixed_model(base_model, target_size_ratio=0.75):
    """
    Create a model that's 75% the size of E4B
    target_size_ratio: 0.5 = E2B, 1.0 = E4B
    """
    e2b_dim = 8192
    e4b_dim = 16384
    target_dim = int(e2b_dim + (e4b_dim - e2b_dim) * target_size_ratio)
    
    # Adjust FFN dimensions per layer
    for layer in base_model.layers:
        layer.mlp.resize_hidden_dim(target_dim)
    
    return base_model
```

### AltUp (Alternating Updates) Architecture

AltUp is a parameter-efficient training technique that alternates between updating different parts of the model.

#### Key Principles

1. **Sparse Activation**
   - Not all parameters are updated in each forward pass
   - Reduces computational cost while maintaining expressivity

2. **Alternating Pattern**
   - Even iterations: Update subset A of parameters
   - Odd iterations: Update subset B of parameters
   - Full model capacity utilized over multiple steps

3. **Memory Efficiency**
   - Only active parameters need gradients computed
   - Significant VRAM savings during training

#### Mathematical Formulation

```
Forward pass at step t:
- If t % 2 == 0: h = W_A * x + b_A
- If t % 2 == 1: h = W_B * x + b_B

Where W_A and W_B are disjoint parameter subsets
```

### Per-Layer Embeddings (PLE)

PLE introduces layer-specific embeddings that adapt representations throughout the model depth.

#### Architecture Details

1. **Layer-Specific Adaptation**
   ```python
   class PLELayer(nn.Module):
       def __init__(self, hidden_size, num_layers):
           super().__init__()
           self.layer_embeddings = nn.Parameter(
               torch.randn(num_layers, hidden_size)
           )
       
       def forward(self, x, layer_idx):
           # Add layer-specific embedding
           return x + self.layer_embeddings[layer_idx]
   ```

2. **Memory Footprint**
   - Traditional: O(vocab_size × hidden_size)
   - With PLE: O(vocab_size × hidden_size + num_layers × hidden_size)
   - Negligible increase for massive quality gains

3. **Benefits**
   - Better gradient flow through deep networks
   - Layer-wise specialization
   - Improved few-shot learning

### LAuReL-LR (Learned Augmented Residual Layers - Low Rank)

LAuReL-LR enhances residual connections with learned, low-rank transformations.

#### Implementation Concept

```python
class LAuReLBlock(nn.Module):
    def __init__(self, hidden_size, rank=64):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, rank)
        self.up_proj = nn.Linear(rank, hidden_size)
        self.gate = nn.Parameter(torch.zeros(1))
    
    def forward(self, x, residual):
        # Learned augmentation of residual connection
        augmentation = self.up_proj(F.gelu(self.down_proj(residual)))
        return x + residual + torch.sigmoid(self.gate) * augmentation
```

## Multimodal Architecture

### Vision Encoder: MobileNet-v5

Gemma 3N uses an optimized MobileNet-v5 backbone for vision tasks:

1. **Efficiency Features**
   - Depthwise separable convolutions
   - Squeeze-and-excitation blocks
   - Neural Architecture Search (NAS) optimized

2. **Integration with LLM**
   ```python
   vision_features = mobilenet_v5(image)  # [B, 1280]
   vision_tokens = vision_projector(vision_features)  # [B, n_tokens, d_model]
   combined = torch.cat([text_tokens, vision_tokens], dim=1)
   ```

### Audio Processing

Audio processing uses a specialized encoder optimized for on-device deployment:

1. **Feature Extraction**
   - 16kHz sampling rate
   - Log-mel spectrograms
   - Sliding window with 25ms frames

2. **Temporal Modeling**
   - Conformer-based architecture
   - Causal attention for streaming

## Performance Characteristics

### Benchmark Results

| Model | Parameters | LMArena Score | Latency (ms) | Memory (GB) |
|-------|------------|---------------|--------------|-------------|
| E2B   | 2B         | 1150          | 15-25        | 4-6         |
| E4B   | 4B         | 1300+         | 25-40        | 8-10        |

### Hardware Requirements

#### Minimum Requirements
- **E2B Model**
  - GPU: 6GB VRAM (RTX 2060, GTX 1660 Ti)
  - CPU: 8GB RAM for CPU inference
  - Storage: 4GB for model weights

- **E4B Model**
  - GPU: 10GB VRAM (RTX 3060, RTX 2080)
  - CPU: 16GB RAM for CPU inference
  - Storage: 8GB for model weights

#### Recommended Setup
- **Development**: RTX 3090/4090 with 24GB VRAM
- **Production**: A100 40GB or H100 for best performance
- **Edge Deployment**: Jetson Orin or similar

## Training Considerations

### Data Requirements

1. **Format**: Conversational or instruction-following format
2. **Preprocessing**: Use chat templates
3. **Tokenization**: SentencePiece tokenizer with 256k vocabulary

Example data format:
```json
{
  "messages": [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
  ]
}
```

### Hyperparameter Guidelines

| Parameter | E2B | E4B | Notes |
|-----------|-----|-----|-------|
| Learning Rate | 2e-4 | 1e-4 | Lower for larger models |
| Batch Size | 8-16 | 4-8 | Depends on VRAM |
| Gradient Accumulation | 2-4 | 4-8 | Effective batch = physical × accumulation |
| Warmup Steps | 100 | 200 | 5-10% of total steps |
| Weight Decay | 0.01 | 0.01 | Standard for AdamW |

### Mixed Precision Training

```python
# Automatic mixed precision setup
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast(dtype=torch.float16):  # or torch.bfloat16
        outputs = model(**batch)
        loss = outputs.loss
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## Deployment Strategies

### ONNX Export

```python
# Export to ONNX for edge deployment
import torch.onnx

dummy_input = {
    "input_ids": torch.randint(0, 32000, (1, 128)),
    "attention_mask": torch.ones(1, 128)
}

torch.onnx.export(
    model,
    (dummy_input,),
    "gemma3n_e2b.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "logits": {0: "batch", 1: "sequence"}
    }
)
```

### Quantization Strategies

1. **Post-Training Quantization**
   ```python
   from transformers import AutoModelForCausalLM
   import torch
   
   model = AutoModelForCausalLM.from_pretrained(
       "google/gemma-3n-e4b",
       torch_dtype=torch.float16,
       load_in_8bit=True  # or load_in_4bit=True
   )
   ```

2. **Dynamic Quantization**
   ```python
   import torch.quantization
   
   quantized_model = torch.quantization.quantize_dynamic(
       model,
       {torch.nn.Linear},
       dtype=torch.qint8
   )
   ```

### Elastic Inference Implementation

```python
class ElasticGemma3N:
    def __init__(self, model_path):
        self.e4b_model = load_model(model_path)
        self.e2b_config = extract_e2b_config()
        self.current_mode = "e4b"
    
    def switch_mode(self, mode="auto"):
        if mode == "auto":
            # Determine based on system load
            gpu_memory = torch.cuda.memory_allocated()
            mode = "e2b" if gpu_memory > 0.8 * torch.cuda.max_memory else "e4b"
        
        if mode == "e2b":
            self._resize_to_e2b()
        else:
            self._resize_to_e4b()
        
        self.current_mode = mode
    
    def _resize_to_e2b(self):
        for layer in self.e4b_model.layers:
            layer.mlp.resize_hidden_dim(8192)
    
    def _resize_to_e4b(self):
        for layer in self.e4b_model.layers:
            layer.mlp.resize_hidden_dim(16384)
```

## Advanced Topics

### Custom Mix-n-Match Configurations

Create application-specific model sizes:

```python
def create_custom_model(base_e4b_path, config):
    """
    config = {
        "layer_dims": [16384, 14336, 12288, 10240, 8192],  # Per-layer FFN dims
        "skip_layers": [15, 16, 17],  # Layers to skip entirely
        "attention_heads": 16,  # Adjust attention heads
    }
    """
    model = load_model(base_e4b_path)
    
    # Adjust FFN dimensions
    for i, layer in enumerate(model.layers):
        if i < len(config["layer_dims"]):
            layer.mlp.resize_hidden_dim(config["layer_dims"][i])
    
    # Skip specified layers
    for skip_idx in config["skip_layers"]:
        model.layers[skip_idx] = nn.Identity()
    
    # Adjust attention if specified
    if "attention_heads" in config:
        for layer in model.layers:
            layer.attention.num_heads = config["attention_heads"]
    
    return model
```

### Memory-Efficient Fine-tuning Patterns

1. **Gradient Checkpointing with Custom Boundaries**
   ```python
   # Checkpoint every N layers instead of every layer
   def custom_checkpoint_model(model, checkpoint_every=4):
       for i, layer in enumerate(model.layers):
           if i % checkpoint_every == 0:
               layer = torch.utils.checkpoint(layer)
       return model
   ```

2. **Layer Freezing Strategies**
   ```python
   # Freeze bottom layers, fine-tune top layers
   def freeze_bottom_layers(model, num_frozen=10):
       for i, layer in enumerate(model.layers):
           if i < num_frozen:
               for param in layer.parameters():
                   param.requires_grad = False
   ```

## References and Further Reading

1. **MatFormer Paper**: "Elastic Inference with Matryoshka Transformer" (2023)
   - ArXiv: https://arxiv.org/pdf/2310.07707

2. **AltUp Paper**: "Scaling Language Models with Alternating Updates" (2023)
   - ArXiv: https://arxiv.org/abs/2301.13310

3. **LAuReL Paper**: "Learned Augmented Residual Layer" (2024)
   - ArXiv: https://arxiv.org/abs/2411.07501

4. **Google Blog Posts**:
   - Introduction: https://developers.googleblog.com/en/introducing-gemma-3n/
   - Developer Guide: https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/

5. **Official Resources**:
   - Model Card: https://ai.google.dev/gemma/docs/gemma-3n
   - Colab Examples: https://github.com/google-gemini/gemma-cookbook

6. **Community Resources**:
   - Reverse Engineering: https://github.com/antimatter15/reverse-engineering-gemma-3n
   - MatFormer Lab: https://colab.research.google.com/github/google-gemini/gemma-cookbook/blob/main/Gemma/[Gemma_3n]MatFormer_Lab.ipynb