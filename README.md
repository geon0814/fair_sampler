# 📦 Fair-Sampler  
*By Geon*

**Adaptive, feedback-controlled sampler that applies negative feedback to keep class distribution fair and stable during AI training.**  
Forces rapid convergence to target ratios — while preserving unbiased estimates via **importance weighting**.

> 실전 롱테일/드리프트 환경에서 샘플링 비율을 실시간 제어하고, 중요도 가중치로 편향 없이 학습합니다.

![Rebalance (99:1 → 50:50)](assets/fair_rebalance.png)
![Drift→Recovery (10 cycles)](assets/drift_recovery.png)

## 🚀 Quick Start

### Installation
```bash
pip install -e .
```

### Basic Usage
```python
from gfs.controller import FeedbackController
from gfs.batch_sampler import FeedbackBatchSampler
from gfs.iw import importance_weights

# Initialize controller and sampler
controller = FeedbackController(
    num_classes=K, 
    target_probs=[1/K]*K, 
    alpha=8.0
)
sampler = FeedbackBatchSampler(
    labels, 
    batch_size=256, 
    steps_per_epoch=100, 
    controller=controller
)

# Training loop
for batch_indices in sampler:
    y = labels[batch_indices]                        # class ids for the batch
    p = controller.get_probs()                       # current sampling probs
    w = importance_weights(y, controller.q, p)       # unbiased IW (mean≈1)
    
    # Compute loss with importance weighting
    loss = (w * criterion(model(x[batch_indices]), torch.as_tensor(y))).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # Update controller (can optionally pass per-sample losses)
    controller.step_update(y, losses=None)
```

## 🎯 Core Concept

### How It Works
- **Negative Feedback Loop**: Undersampled classes receive higher sampling probability, automatically counteracting class imbalance.
- **Softmax Control**: Dynamically adjusts sampling distribution using:
  ```
p_i(t) = softmax(α * (λ₁ * deficit_i + λ₂ * ewma_loss_i))
  ```
- **Unbiased Training**: Importance weights normalize gradients:
  ```
w = q[labels] / p[labels]  (normalized to mean ≈ 1)
  ```
  This ensures zero bias in gradient estimates while the sampler adapts.

### Key Parameters
| Parameter | Description | Default |
|-----------|-------------|---------|
| `alpha` | Temperature for softmax control (higher = faster adaptation) | 8.0 |
| `target_probs` | Target class distribution | Uniform |
| `min_prob` | Minimum sampling probability (stability floor) | 1e-6 |
| `max_w` | Maximum importance weight clipping | None |

## ⚙️ Configuration & Stability

### Recommended Settings
```python
# Default (balanced dataset, stable)
controller = FeedbackController(num_classes=10, target_probs=[0.1]*10, alpha=8.0)

# For extreme imbalance (e.g., 99:1)
controller = FeedbackController(num_classes=2, target_probs=[0.5, 0.5], alpha=12.0)
importance_weights(..., max_w=50.0)  # Cap extreme weights

# For streaming/drift scenarios
controller = FeedbackController(num_classes=K, alpha=6.0)  # Lower alpha for gradual changes
```

### Stability Tips
- **Probability Floor**: Set `min_prob=1e-6` (default) to prevent extreme importance weights.
- **Weight Clipping**: If drift is aggressive, use `importance_weights(..., max_w=50.0)`. 
- **Alpha Scheduling**: Consider warming up/cooling down `alpha` for smooth adaptation:
  ```python
  alpha_warm_up = min(8.0, 8.0 * epoch / 10)  # Ramp up over 10 epochs
  ```

## 📊 Example: Training with Fair-Sampler

See `train_mnist.py` for a complete runnable example. The same approach works directly with:
- **Long-tailed datasets** (CIFAR-100-LT, ImageNet-LT)
- **Medical imaging** (imbalanced diagnosis classes)
- **Streaming data** with concept drift
- **Custom datasets** (any labeled data)

```python
# Pseudocode: Apply to your own data
from gfs.controller import FeedbackController
from gfs.batch_sampler import FeedbackBatchSampler

labels = torch.tensor([...])  # Your class labels
controller = FeedbackController(num_classes=num_classes, target_probs=target_dist)
sampler = FeedbackBatchSampler(labels, batch_size=256, steps_per_epoch=steps_per_epoch, controller=controller)

for batch_indices in sampler:
    # Your training code here
    pass
```

## 🔄 Reproducibility

Ensure deterministic results:
```python
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
```

## 📚 API Reference

### FeedbackController
```python
controller = FeedbackController(
    num_classes: int,
    target_probs: List[float],
    alpha: float = 8.0,
    min_prob: float = 1e-6
)

# Methods
p = controller.get_probs()                    # Get current sampling distribution
controller.step_update(batch_labels, losses=None)  # Update controller state
```

### FeedbackBatchSampler
```python
sampler = FeedbackBatchSampler(
    labels: torch.Tensor,
    batch_size: int,
    steps_per_epoch: int,
    controller: FeedbackController
)
```

### importance_weights
```python
w = importance_weights(
    batch_labels: torch.Tensor,
    q: torch.Tensor,              # True class distribution
    p: torch.Tensor,              # Current sampling distribution
    max_w: Optional[float] = None # Weight clipping threshold
)
```

## 🐛 Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Unstable training | Alpha too high, aggressive sampling changes | Reduce `alpha`, use weight clipping `max_w` |
| Slow convergence to target | Alpha too low | Increase `alpha` gradually |
| Extreme importance weights | Large p-q mismatch | Use `max_w` clipping or increase `min_prob` |
| Loss spikes | Rapid concept drift | Reduce `alpha`, implement alpha scheduling |

## 📄 License

See `LICENSE` file for details.

## 🤝 Contributing

Contributions are welcome! Please open issues or pull requests for bugs, features, or documentation improvements.

---

**Questions or feedback?** Open an issue on GitHub.