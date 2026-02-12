#  fair_sampler

**Fair Sampler: An adaptive sampling method using negative feedback for balanced AI training.**

Adaptive, feedback-controlled sampler that applies negative feedback to keep class distribution fair and stable during AI training. Forces rapid convergence to target ratios — while preserving unbiased estimates via **importance weighting**.

*(실전 롱테일/드리프트 환경에서 샘플링 비율을 실시간 제어하고, 중요도 가중치로 편향 없이 학습)*

![Rebalance (99:1 → 50:50)](assets/fair_rebalance.png)
![Drift→Recovery (10 cycles)](assets/drift_recovery.png)

---

##  Key Features
* **Real-time Adaptation**: Instantly responds to class imbalance using negative feedback.
* **Unbiased Learning**: Mathematical consistency via Importance Weighting (IW).
* **M-Series Optimized**: Specifically tuned for Apple Silicon (M1/M2/M3) & Python 3.14+.

---

##  Installation & Setup for Mac

```bash
pip install -e .

[!IMPORTANT]
Note for Mac/Python 3.14+ Users:
To avoid PickleError and MPS (GPU) memory warnings, always configure your DataLoader as follows:

num_workers=0 (Required for multiprocessing compatibility)

pin_memory=False (Recommended to avoid MPS warnings)


from gfs.controller import FeedbackController
from gfs.batch_sampler import FeedbackBatchSampler
from gfs.iw import importance_weights

---

# 1. Initialize Controller
controller = FeedbackController(num_classes=K, target_probs=[1/K]*K, alpha=8.0)

# 2. Wrap your Dataset
sampler = FeedbackBatchSampler(labels, batch_size=256, steps_per_epoch=100, controller=controller)

# 3. Training Loop
for batch_indices in sampler:
    y = labels[batch_indices]
    p = controller.get_probs()
    w = importance_weights(y, controller.q, p)
    
    # Apply Importance Weights to the loss
    loss = (w * criterion(model(x[batch_indices]), y)).mean()
    loss.backward(); optimizer.step(); optimizer.zero_grad()
    
    # Update Feedback Controller
    controller.step_update(y)
Core Idea (Math)
1. Negative Feedback Control

부족한 클래스에 더 높은 샘플링 확률을 부여하여 불균형을 즉각 해소합니다.

p 
i
​	
 (t)=softmax(α⋅(λ 
1
​	
 ⋅deficit 
i
​	
 +λ 
2
​	
 ⋅ewma_loss 
i
​	
 ))
2. Unbiased Training (Importance Weighting)

샘플링 확률 p가 가변적이더라도 중요도 가중치 w를 통해 데이터 분포 q에 대한 비편향 추정을 유지합니다.

w= 
p[labels]
q[labels]
​
