---

| rollout/ | |
| ep_len_mean | 6.07 |
| ep_rew_mean | 0.919 |
| exploration_rate | 0.05 |
| time/ | |
| episodes | 12120 |
| fps | 3168 |
| time_elapsed | 31 |
| total_timesteps | 99996 |
| train/ | |
| learning_rate | 0.0001 |
| loss | 1.26e-05 |
| n_updates | 24973 |

---

### **Why Does the Table Appear?**

The table appears **periodically during training** because you are likely using **Stable-Baselines3 (SB3) or a similar RL framework**, which logs training statistics after a certain number of steps or episodes.

### **When Does It Appear?**

The table appears **after a fixed number of steps or episodes** have been completed. This depends on the logging interval set by the framework. If you're using `Stable-Baselines3`, it likely logs this table **every `log_interval` steps**.

### **What the Table Represents?**

It summarizes key training statistics:

| **Metric**         | **Meaning**                                                       |
| ------------------ | ----------------------------------------------------------------- |
| `ep_len_mean`      | Average episode length (in steps).                                |
| `ep_rew_mean`      | Average reward per episode.                                       |
| `exploration_rate` | The current exploration rate (used in epsilon-greedy strategies). |
| `episodes`         | Total number of episodes completed.                               |
| `fps`              | Frames per second (how fast the training is running).             |
| `time_elapsed`     | Time elapsed since training started (in seconds).                 |
| `total_timesteps`  | Total number of steps taken so far in training.                   |
| `learning_rate`    | The current learning rate of the RL algorithm.                    |
| `loss`             | Loss value during training (lower is better).                     |
| `n_updates`        | Number of training updates performed.                             |

### **How to Modify Logging Frequency?**

If you're using `Stable-Baselines3`, you can change the logging frequency with:

```python
model.learn(total_timesteps=100000, log_interval=10)  # Logs every 10 episodes
```
