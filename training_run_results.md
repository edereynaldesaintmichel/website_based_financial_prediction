
gated_head
Epoch 3/3 — val loss: 1.1326  text: 0.4608  sign: 0.0085  mag: 0.6633

normal MLP head.
Epoch 1/3 — val loss: 1.2040  text: 0.5071  sign: 0.0106  mag: 0.6863

W/O data leakage:
Epoch 3/3 — val loss: 1.2828  text: 0.5833  sign: 0.0045  mag: 0.6950

W/O data leakage, full fine-tuning.
Epoch 1/3 — val loss: 1.2633  text: 0.5477  sign: 0.0046  mag: 0.7109
Epoch 2/3 — val loss: 1.1749  text: 0.4921  sign: 0.0034  mag: 0.6793

W/O data leakage, full fine-tuning, better number masking:
Epoch 1/3 — val loss: 1.2434  text: 0.5565  sign: 0.0047  mag: 0.6821
Epoch 2/3 — val loss: 1.1659  text: 0.5009  sign: 0.0044  mag: 0.6605

W/O data leakage, full fine-tuning, good number masking, weight tying:
SHIT

W/O data leakage, full fine-tuning, good number masking, gated embedder & head:
Epoch 1/3 — val loss: 1.2374  text: 0.5570  sign: 0.0050  mag: 0.6754 
LR far too high. Training instability. Could have been much lower I think.


Same thing, but lower lr:
Epoch 1/3 — val loss: 1.2382  text: 0.5531  sign: 0.0058  mag: 0.6793
Epoch 2/3 — val loss: 1.1868  text: 0.5208  sign: 0.0048  mag: 0.6612
Epoch 3/3 — val loss: 1.1767  text: 0.5164  sign: 0.0047  mag: 0.6556


Lower batch size, 2D RoPE. Embedder is still shitty:
Epoch 1/3 — val loss: 0.9869  text: 0.4852  sign: 0.0000  mag: 0.5017 -> A little better, isn't it?


Baseline:
Epoch 1/3:  19%|██████                          | 1419/7499 [10:25<44:40,  2.27batch/s, loss=1.0710, lr=4.98e-05, mag=0.5287, reg=1.4280, txt=0.5423]

Learned positional embeddings:
Epoch 1/3 (last 100 batches) — loss: 0.9548  text: 0.4480  mag: 0.5068  reg: 1.3583
Epoch 1/3 — val loss: 0.9706  text: 0.4740  mag: 0.4966
  Saved to checkpoints/mlm_full/checkpoint_epoch1
Epoch 2/3: 100%|████████████████████████████| 31771/31771 [54:58<00:00,  9.63batch/s, loss=0.9429, lr=6.31e-06, mag=0.4735, reg=1.3059, txt=0.4693]
Epoch 2/3 (last 100 batches) — loss: 0.9429  text: 0.4693  mag: 0.4735  reg: 1.3059
  Validation: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 3503/3503 [01:31<00:00, 38.12batch/s]
Epoch 2/3 — val loss: 0.9343  text: 0.4449  mag: 0.4894
  Saved to checkpoints/mlm_full/checkpoint_epoch2
Epoch 3/3: 100%|████████████████████████████| 31771/31771 [54:58<00:00,  9.63batch/s, loss=0.9076, lr=0.00e+00, mag=0.4873, reg=1.3292, txt=0.4204]
Epoch 3/3 (last 100 batches) — loss: 0.9076  text: 0.4204  mag: 0.4873  reg: 1.3292
  Validation: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 3503/3503 [01:31<00:00, 38.14batch/s]
Epoch 3/3 — val loss: 0.9250  text: 0.4378  mag: 0.4872
  Saved to checkpoints/mlm_full/checkpoint_epoch3


