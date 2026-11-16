import torch
import torch.nn as nn

print("PyTorch version:", torch.__version__)

# Create tiny example predictions and labels
predictions = torch.tensor(
    [[0.2, -0.5], [0.8, 0.1]],
    dtype=torch.float32
)

labels = torch.tensor(
    [[1.0, 0.0], [1.0, 1.0]],
    dtype=torch.float32
)

print("Tensors created successfully!")

# Test the loss function
loss_function = nn.BCEWithLogitsLoss()

print("Calculating loss...")

loss = loss_function(predictions, labels)

print("Loss calculated successfully!")
print("Loss:", loss.item())