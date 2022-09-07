from model import ShiftInvariantTransformer
from dataset import generate_dataset, draw_dataset_sample
from functions import train_loop, Batchifyer
import IPython
import torch

x_train, y_train = generate_dataset(n=700)
x_val, y_val = generate_dataset(n=150)

draw_dataset_sample(x_val, y_val)

classes = ["square", "cross"]
device = "cuda:0"
train_data = Batchifyer(x_train, y_train, classes, device=device, batch_size=10, n_batches=1)
val_data = Batchifyer(x_val, y_val, classes, device=device, batch_size=10, n_batches=1)
model = ShiftInvariantTransformer(classes, n_stages=4, projection_dim=16, n_heads=4, t_scaling_factors=[0.5, 0.75, 1., 1.25, 1.5])
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1.0E-3)
train_loop(model, optimizer, train_data, val_data, n_steps=1000, patience=100)

IPython.embed()
