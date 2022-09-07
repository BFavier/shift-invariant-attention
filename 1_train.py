from model import ShiftInvariantTransformer
from dataset import generate_dataset, draw_dataset_sample, x_to_tensors, y_to_tensors


x_train, y_train = generate_dataset(n=700)
x_val, y_val = generate_dataset(n=150)

draw_dataset_sample(x_val, y_val)

classes = ["square", "cross"]
X_train, Y_train = x_to_tensors(x_train), y_to_tensors(y_train, classes)
X_val, Y_val = x_to_tensors(x_val), y_to_tensors(y_val, classes)
model = ShiftInvariantTransformer(classes, n_stages=4, projection_dim=16, n_heads=4, t_scaling_factors=[0.5, 0.75, 1., 1.25, 1.5])
