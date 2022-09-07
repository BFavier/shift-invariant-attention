from model import ShiftInvariantTransformer

classes = ["1", "2", "3", "4", "5", "6", "7", "8"]
model = ShiftInvariantTransformer(classes, n_stages=4, projection_dim=16, n_heads=4, t_scaling_factors=[0.5, 0.75, 1., 1.25, 1.5])