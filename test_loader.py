from data_loader import load_dataset

dataset = load_dataset("data/roi", "data/labels")

x, y = dataset[0]

print("Image tensor shape:", x.shape)
print("Label tensor shape:", y.shape)
