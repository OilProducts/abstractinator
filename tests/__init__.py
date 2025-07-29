import os

# Disable torch.compile to avoid indutor failures during tests
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
