import fiftyone as fo

# List all dataset names
datasets = fo.list_datasets()
print("Available datasets:")
for name in datasets:
    print(f"  - {name}")

# Get more details about each dataset
print("\nDataset details:")
for name in datasets:
    dataset = fo.load_dataset(name)
    print(f"  {name}:")
    print(f"    Samples: {len(dataset)}")
    print(f"    Media type: {dataset.media_type}")
    print(f"    Persistent: {dataset.persistent}")