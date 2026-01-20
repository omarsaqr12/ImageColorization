#!/usr/bin/env python3
"""
Missing Method for visualize_trained_models.py
Add this method to your TrainedModelVisualizer class
"""

# Add this method to your TrainedModelVisualizer class:

def load_cifar10_samples(self, num_samples=10):
    """Load random samples from CIFAR-10 test set"""
    print(f"Loading {num_samples} random CIFAR-10 samples...")
    
    # Load CIFAR-10 test set
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./cifar-10-python', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Get random indices
    total_samples = len(test_dataset)
    random_indices = random.sample(range(total_samples), min(num_samples, total_samples))
    
    samples = []
    for idx in random_indices:
        image, label = test_dataset[idx]
        class_name = test_dataset.classes[label]
        samples.append({
            'image': image,
            'label': label,
            'class_name': class_name,
            'index': idx
        })
    
    print(f"✅ Loaded {len(samples)} samples")
    return samples

# INSTRUCTIONS:
# 1. Open visualize_trained_models.py on your device
# 2. Find the TrainedModelVisualizer class
# 3. Add this method after the colorize_with_trained method
# 4. Make sure to add proper indentation (4 spaces for each level)
# 5. Save the file and run again

print("Instructions:")
print("1. Open visualize_trained_models.py")
print("2. Find the TrainedModelVisualizer class")
print("3. Add the load_cifar10_samples method above")
print("4. Make sure indentation is correct (4 spaces per level)")
print("5. Save and run again")
