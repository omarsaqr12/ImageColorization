#!/usr/bin/env python3
"""
Quick Fix for visualize_trained_models.py
Apply this fix to resolve the tensor dimension error
"""

# The issue is in the rgb_to_lab function. Replace this function:

def rgb_to_lab(self, rgb_tensor):
    """Convert RGB tensor to LAB tensor"""
    # Ensure tensor is 4D (batch, channels, height, width)
    if rgb_tensor.dim() == 3:
        rgb_tensor = rgb_tensor.unsqueeze(0)  # Add batch dimension
    
    rgb_np = rgb_tensor.permute(0, 2, 3, 1).cpu().numpy()
    lab_np = np.zeros_like(rgb_np)
    
    for i in range(rgb_np.shape[0]):
        rgb_normalized = np.clip(rgb_np[i], 0, 1)
        lab_np[i] = color.rgb2lab(rgb_normalized)
    
    lab_tensor = torch.from_numpy(lab_np).permute(0, 3, 1, 2).float()
    return lab_tensor.to(self.device)

# INSTRUCTIONS:
# 1. Open visualize_trained_models.py on your device
# 2. Find the rgb_to_lab function (around line 55-69)
# 3. Replace the entire function with the code above
# 4. Save the file
# 5. Run the script again

print("Fix instructions:")
print("1. Open visualize_trained_models.py")
print("2. Find the rgb_to_lab function")
print("3. Replace it with the fixed version above")
print("4. Save and run again")
