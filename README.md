# Frequency Disentangled Learning for Segmentation of Midbrain Structures from Quantitative Susceptibility Mapping Data

This repository is released for review purposes of MICCAI 2023 conference. Any other use of the proposed methodology or code without explicit permission from the authors is prohibited. The code will be made Open source in a deanonymized version if the paper gets accepted. 

## Prerequisites

This code was tested on Python 3.9. Run the following commands to set up the environment and all dependencies.

``` bash
conda create -n freq-disen python=3.9
conda install --file requirements.txt
```

## Repository structure

TODO

 ## Method proposed

Pipeline of the proposed method. We introduce two ways of fusion: early and late fusion. Details can be seen in the Figure below: 

![method proposed](https://drive.google.com/uc?export=download&id=1GZzq7M6zabk_96CauubxbdFPi9VVeNzm)

In the following sections, we present an highlight on the key functions of the code :
1) Imaging data preparation for deep learning : `disentangle_tensor`
2) Training of the model : `train_epoch`
3) Frequency fusion : `fuse_center_border`, `forward_early_fusion`, `forward_late_fusion`

## 1. Data frequency disentangle representation 

* Load package: `torch` and `fftn`, `ifftn`

```python
import torch
from torch.fft import fftn, ifftn
```

* Prepare data for training deep learning models by distangle it to extract high and low frequency
```python

def disentangle_tensor(input_tensor, theta=0.1):
    """
    Splits a 5D input tensor into a central part and a border part.
    The central part is a square of size `theta` times the width of the tensor,
    centered in the middle of the tensor. The border part is everything outside the central part.
    
    Args:
        input_tensor: A 5D input tensor of shape (batch_size, num_classes, width, height, depth).
        theta: The fraction of the width of the tensor to use for the central square (default: 0.1).
    
    Returns:
        Two 5D tensors of the same shape as the input tensor. The first tensor is the central part, and the second tensor is the border part.
    """

    # Get the size of the input tensor
    batch = input_tensor.size()[0]
    classes = input_tensor.size()[1]
    width = input_tensor.size()[2]
    height = input_tensor.size()[3]
    depth = input_tensor.size()[4]

    # Check that the tensor is square
    assert width == height

    # Calculate the dimensions of the mask
    mask_width = int(width * theta)
    mask_star_x = int(width / 2 - mask_width / 2)
    mask_star_y = mask_star_x
    mask_end_x = int(width / 2 + mask_width / 2)
    mask_end_y = mask_end_x

    # Create a binary mask tensor that is 1 inside the mask and 0 outside
    mask = torch.zeros((batch, classes, width, width, depth))
    mask[:,:,mask_star_x:mask_end_x, mask_star_y:mask_end_y, :] = torch.ones((batch, classes, mask_width, mask_width, depth))

    # Extract the central part of the tensor (inside the mask) and the border part (outside the mask)
    input_fft_center = input_tensor[:,:,mask_star_x:mask_end_x, mask_star_y:mask_end_y, :]
    input_fft_border = torch.multiply(input_tensor, 1 - mask)

    # Return the central and border parts of the tensor
    return input_fft_center, input_fft_border

```

## 2. Model training

For each epoch, the training procedure is as follow :
  1. Load data: `input_tensor` and `target`.
  2. Fourier transfrom for `input_tensor`: `fftn()`.
  3. Disentangle `input_tensor` into high-frequency (`input_center_f`) and low-frequency (`input_boundary_f`) components by using `disentangle_tensor()`.
  4. Inverse Fourier transform from frequency space to image space: `input_center_f`  to `input_center_i` and  `input_boundary_f` to `input_boundary_i` by using `ifftn()`.
  5. Send disentangled representation  `input_center_i` and  `input_boundary_i` to model for training.
```python
def train_epoch(self):
    """
    Trains the model.
    """
    # Set up the input tensor and target tensor with random data
    num_batch = 2
    num_classes = 2
    image_height = 160
    image_weight = 160
    image_slice = 128
    tensor_size = (num_batch, num_classes, image_height, image_weight, image_slice)
    input_tensor, target = torch.rand(size=tensor_size), torch.ones(size=tensor_size)

    # Ensure that the input and target tensors have the same size
    assert input_tensor.size() == target.size()

    # Get the dimensions of the input tensor to perform the FFT along
    input_dim = tuple(range(2, input_tensor.ndim))

    # Apply the FFT to the input tensor to disentangle its frequencies
    input_f = fftn(input_tensor, dim=input_dim)

    # Use a mask to get the center and border parts of the input tensor
    input_center_f, input_border_f = disentangle_tensor(input_f)

    # Apply the inverse FFT to the center and border parts to convert them back to the spatial domain
    input_center_i = ifftn(input_center_f, dim=tuple(range(2, input_center_f.ndim))).type(torch.float32)
    input_border_i = ifftn(input_border_f, dim=tuple(range(2, input_border_f.ndim))).type(torch.float32)

    # Move the center and border parts to the GPU and set requires_grad to True to enable gradient computation
    input_center_i, input_border_i = input_center_i.cuda(), input_border_i.cuda()
    input_center_i.requires_grad = True
    input_border_i.requires_grad = True

    # Pass the input data to the model architecture
    output = self.model(input_center_i, input_border_i)
```


## 3. Model architecture and frequency fusion

* The function of frequency fusion is to set the value of the center of the frequency domain directly. In this case we never break the original image information. This function requires the inputs in the frquency domain.

```python
def fuse_center_border(tensor_center, tensor_border, theta=0.1):
    """
    Fuses the central and border parts of a tensor to create a new tensor.
    The central part is inserted into the border part at the center of the tensor.
    Args:
        tensor_center: A 5D tensor that represents the central part of the original tensor.
        tensor_border: A 5D tensor that represents the border part of the original tensor.
        theta: The fraction of the width of the tensor to use for the central square (default: 0.1).
    Returns:
        A new 5D tensor that represents the fused tensor.
    """
    # Get the size of the tensor and calculate the mask parameters
    width = tensor_border.size()[2]
    height = input_tensor.size()[3]
    assert width == height
    
    mask_width = int(width * theta)
    mask_star_x = int(width / 2 - mask_width / 2)
    mask_star_y = mask_star_x
    mask_end_x = int(width / 2 + mask_width / 2)
    mask_end_y = mask_end_x
    
    # Insert the central part into the border part at the center of the tensor
    tensor_border[:, :, mask_star_x:mask_end_x, mask_star_y:mask_end_y, :] = tensor_center
    
    # Return the fused tensor
    return tensor_border
```

### 3.1 Early fusion

1. Input the center componet `input_center_i` and boundary component `input_border_i` extracted in frequency domain which represent high and low frequency respectively into two `CNN()` layers. The convolution is performed in the image space domain.
1. Transform to frequency domain and do the feature fusion operation `fuse_center_border()`.
1. Transform the fused representation to image domain (`output_center_border_f` to `output_center_border_i`) and concat to the original model architecture (`self.unet()` for example).

```python
def forward_early_fusion(self, input_center_i, input_border_i):
    """
    Performs the forward pass for the early fusion architecture.
    Args:
        input_center_i: A 5D tensor that represents the central part of the input tensor in the spatial domain.
        input_border_i: A 5D tensor that represents the border part of the input tensor in the spatial domain.
    Returns:
        A 5D tensor that represents the output of the model in the spatial domain.
    """
    # Use convolutional layers to process the input tensors in the spatial domain
    output_border_i = self.conv3d_c1_border(input_border_i)
    output_center_i = self.conv3d_c1_center(input_center_i)
    
    # Use the FFT to transfer the outputs to the frequency domain
    output_border_f = fftn(output_border_i, dim=tuple(range(2, output_border_i.ndim)))
    output_center_f = fftn(output_center_i, dim=tuple(range(2, output_center_i.ndim)))
    
    # Use the `fuse_center_border` function to combine the output tensors in the frequency domain
    output_center_border_f = fuse_center_border(output_center_f, output_border_f)
    
    # Use the inverse FFT to transform the output back to the spatial domain
    output_center_border_i = ifftn(output_center_border_f, dim=tuple(range(2, output_center_border_f.ndim))).type(torch.float32)
    
    # Pass the output to the U-Net architecture
    out = output_center_border_i
    out = self.unet(out)
    
    # Do downstream task
    # ...
    
    # Return the output tensor
    return out
```

### 3.2 Late fusion
1. Input the center componet `input_center_i` and boundary component `input_border_i` extracted in frequency domain which represent high and low frequency respectively into two `CNN()` layers. The convolution is performed in the image space domain.
1. Process the `output_border_i` using the original model architecture (`self.unet()` for example).
1. Transform the output of the model to frequency domain and do the feature fusion operation `fuse_center_border()`.
1. Transform the fused representation to image domain (`output_center_border_f` to `output_center_border_i`) and concat the operation block for downstream task.


```python
def forward_late_fusion(self, input_center_i, input_border_i):
    """
    Performs the forward pass for the late fusion architecture.
    Args:
        input_center_i: A 5D tensor that represents the central part of the input tensor in the spatial domain.
        input_border_i: A 5D tensor that represents the border part of the input tensor in the spatial domain.
    Returns:
        A 5D tensor that represents the output of the model in the spatial domain.
    """
    # Use convolutional layers to process the input tensors in the spatial domain
    output_border_i = self.conv3d_c1_border(input_border_i)
    output_center_i = self.conv3d_c1_center(input_center_i)
    
    # Pass the output of the border convolutional layers to the U-Net architecture
    output_border_i = self.unet(output_border_i)
    
    # Use the FFT to transfer the outputs to the frequency domain
    output_border_f = fftn(output_border_i, dim=tuple(range(2, output_border_i.ndim)))
    output_center_f = fftn(output_center_i, dim=tuple(range(2, output_center_i.ndim)))
    
    # Use the `fuse_center_border` function to combine the output tensors in the frequency domain
    output_center_border_f = fuse_center_border(output_center_f, output_border_f)
    
    # Use the inverse FFT to transform the output back to the spatial domain
    output_center_border_i = ifftn(output_center_border_f, dim=tuple(range(2, output_center_border_f.ndim))).type(torch.float32)
    
    # Pass the output to downstream task
    out = output_center_border_i
    # ...
    
    # Return the output tensor
    return out

```
