from PIL import Image
import random
import matplotlib.pyplot as plt

def plot_transformed_images(image_paths, transform, n=2):
    """ Plots a series of random images from image_paths.

    Will open n image paths from image_paths, transform them
    with transform and plot them side by side.

    Args:
        image_paths (list): List of target image paths. 
        transform (PyTorch Transforms): Transforms to apply to images.
        n (int, optional): Number of images to plot. Defaults to 3.
        seed (int, optional): Random seed for the random generator. Defaults to 42.
    """
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
            ax1.imshow(f) 
            ax1.set_title(f"Original \nSize: {f.size}")
            ax1.axis("off")

            # Transform and plot image
            # Note: permute() will change the shape of the image to suit matplotlib 
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            transformed_image = transform(f).permute(1, 2, 0) 
            ax2.imshow(transformed_image) 
            ax2.set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax2.axis("off")

def plot_images(image, mask, pred_image = None): 
  if pred_image == None:
      fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
      
      ax1.imshow(image.permute(1,2,0).squeeze(), cmap = 'gray')
      ax1.set_title('Image')

      ax2.imshow(mask.permute(1,2,0).squeeze(), cmap = 'gray')
      ax2.set_title('Segmented Image')
      
  elif pred_image != None :
      fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10,5))

      ax1.imshow(image.permute(1,2,0).squeeze(), cmap = 'gray')
      ax1.set_title('Image')
      
      ax2.imshow(mask.permute(1,2,0).squeeze(), cmap = 'gray')
      ax2.set_title('Segmented Image')
      
      ax3.imshow(pred_image.permute(1,2,0).squeeze(), cmap = 'gray')
      ax3.set_title('Predicted Segmentation')