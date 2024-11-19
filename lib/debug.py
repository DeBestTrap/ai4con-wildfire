import matplotlib.pyplot as plt

from lib.data import remove_filler_values, rgb_float_to_uint8

def plot_burned_area_mask(mask):
    plt.figure(figsize=(10, 10))
    plt.imshow(remove_filler_values(mask))
    plt.show()

def plot_highlighted_rgb(rgb_norm, burned_area_mask):
    # Create an RGB composite with masked areas highlighted
    highlighted_rgb = rgb_float_to_uint8(rgb_norm.copy())
    burned_color = [255, 0, 0]  # Red color for burned areas

    # Apply mask: set burned areas to red
    highlighted_rgb[burned_area_mask == 1] = burned_color

    # Plot
    plot_rgb(highlighted_rgb)

def plot_rgb_norm(rgb_norm):
    '''
    Plot a RGB image stored with float
    '''
    plot_rgb(rgb_float_to_uint8(rgb_norm))

def plot_rgb(rgb):
    '''
    Plot a RGB image stored with uint8
    '''
    plt.figure(figsize=(20, 20))
    plt.imshow(rgb)
    plt.title("RGB Composite with Burned Areas Highlighted")
    plt.axis('off')
    plt.show()
