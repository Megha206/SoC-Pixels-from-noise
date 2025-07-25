#!/usr/bin/env python3

"""
6.101 Lab:
Image Processing
"""

import math

from PIL import Image

# NO ADDITIONAL IMPORTS ALLOWED!


def get_pixel(image, row, col):
    """
    Given row index and column index, this function
    returns the pixel from image["pixels"] 
    based on the boundary behavior
    """
    return image["pixels"][col, row]


def set_pixel(image, row, col, color):
    image["pixels"][row*image["width"] + col] = color


def apply_per_pixel(image, func):
    
    """
    Apply the given function to each pixel in an image and return the resulting image.

    Parameters:
    image (dict): A dictionary representing an image, with keys:
                    - "height" (int): The height of the image.
                    - "width" (int): The width of the image.
                    - "pixels" (list): A list of pixel values.
    func (callable): A function that takes a single pixel value and returns a new pixel value.

    Returns:
    dict: A new image dictionary with the same dimensions as the input image, but with each
            pixel value modified by the given function.
    """
    new_pixels= [func(p) for p in image["pixels"]]
    return {
        'height': image['height'],
        'width': image['widht'],
        'pixels': new_pixels

    }
    
    raise NotImplementedError


def inverted(image):
    return apply_per_pixel(image, lambda color: 255-color)


# HELPER FUNCTIONS

def correlate(image, kernel, boundary_behavior):
    """
    Compute the result of correlating the given image with the given kernel.
    `boundary_behavior` will one of the strings "zero", "extend", or "wrap",
    and this function will treat out-of-bounds pixels as having the value zero,
    the value of the nearest edge, or the value wrapped around the other edge
    of the image, respectively.

    if boundary_behavior is not one of "zero", "extend", or "wrap", return
    None.

    Otherwise, the output of this function should have the same form as a 6.101
    image (a dictionary with "height", "width", and "pixels" keys), but its
    pixel values do not necessarily need to be in the range [0,255], nor do
    they need to be integers (they should not be clipped or rounded at all).

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.

    DESCRIBE YOUR KERNEL REPRESENTATION HERE
    Kernel representation: 2D list in python
    Ex: k=[
    [0,0,0],
    [0,1,0],       ---- The identitiy filter 
    [0,0,0]
    ]
        
    """

    
    if boundary_behavior not in ["zero", "extend", "wrap"]:
        return None

    height, width = image["height"], image["width"]
    k_size = len(kernel)
    k_half = k_size // 2

    def get_pixel_with_bounds(r, c):
        if 0 <= r < height and 0 <= c < width:
            return image["pixels"][r * width + c]
        if boundary_behavior == "zero":
            return 0
        elif boundary_behavior == "extend":
            r = min(max(r, 0), height - 1)
            c = min(max(c, 0), width - 1)
            return image["pixels"][r * width + c]
        elif boundary_behavior == "wrap":
            r = r % height
            c = c % width
            return image["pixels"][r * width + c]

    result_pixels = []

    for r in range(height):
        for c in range(width):
            acc = 0
            for i in range(k_size):
                for j in range(k_size):
                    kr = r + i - k_half
                    kc = c + j - k_half
                    acc += kernel[i][j] * get_pixel_with_bounds(kr, kc)
            result_pixels.append(acc)

    return {
        "height": height,
        "width": width,
        "pixels": result_pixels
    }


    



def round_and_clip_image(image):
    """
    Given a dictionary, ensure that the values in the "pixels" list are all
    integers in the range [0, 255].

    All values should be converted to integers using Python's `round` function.

    Any locations with values higher than 255 in the input should have value
    255 in the output; and any locations with values lower than 0 in the input
    should have value 0 in the output.
    """

    clipped_pixels = [
        max(0, min(255, round(p))) for p in image["pixels"]
    ]
    return {
        "height": image["height"],
        "width": image["width"],
        "pixels": clipped_pixels
    }


def make_box_blur_kernel(n):
    value = 1 / (n * n)
    kernel = []  

    for i in range(n):
        row = []  
        for j in range(n):
            row.append(value)  
        kernel.append(row) 

    return kernel

# FILTERS

def blurred(image, kernel_size):
    """
    Return a new image representing the result of applying a box blur (with the
    given kernel size) to the given input image.

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.
    """
    # first, create a representation for the appropriate n-by-n kernel (you may
    # wish to define another helper function for this)

    # then compute the correlation of the input image with that kernel

    # and, finally, make sure that the output is a valid image (using the
    # helper function from above) before returning it.
    kernel = make_box_blur_kernel(kernel_size)
    correlated = correlate(image, kernel, boundary_behavior="extend")
    return round_and_clip_image(correlated)
    

def sharpened(image, n):
    """
    Return a new image which is sharper than the input image.

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.
    Output list should be clipped.
    For this function, I have used the method as suggested. 
    --First create a blurring kernel and blur the given image
    --Then calculate each pixel value for the sharpened image using the formula:
             S=2*I-B
    --Now, clip the pixel values of the sharpened image
    """
    kernel = make_box_blur_kernel(n)
    blurred_img = correlate(image, kernel, boundary_behavior="extend")

    sharpened_pixels = []
    for i in range(len(image["pixels"])):
        original = image["pixels"][i]
        blurred = blurred_img["pixels"][i]
        sharpened = 2 * original - blurred
        sharpened_pixels.append(sharpened)
    temp_img = {
        "height": image["height"],
        "width": image["width"],
        "pixels": sharpened_pixels
    }
    return round_and_clip_image(temp_img)

    

def edges(image):
    """
    Here we're using two sobel filters, one for horizontal edge and the other for vertical edge detection. 
    the resutlant "edge" is like the magnitude of the gradient at each point in the image 
    """

    K1 = [
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ]

    K2 = [
        [ 1,  0, -1],
        [ 2,  0, -2],
        [ 1,  0, -1]
    ]

   
    H_1 = correlate(image, K1, boundary_behavior="extend")
    H_2 = correlate(image, K2, boundary_behavior="extend")

    
    final_pixels = []
    for p1, p2 in zip(H_1["pixels"], H_2["pixels"]):
        mag = round((p1 ** 2 + p2 ** 2) ** 0.5)
        final_pixels.append(mag)


    final_image = {
        "height": image["height"],
        "width": image["width"],
        "pixels": final_pixels
    }

 
    return round_and_clip_image(final_image)


def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns a dictionary
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_greyscale_image("test_images/cat.png")
    """
    with open(filename, "rb") as img_handle:
        img = Image.open(img_handle)
        img_data = img.getdata()
        if img.mode.startswith("RGB"):
            pixels = [round(.299 * p[0] + .587 * p[1] + .114 * p[2])
                      for p in img_data]
        elif img.mode == "LA":
            pixels = [p[0] for p in img_data]
        elif img.mode == "L":
            pixels = list(img_data)
        else:
            raise ValueError(f"Unsupported image mode: {img.mode}")
        width, height = img.size
        return {"height": height, "width": width, "pixels": pixels}


def save_greyscale_image(image, filename, mode="PNG"):
    """
    Saves the given image to disk or to a file-like object.  If filename is
    given as a string, the file type will be inferred from the given name.  If
    filename is given as a file-like object, the file type will be determined
    by the "mode" parameter.
    """
    out = Image.new(mode="L", size=(image["width"], image["height"]))
    out.putdata(image["pixels"])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


if __name__ == "__main__":
    # code in this block will only be run when you explicitly run your script,
    # and not when the tests are being run.  this is a good place for
    # generating images, etc.
    pass
