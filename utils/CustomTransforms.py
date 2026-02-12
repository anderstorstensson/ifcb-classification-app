import torchvision.transforms.functional as F
import numpy as np


class SquarePad:
    def __call__(self, image):
        s = image.size()

        max_wh = np.max([s[-1], s[-2]])
        hp = int((max_wh - s[-1]) / 2)
        vp = int((max_wh - s[-2]) / 2)

        #sample each of the four corners of the image to get an approsimate background colour
        top_left_value = image[0][0][0]
        bottom_left_value = image[0][-1][0]
        top_right_value = image[0][0][-1]
        bottom_right_value = image[0][-1][-1]

        average_value = (top_left_value.item() + bottom_left_value.item() + top_right_value.item() + bottom_right_value.item()) / 4

        padding = (hp, vp, hp, vp)

        return F.pad(image, padding, average_value, 'constant')
