import cv2
import numpy as np


class Transformer:
    @staticmethod
    def rotate(image, angle=90, center=None, scale=1.0):
        if image is None:
            return None
        h, w = image.shape[:2]
        if center is None:
            center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        return cv2.warpAffine(image, matrix, (w, h))

    @staticmethod
    def flip_horizontal(image):
        if image is None:
            return None
        return cv2.flip(image, 1)

    @staticmethod
    def flip_vertical(image):
        if image is None:
            return None
        return cv2.flip(image, 0)

    @staticmethod
    def resize(image, scale_x=1.0, scale_y=1.0):
        if image is None:
            return None
        h, w = image.shape[:2]
        new_w = int(w * scale_x)
        new_h = int(h * scale_y)
        new_w = max(1, new_w)
        new_h = max(1, new_h)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def translate(image, tx=0, ty=0):
        if image is None:
            return None
        h, w = image.shape[:2]
        matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(image, matrix, (w, h))

    @staticmethod
    def erode(image, kernel_size=3, iterations=1):
        if image is None:
            return None
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.erode(image, kernel, iterations=iterations)

    @staticmethod
    def dilate(image, kernel_size=3, iterations=1):
        if image is None:
            return None
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.dilate(image, kernel, iterations=iterations)

    @staticmethod
    def morph_open(image, kernel_size=3):
        if image is None:
            return None
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    @staticmethod
    def morph_close(image, kernel_size=3):
        if image is None:
            return None
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    @staticmethod
    def affine_transform(image, src_points, dst_points):
        if image is None:
            return None
        matrix = cv2.getAffineTransform(np.float32(src_points), np.float32(dst_points))
        h, w = image.shape[:2]
        return cv2.warpAffine(image, matrix, (w, h))

    @staticmethod
    def crop(image, x, y, w, h):
        if image is None:
            return None
        img_h, img_w = image.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        return image[y:y + h, x:x + w]
