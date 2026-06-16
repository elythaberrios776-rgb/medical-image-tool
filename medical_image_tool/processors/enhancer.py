import cv2
import numpy as np


class Enhancer:
    @staticmethod
    def adjust_brightness(image, value=0):
        if image is None:
            return None
        result = image.astype(np.float64) + value
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def adjust_contrast(image, alpha=1.0):
        if image is None:
            return None
        result = image.astype(np.float64) * alpha
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def adjust_brightness_contrast(image, brightness=0, contrast=1.0):
        if image is None:
            return None
        result = image.astype(np.float64) * contrast + brightness
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def histogram_equalization(image):
        if image is None:
            return None
        if len(image.shape) == 2:
            return cv2.equalizeHist(image)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

    @staticmethod
    def clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
        if image is None:
            return None
        clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        if len(image.shape) == 2:
            return clahe_obj.apply(image)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = clahe_obj.apply(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

    @staticmethod
    def gaussian_blur(image, ksize=5):
        if image is None:
            return None
        ksize = ksize if ksize % 2 == 1 else ksize + 1
        return cv2.GaussianBlur(image, (ksize, ksize), 0)

    @staticmethod
    def median_blur(image, ksize=5):
        if image is None:
            return None
        ksize = ksize if ksize % 2 == 1 else ksize + 1
        return cv2.medianBlur(image, ksize)

    @staticmethod
    def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
        if image is None:
            return None
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    @staticmethod
    def add_gaussian_noise(image, mean=0, sigma=25):
        if image is None:
            return None
        noise = np.random.normal(mean, sigma, image.shape)
        result = image.astype(np.float64) + noise
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def add_salt_pepper_noise(image, amount=0.02):
        if image is None:
            return None
        result = image.copy()
        total_pixels = image.size
        num_salt = int(np.ceil(amount * total_pixels / 2))
        num_pepper = int(np.ceil(amount * total_pixels / 2))
        coords_salt = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
        coords_pepper = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
        if len(image.shape) == 2:
            result[coords_salt[0], coords_salt[1]] = 255
            result[coords_pepper[0], coords_pepper[1]] = 0
        else:
            result[coords_salt[0], coords_salt[1], :] = 255
            result[coords_pepper[0], coords_pepper[1], :] = 0
        return result

    @staticmethod
    def invert(image):
        if image is None:
            return None
        return 255 - image

    @staticmethod
    def gamma_correction(image, gamma=1.0):
        if image is None:
            return None
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype(np.uint8)
        return cv2.LUT(image, table)
