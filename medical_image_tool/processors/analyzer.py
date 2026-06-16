import cv2
import numpy as np
from collections import deque


class Analyzer:
    @staticmethod
    def compute_histogram(image):
        if image is None:
            return None
        if len(image.shape) == 2:
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            return {'gray': hist.flatten()}
        else:
            histograms = {}
            channels = {'R': 0, 'G': 1, 'B': 2}
            for name, idx in channels.items():
                hist = cv2.calcHist([image], [idx], None, [256], [0, 256])
                histograms[name] = hist.flatten()
            return histograms

    @staticmethod
    def sobel_edge(image, ksize=3):
        if image is None:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        return np.clip(magnitude, 0, 255).astype(np.uint8)

    @staticmethod
    def canny_edge(image, threshold1=100, threshold2=200):
        if image is None:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        return cv2.Canny(gray, threshold1, threshold2)

    @staticmethod
    def laplacian_edge(image):
        if image is None:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return np.clip(np.abs(lap), 0, 255).astype(np.uint8)

    @staticmethod
    def global_threshold(image, threshold=127):
        if image is None:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        _, result = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        return result

    @staticmethod
    def otsu_threshold(image):
        if image is None:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return result

    @staticmethod
    def adaptive_threshold(image, block_size=11, c=2):
        if image is None:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, c
        )

    @staticmethod
    def region_growing(image, seed_point, tolerance=10):
        if image is None:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image.copy()
        h, w = gray.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        seed_val = gray[seed_point[1], seed_point[0]]
        visited = np.zeros((h, w), dtype=bool)
        queue = deque([seed_point])
        while queue:
            x, y = queue.popleft()
            if x < 0 or x >= w or y < 0 or y >= h:
                continue
            if visited[y, x]:
                continue
            visited[y, x] = True
            if abs(int(gray[y, x]) - int(seed_val)) <= tolerance:
                mask[y, x] = 255
                queue.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])
        return mask

    @staticmethod
    def find_contours(image):
        if image is None:
            return None, []
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB) if len(image.shape) == 2 else image.copy()
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        return result, contours

    @staticmethod
    def measure_distance(point1, point2, pixel_spacing=None):
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        dist_pixels = np.sqrt(dx ** 2 + dy ** 2)
        if pixel_spacing is not None:
            dist_mm = dist_pixels * pixel_spacing
            return dist_pixels, dist_mm
        return dist_pixels, None

    @staticmethod
    def measure_area(contour, pixel_spacing=None):
        area_pixels = cv2.contourArea(contour)
        if pixel_spacing is not None:
            area_mm2 = area_pixels * (pixel_spacing ** 2)
            return area_pixels, area_mm2
        return area_pixels, None

    @staticmethod
    def measure_angle(point1, vertex, point2):
        v1 = np.array([point1[0] - vertex[0], point1[1] - vertex[1]])
        v2 = np.array([point2[0] - vertex[0], point2[1] - vertex[1]])
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        return angle
