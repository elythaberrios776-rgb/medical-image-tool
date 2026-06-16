import os
import cv2
import numpy as np
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from utils.dicom_reader import DicomReader


class FileManager:
    IMAGE_FILTERS = "图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff *.dcm *.dicom);;PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;DICOM (*.dcm *.dicom);;所有文件 (*)"
    SAVE_FILTERS = "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp)"

    def __init__(self):
        self.dicom_reader = DicomReader()
        self.current_file_path = None
        self.is_dicom = False

    def open_image(self, parent=None):
        file_path, _ = QFileDialog.getOpenFileName(
            parent, "打开图像", "", self.IMAGE_FILTERS
        )
        if not file_path:
            return None, False
        return self.load_image(file_path)

    def load_image(self, file_path):
        if not os.path.exists(file_path):
            return None, False
        ext = os.path.splitext(file_path)[1].lower()
        self.current_file_path = file_path
        if ext in ('.dcm', '.dicom'):
            self.is_dicom = True
            success = self.dicom_reader.read(file_path)
            if success:
                image = self.dicom_reader.get_normalized_image()
                return image, True
            return None, False
        else:
            self.is_dicom = False
            image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if image is not None:
                if len(image.shape) == 2:
                    pass
                elif image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                elif image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return image, True
            return None, False

    def save_image(self, image, parent=None):
        if image is None:
            return False
        file_path, _ = QFileDialog.getSaveFileName(
            parent, "保存图像", "", self.SAVE_FILTERS
        )
        if not file_path:
            return False
        try:
            if len(image.shape) == 3 and image.shape[2] == 3:
                save_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                save_img = image
            ext = os.path.splitext(file_path)[1].lower()
            if ext in ('.jpg', '.jpeg'):
                params = [cv2.IMWRITE_JPEG_QUALITY, 95]
            elif ext == '.png':
                params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
            else:
                params = []
            cv2.imwrite(file_path, save_img, params)
            return True
        except Exception as e:
            QMessageBox.warning(parent, "保存失败", f"无法保存图像: {e}")
            return False

    def get_image_info(self, image):
        if image is None:
            return {}
        info = {
            '宽度': image.shape[1],
            '高度': image.shape[0],
            '通道数': image.shape[2] if len(image.shape) == 3 else 1,
            '数据类型': str(image.dtype),
            '像素范围': f"{image.min()} - {image.max()}",
        }
        if self.is_dicom and self.dicom_reader.metadata:
            info['格式'] = 'DICOM'
            info.update(self.dicom_reader.metadata)
        else:
            info['格式'] = os.path.splitext(self.current_file_path)[1].upper() if self.current_file_path else '未知'
        return info
