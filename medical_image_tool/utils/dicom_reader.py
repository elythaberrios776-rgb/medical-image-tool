import pydicom
import numpy as np
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QDialogButtonBox


class DicomReader:
    def __init__(self):
        self.dataset = None
        self.pixel_array = None
        self.metadata = {}

    def read(self, file_path):
        try:
            self.dataset = pydicom.dcmread(file_path)
            self.pixel_array = self._get_pixel_array()
            self.metadata = self._extract_metadata()
            return True
        except Exception as e:
            print(f"读取DICOM文件失败: {e}")
            return False

    def _get_pixel_array(self):
        if self.dataset is None:
            return None
        pixel_array = self.dataset.pixel_array.astype(np.float64)
        if hasattr(self.dataset, 'RescaleSlope') and hasattr(self.dataset, 'RescaleIntercept'):
            slope = float(self.dataset.RescaleSlope)
            intercept = float(self.dataset.RescaleIntercept)
            pixel_array = pixel_array * slope + intercept
        return pixel_array

    def _extract_metadata(self):
        if self.dataset is None:
            return {}
        metadata = {}
        tags = {
            'PatientName': (0x0010, 0x0010),
            'PatientID': (0x0010, 0x0020),
            'PatientBirthDate': (0x0010, 0x0030),
            'PatientSex': (0x0010, 0x0040),
            'StudyDate': (0x0008, 0x0020),
            'Modality': (0x0008, 0x0060),
            'StudyDescription': (0x0008, 0x1030),
            'SeriesDescription': (0x0008, 0x103E),
            'InstitutionName': (0x0008, 0x0080),
            'Manufacturer': (0x0008, 0x0070),
            'SliceThickness': (0x0018, 0x0050),
            'KVP': (0x0018, 0x0060),
            'Rows': (0x0028, 0x0010),
            'Columns': (0x0028, 0x0011),
            'PixelSpacing': (0x0028, 0x0030),
            'BitsAllocated': (0x0028, 0x0100),
            'BitsStored': (0x0028, 0x0101),
            'WindowCenter': (0x0028, 0x1050),
            'WindowWidth': (0x0028, 0x1051),
        }
        for name, tag in tags.items():
            if tag in self.dataset:
                value = self.dataset[tag].value
                metadata[name] = str(value)
        return metadata

    def get_normalized_image(self):
        if self.pixel_array is None:
            return None
        arr = self.pixel_array.copy()
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-10) * 255
        return arr.astype(np.uint8)

    def get_windowed_image(self, center=None, width=None):
        if self.pixel_array is None:
            return None
        if center is None:
            center = float(self.metadata.get('WindowCenter', 127.5))
        if width is None:
            width = float(self.metadata.get('WindowWidth', 255))
        arr = self.pixel_array.copy()
        lower = center - width / 2
        upper = center + width / 2
        arr = np.clip(arr, lower, upper)
        arr = (arr - lower) / (upper - lower) * 255
        return arr.astype(np.uint8)

    @staticmethod
    def show_metadata_dialog(metadata, parent=None):
        dialog = QDialog(parent)
        dialog.setWindowTitle("DICOM元数据")
        dialog.setMinimumSize(500, 400)
        layout = QVBoxLayout(dialog)
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        lines = []
        for key, value in metadata.items():
            lines.append(f"<b>{key}</b>: {value}")
        text_edit.setHtml("<br>".join(lines))
        layout.addWidget(text_edit)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(dialog.accept)
        layout.addWidget(button_box)
        return dialog
