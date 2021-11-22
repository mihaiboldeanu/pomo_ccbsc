from datetime import datetime
from enum import Enum

from PIL import Image
import numpy as np

class ImageType(Enum):
    BLACK_WHITE = 1


class MimeType(Enum):
    IMAGE_PNG = 1


class Photo:
    def __init__(self, id: str, device_id: str, timestamp: datetime, image: Image, mime_type: MimeType = MimeType.IMAGE_PNG, image_type: ImageType = ImageType.BLACK_WHITE):
        self.id = id
        self.device_id = device_id
        self.timestamp = timestamp
        self.image = image
        self.mime_type = mime_type
        self.image_type = image_type

class Photo_fake:
    def __init__(self, path):
        self.id = path.split("\\")[-1]
        self.image = self.get_image(path)
        
    def get_image(self,path):
        
        img = Image.open(path).convert("L")
       
        image = np.array(img)
        
        return image

    