import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class OCRTransform:
    def __init__(self, img_size=(64, 256), stage="train") -> None:
        self.img_size = img_size

        if stage == "train":
            self.transform = A.Compose([
                A.ShiftScaleRotate(shift_limit=0, scale_limit=(0., 0.15), rotate_limit=1,
                                   border_mode=0, interpolation=3, value=[255, 255, 255], p=0.7),
                A.GridDistortion(distort_limit=0.1, border_mode=0, interpolation=3,
                                 value=[255, 255, 255], p=.5),
                A.GaussNoise(10, p=.2),
                A.RandomBrightnessContrast(.05, (-.2, 0), True, p=0.2),
                A.ImageCompression(95, p=.3),
                A.PadIfNeeded(min_height=img_size[0], min_width=img_size[1],
                              position=A.PadIfNeeded.PositionType.TOP_LEFT,
                              border_mode=cv2.BORDER_CONSTANT, value=[255, 255, 255]),
                A.ToGray(always_apply=True),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.PadIfNeeded(min_height=img_size[0], min_width=img_size[1],
                              position=A.PadIfNeeded.PositionType.TOP_LEFT,
                              border_mode=cv2.BORDER_CONSTANT, value=[255, 255, 255]),
                A.ToGray(always_apply=True),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2()
            ])

    def __call__(self, image):
        height, width, _ = image.shape
        image = cv2.resize(image, (min(self.img_size[1], int(self.img_size[0]/height*width)), self.img_size[0]))
        return self.transform(image=image)['image']
