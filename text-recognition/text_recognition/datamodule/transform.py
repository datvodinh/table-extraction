import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class OCRTransform:
    FILL_VALUE = (255, 255, 255)

    def __init__(self, img_size=(64, 256), stage="train") -> None:
        self.img_size = img_size

        if stage == "train":
            self.transform = A.Compose([
                A.GridDistortion(distort_limit=0.1, border_mode=0, interpolation=3,
                                 value=self.FILL_VALUE, p=.3),
                A.Defocus(radius=(1, 3), p=0.3),
                A.PixelDropout(dropout_prob=0.01, drop_value=255, p=0.3),
                A.GaussNoise(5, p=.3),
                A.RandomBrightnessContrast(.1, .2, True, p=0.3),
                A.ImageCompression(95, p=.3),
                A.PadIfNeeded(min_height=img_size[0], min_width=img_size[1],
                              position=A.PadIfNeeded.PositionType.BOTTOM_LEFT,
                              border_mode=cv2.BORDER_CONSTANT, value=self.FILL_VALUE),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.PadIfNeeded(min_height=img_size[0], min_width=img_size[1],
                              position=A.PadIfNeeded.PositionType.BOTTOM_LEFT,
                              border_mode=cv2.BORDER_CONSTANT, value=self.FILL_VALUE),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2()
            ])

    def __call__(self, image, ratio):
        height, width, _ = image.shape
        image = cv2.resize(image, (self.img_size[0] * ratio, self.img_size[0]))
        return self.transform(image=image)['image']
