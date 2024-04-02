import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class OCRTransform:
    FILL_VALUE = (255, 255, 255)

    def __init__(self, img_size=(64, 256), stage="train") -> None:
        self.img_size = img_size

        if stage == "train":
            Pad_or_Rezise = A.OneOf([
                A.PadIfNeeded(min_height=img_size[0], min_width=img_size[1],
                              position=A.PadIfNeeded.PositionType.BOTTOM_LEFT,
                              border_mode=cv2.BORDER_CONSTANT, value=self.FILL_VALUE),
                A.Resize(height=img_size[0], width=img_size[1])

            ], p=1.)
            Rotate = A.OneOf([
                A.SafeRotate(limit=5, interpolation=3, border_mode=cv2.BORDER_CONSTANT, value=self.FILL_VALUE, p=1)
            ])
            self.transform = A.Compose([
                A.GridDistortion(distort_limit=0.1, border_mode=0, interpolation=3,
                                 value=self.FILL_VALUE, p=.3),
                A.Defocus(radius=(1, 3), p=0.3),
                A.PixelDropout(dropout_prob=0.01, drop_value=255, p=0.3),
                A.GaussNoise(5, p=.3),
                A.RandomBrightnessContrast(.1, .2, True, p=0.3),
                A.ImageCompression(95, p=.3),
                A.OneOf([
                    A.Compose([Rotate, Pad_or_Rezise]),
                    A.Compose([Pad_or_Rezise, Rotate]),
                ], p=1),
                A.Normalize(mean=(0., 0., 0.), std=(1., 1., 1.)),
                ToTensorV2()
            ])

        else:
            self.transform = A.Compose([
                A.PadIfNeeded(min_height=img_size[0], min_width=img_size[1],
                              position=A.PadIfNeeded.PositionType.BOTTOM_LEFT,
                              border_mode=cv2.BORDER_CONSTANT, value=self.FILL_VALUE),
                A.Normalize(mean=(0., 0., 0.), std=(1., 1., 1.)),
                ToTensorV2()
            ])

    def __call__(self, image):
        height, width, _ = image.shape
        image = cv2.resize(
            image,
            (min(self.img_size[1], int(self.img_size[0]/height*width)), self.img_size[0])
        )
        return self.transform(image=image)['image']
