import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2


class OCRTransform:
    def __init__(self, img_size=(64, 256), stage="train") -> None:
        self.img_size = img_size
        if stage == "train":
            Pad_or_Rezise = A.OneOf([
                A.PadIfNeeded(min_height=img_size[0], min_width=img_size[1],
                              position=A.PadIfNeeded.PositionType.RANDOM,
                              border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)),
                A.Resize(height=img_size[0], width=img_size[1])

            ], p=1.)
            Curve_or_Rotate = A.OneOf([
                Curve(p=1),
                A.SafeRotate(limit=10, interpolation=3, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0), p=1)
            ])
            self.transform = A.Compose([
                A.GridDistortion(distort_limit=0.1, border_mode=0, interpolation=3,
                                 value=[0, 0, 0], p=.5),
                A.Defocus(radius=(1, 3), p=0.5),
                A.PixelDropout(dropout_prob=0.01, drop_value=255, p=0.5),
                A.GaussNoise(10, p=.5),
                A.RandomBrightnessContrast(.1, .2, True, p=0.5),
                A.ImageCompression(95, p=.5),
                A.OneOf([
                    A.Compose([Curve_or_Rotate, Pad_or_Rezise]),
                    A.Compose([Pad_or_Rezise, Curve_or_Rotate]),
                ], p=1),
                A.Normalize(mean=(0., 0., 0.), std=(1., 1., 1.)),
                ToTensorV2()
            ])

        else:
            self.transform = A.Compose([
                A.PadIfNeeded(min_height=img_size[0], min_width=img_size[1],
                              position=A.PadIfNeeded.PositionType.CENTER,
                              border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)),
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


class Curve:
    def __init__(self, p=1.):
        self.p = p

    def __call__(self, **img):
        img = img['image']
        if np.random.uniform(0, 1) > self.p:
            return {"image": img}

        height, width = img.shape[:2]
        # Generate a meshgrid of the same shape as the image
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        # Design warping
        # Normalize the coordinates to the range [-1, 1]
        x = (x - (width / 2)) / (width / 2)
        y = (y - (height / 2)) / (height / 2)
        # # Map the distorted coordinates back to the image space
        temp = np.random.uniform(0, 1)
        if temp > 0.5:
            x = (x + np.sin(y*2)*0.1).astype(np.float32)
        else:
            x = (x + np.sin(y*2)*-0.1).astype(np.float32)
        temp = np.random.uniform(0, 1)
        curve = np.random.uniform(0.2, 0.4)
        if temp > 0.5:
            y = (y + np.cos(x*2)*-curve).astype(np.float32)
        else:
            y = (y + np.cos(x*2)*curve).astype(np.float32)

        x = ((x * (width / 2)) + (width / 2)).astype(np.float32)
        y = ((y * (height / 2)) + (height / 2)).astype(np.float32)

        # Remap the image using the distorted coordinates
        curved_image = cv2.remap(img, x, y, cv2.INTER_LINEAR)

        return {"image": curved_image}
