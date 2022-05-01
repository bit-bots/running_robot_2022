import math
import cv2
import numpy as np
from torch.utils.data import Dataset

class DummyData(Dataset):
    """Easy debug dataset."""

    def __init__(self, sequence_length=20, img_size=224, transform=None):
        self.sequence_length = sequence_length
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        sample = []
        for i in range(self.sequence_length):
            img = np.zeros((self.img_size, self.img_size, 3))
            if idx % 2 == 0:
                direction = 1
            else:
                direction = -1
            vector_x = math.sin(direction*(i + 100 * idx)*0.1) * 50
            vector_y = math.cos(direction*(i + 100 * idx)*0.1) * 50
            point = (int(img.shape[0]/2 + vector_x), int(img.shape[1]/2 + vector_y))
            # Only show circle on certain images to test the transformer
            if np.random.randint(2):
                img = cv2.circle(img, point, 10, (255, 255, 255), -1)
            sample.append((img, point))

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":
    data = DummyData()

    for elements in data:
        print("New sequence\n-----------------------------------------")
        for sample in elements:
            canvas = sample[0]
            canvas = cv2.circle(canvas, sample[1], 4, (0, 0, 255), -1)
            print(sample[1])
            cv2.imshow("Sample", canvas)
            cv2.waitKey(0)
    cv2.destroyAllWindows()

