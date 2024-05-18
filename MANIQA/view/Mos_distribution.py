import csv
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def mos_load(csv_file):
    imgname = []
    mos_all = []
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            imgname.append(row["img_path"])
            mos = np.array(float(row["mos"])).astype(np.float32)
            mos_all.append(mos)

    sample = []
    for img_name, mos_score in zip(imgname, mos_all):
        sample.append(mos_score)

    return sample

if __name__ == '__main__':
    samsung_train_path = "/home/compu/LJC/samsung_LK99/dataset/train_mos.csv"
    samsung_valid_path = "/home/compu/LJC/samsung_LK99/dataset/valid_mos.csv"

    sample = mos_load(samsung_train_path)
    # import pdb; pdb.set_trace()

    # 히스토그램 그리기
    plt.hist(sample, bins=20, range=(0.0, 10.0), edgecolor='black')
    plt.xlabel('MOS (Mean Opinion Score)')
    plt.ylabel('Frequency')
    plt.title('MOS Distribution')
    plt.grid(True)

    # 그래프 표시
    plt.show()
