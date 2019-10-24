import csv
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    with open('ISIC2018_Task3_Training_GroundTruth.csv', 'r', newline='') as f:
        y = []
        for i, row in enumerate(csv.DictReader(f)):
            label = int(float(row['MEL'])) # 1 is melanoma, 0 is non-melanoma
            y.append(label)

        labels, occurrences = np.unique(y, return_counts=True)
        plt.bar(labels, height=occurrences)
        print(occurrences, occurrences/10015)
        plt.xticks(labels, ['Non-melanoma', 'Melanoma'])
        plt.ylabel('Number of samples');
        plt.xlabel('Label');
        plt.show()
        plt.close()
