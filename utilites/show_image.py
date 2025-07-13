
import matplotlib.pyplot as plt


def show_all_plants(train_dataset):
    plt.figure(figsize=(14, 14))
    for i in range(8):
        plt.subplot(1, 8, i+1)
        plt.xticks([])
        plt.yticks([])
        images = (train_dataset[i][0]*255).transpose(2,0).int().numpy()
        plt.imshow(images, cmap=plt.cm.binary)
        plt.text(10, 12, 'healthy', color='white', backgroundcolor='black', fontsize=10)

    plt.figure(figsize=(14, 14))
    for i in range(8):
        plt.subplot(1, 8, i + 1)
        plt.xticks([])
        plt.yticks([])
        images = (train_dataset[i + 900][0] * 255).transpose(2, 0).int().numpy()
        plt.imshow(images, cmap=plt.cm.binary)
        plt.text(10, 12, 'sick_early', color='white', backgroundcolor='black', fontsize=10)

    plt.figure(figsize=(14, 14))
    for i in range(8):
        plt.subplot(1, 8, i + 1)
        plt.xticks([])
        plt.yticks([])
        images = (train_dataset[i + 1600][0] * 255).transpose(2, 0).int().numpy()
        plt.imshow(images, cmap=plt.cm.binary)
        plt.text(10, 12, 'sick_late', color='white', backgroundcolor='black', fontsize=10)

    plt.show()