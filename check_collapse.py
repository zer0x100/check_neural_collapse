""" Check Neural Collapse
"""
import numpy as np
import matplotlib.pyplot as plt
import os

dir = "./trained_2"
path = os.path.join(dir, "OUT_results.npz")
data = np.load(path)

#parameters
nclass = 10

train_class_ftrs_mean = data["train_class_ftrs_mean"]
test_class_ftrs_mean = data["test_class_ftrs_mean"]
train_ftrs_mean = np.mean(train_class_ftrs_mean, axis=1)
test_ftrs_mean = np.mean(test_class_ftrs_mean, axis=1)
train_class_ftrs_mean_c = np.array([subarray - mean for subarray, mean in zip(train_class_ftrs_mean, train_ftrs_mean)])
test_class_ftrs_mean_c = np.array([subarray - mean for subarray, mean in zip(test_class_ftrs_mean, test_ftrs_mean)])

#Check if class-means become equinorm
train_norm_var = []
test_norm_var = []
for class_means in train_class_ftrs_mean_c:
    class_means_norm = np.sqrt(np.sum(class_means**2, axis=1))
    train_norm_var.append(np.std(class_means_norm) / np.mean(class_means_norm))
for class_means in test_class_ftrs_mean_c:
    class_means_norm = np.sqrt(np.sum(class_means**2, axis=1))
    test_norm_var.append(np.std(class_means_norm) / np.mean(class_means_norm))

plt.plot(train_norm_var, label='train')
plt.plot(test_norm_var, label='test')
plt.title("Std_c(|mu_c - mu_G|)/Avg_c(|mu_c - mu_G)")
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()
plt.savefig(os.path.join(dir, 'check_equinorm.png'))
plt.show()

#Check class-means approach equiangularity
#Check class-means approach maximal-angle equiangularitys
train_cos_std= []
test_cos_std = []
train_cos_c = []
test_cos_c = []
for class_means in train_class_ftrs_mean_c:
    coses = []
    for x in class_means:
        for y in class_means:
            if (x == y).all():
                continue
            coses.append(np.dot(x, y) / np.linalg.norm(x) / np.linalg.norm(y))
    train_cos_std.append(np.std(coses))
    train_cos_c.append(np.abs(np.array(coses) + 1/(nclass-1)).mean())

for class_means in test_class_ftrs_mean_c:
    coses = []
    for x in class_means:
        for y in class_means:
            if (x == y).all():
                continue
            coses.append(np.dot(x, y) / np.linalg.norm(x) / np.linalg.norm(y))
    test_cos_std.append(np.std(coses))
    test_cos_c.append(np.abs(np.array(coses) + 1/(nclass-1)).mean())
plt.plot(train_cos_std, label='train')
plt.plot(test_cos_std, label='test')
plt.title("Std_b_c cos(b, c)")
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()
plt.savefig(os.path.join(dir, 'check_equiang.png'))
plt.show()

plt.plot(train_cos_c, label='train')
plt.plot(test_cos_c, label='test')
plt.title("Avg_b_c|cos(b, c) + 1/(nclass-1)|")
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()
plt.savefig(os.path.join(dir, 'check_max_ang_equi.png'))
plt.show()
