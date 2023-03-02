"""
Clustering illuminant chroma using K-means algorithm
Use euclidian distance as distance metric

Ignored place no (outliers) : 496,497,498,544,709,717
"""
import json,pickle,scipy,os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from scipy import io

nus_root = '../dataset/NUS-8/'
N_clusters = 7
CAMERA = 'NikonD5200'

meta = io.loadmat(os.path.join(nus_root,CAMERA,f'{CAMERA}_gt.mat'))['groundtruth_illuminants']
meta = (meta / meta[:,[1]])[:,[0,2]]

index = []
illum_chroma = []
colors = []

# gather illuminant chroma
for i in range(len(meta)):
    if (meta[i,0]>10 or meta[i,1]>3):
        print(i,meta[i])
    else:
        illum_chroma.append(meta[i])
        color = [item / max(meta[i][0],1.,meta[i][1]) for item in [meta[i][0],1.,meta[i][1]]]
        colors.append(color)

# visualize initial illum chroma plot
illum_chroma = np.array(illum_chroma)
for i in range(len(illum_chroma)):
    plt.plot(illum_chroma[i,:1],illum_chroma[i,1:], c=color, marker='o', alpha=0.5, markersize=5)
plt.savefig(f'cluster_result/{CAMERA}_initinal_plot.png')
plt.clf()

# illum_chroma = np.array(illum_chroma)
# for i in range(len(illum_chroma)):
#     if i <= 181:
#         split_color = 'r'
#     elif i <= 233:
#         split_color = 'g'
#     elif i <= 259:
#         split_color = 'b'
#     plt.plot(illum_chroma[i,:1],illum_chroma[i,1:], c=split_color, marker='o', alpha=0.5, markersize=5)
# plt.savefig(f'cluster_result/{CAMERA}_initinal_split_plot.png')
# plt.clf()

# K means clustering & plot
kmeans = KMeans(n_clusters=N_clusters,random_state=0).fit(illum_chroma)
# cluster_centers contains coordinates of the centerpoints (N_clusters)
# ex) [[1.5, 1.2], [1.1, 1.3], ...]
cluster_centers = kmeans.cluster_centers_
col = cluster_centers[:,0]
idx = np.argsort(col)
cluster_centers = cluster_centers[idx]
print(f"centerpoints of clusters")
print(cluster_centers)
# labels contains the clustered class (label). ex) 0, 1, 1, 4, 6 ...
labels = kmeans.labels_

# plot all illum chroma points using cluster color (c=labels)
plt.scatter(illum_chroma[:,:1],illum_chroma[:,1:],c=labels,s=3**2)
# plot centerpoints of clusters
plt.plot(cluster_centers[:,:1],cluster_centers[:,1:], 'rx')
plt.savefig(f'cluster_result/{CAMERA}_{N_clusters}_cluster_plot.png')
plt.clf()

# plot distributions of labels
dist = []
for i in range(N_clusters):
    dist.append(len(labels[labels==i]))
plt.plot(range(N_clusters),dist)
plt.savefig(f'cluster_result/{CAMERA}_{N_clusters}_dist_labels.png')
plt.clf()

# plot center point of clusters
plt.plot(cluster_centers[:,:1],cluster_centers[:,1:], 'bo')
plt.savefig(f'cluster_result/{CAMERA}_centers_plot_{N_clusters}.png')
plt.clf()
# save center coordinates of clusters
np.save(f'cluster_result/{CAMERA}_cluster_{N_clusters}.npy',cluster_centers)

# # insert illuminant cluster data to Json & save Json
# for i,idx in enumerate(index):
#     place,light = idx.split('_')
#     cluster_no = labels[i]
#     meta[place][light+'_cluster'] = int(cluster_no)

# with open(f'{CAMERA}_meta_{N_clusters}_clustered.json', 'w') as out_file:
#     json.dump(meta, out_file, indent=4)

# with open(f'cluster_result/{CAMERA}_{N_clusters}_kmeans', 'wb') as pickle_file:
#     pickle.dump({'kmeans':kmeans}, pickle_file)