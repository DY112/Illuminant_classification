"""
Clustering illuminant chroma using K-means algorithm
Use euclidian distance as distance metric

Ignored place no (outliers) : 496,497,498,544,709,717
"""
import json,pickle,os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

N_clusters = 3
CAMERA = 'galaxy'
USE_RB_CHROMATICITY = True
USE_G1_CHROMATICITY = False
CLUSTER_SAVE_DIR = '../Slot-IID/datasets/rb_clusters/'

with open(f'meta_{CAMERA}.json','r') as json_file:
    meta = json.load(json_file)

index = []
illum_chroma = []
l1_chroma = []
sub_chroma = []
colors = []
l1_colors = []
sub_colors = []

# gather illuminant information
for place in meta:
    place_meta = meta[place]
    
    l1 = place_meta['Light1']
    if USE_RB_CHROMATICITY:
        l1 = l1 / np.sum(l1)
    elif USE_G1_CHROMATICITY:
        pass

    if True in np.isnan(l1) or l1[0] > 10:
        print(place,l1)
    else:
        if USE_G1_CHROMATICITY:
            illum_chroma.append([l1[0],l1[2]])
            l1_chroma.append([l1[0],l1[2]])
        elif USE_RB_CHROMATICITY:
            illum_chroma.append(l1)
            l1_chroma.append(l1)
        color = l1 / max(l1)
        colors.append(color)
        l1_colors.append(color)
        index.append(place+'_Light1')

    l2 = place_meta['Light2']
    if USE_RB_CHROMATICITY:
        l2 = l2 / np.sum(l2)
    elif USE_G1_CHROMATICITY:
        pass

    if True in np.isnan(l2) or l2[0] > 10:
        print(place,l2)
    else:
        if USE_G1_CHROMATICITY:
            illum_chroma.append([l2[0],l2[2]])
            sub_chroma.append([l2[0],l2[2]])
        elif USE_RB_CHROMATICITY:
            illum_chroma.append(l2)
            sub_chroma.append(l2)
        color = l2 / max(l2)
        colors.append(color)
        sub_colors.append(color)
        index.append(place+'_Light2')
    
    if place_meta['NumOfLights'] == 3:
        l3 = place_meta['Light3']
        if USE_RB_CHROMATICITY:
            l3 = l3 / np.sum(l3)
        elif USE_G1_CHROMATICITY:
            pass

        if True in np.isnan(l3) or l3[0] > 10:
            print(place,l3)
        else:
            if USE_G1_CHROMATICITY:
                illum_chroma.append([l3[0],l3[2]])
                sub_chroma.append([l3[0],l3[2]])
            elif USE_RB_CHROMATICITY:
                illum_chroma.append(l3)
                sub_chroma.append(l3)
            color = l3 / max(l3)
            colors.append(color)
            sub_colors.append(color)
            index.append(place+'_Light3')

# visualize initial illum chroma plot
illum_chroma = np.array(illum_chroma)
for i in range(len(illum_chroma)):
    if USE_G1_CHROMATICITY:
        plt.plot(illum_chroma[i,:1],illum_chroma[i,1:], c=colors[i], marker='o', alpha=0.5, markersize=5)
    elif USE_RB_CHROMATICITY:
        plt.plot(illum_chroma[i,:1], illum_chroma[i,2:], c=colors[i], marker='o', alpha=0.5, markersize=5)
plt.savefig(f'cluster_result/{CAMERA}_initinal_plot.png')
plt.clf()

# visualize l1_chroma plot
l1_chroma = np.array(l1_chroma)
for i in range(len(l1_chroma)):
    if USE_G1_CHROMATICITY:
        plt.plot(l1_chroma[i,:1],l1_chroma[i,1:], c=l1_colors[i], marker='o', alpha=0.5, markersize=5)
    elif USE_RB_CHROMATICITY:
        plt.plot(l1_chroma[i,:1], l1_chroma[i,2:], c=l1_colors[i], marker='o', alpha=0.5, markersize=5)
plt.savefig(f'cluster_result/{CAMERA}_l1_plot.png')
plt.clf()

# # visualize sub_chroma plot (constrain x axis to 0~3.7)
# sub_chroma = np.array(sub_chroma)
# for i in range(len(sub_chroma)):
#     if USE_G1_CHROMATICITY:
#         plt.plot(sub_chroma[i,:1],sub_chroma[i,1:], c=sub_colors[i], marker='o', alpha=0.5, markersize=5)
#     elif USE_RB_CHROMATICITY:
#         plt.plot(sub_chroma[i,:1], sub_chroma[i,2:], c=sub_colors[i], marker='o', alpha=0.5, markersize=5)
# # plt.xlim(0,3.7)
# # plt.ylim(0,3.2)
# plt.savefig(f'cluster_result/{CAMERA}_sub_plot.png')
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
if USE_G1_CHROMATICITY:
    plt.scatter(illum_chroma[:,:1],illum_chroma[:,1:],c=labels,s=3**2)
    # plot centerpoints of clusters
    plt.plot(cluster_centers[:,:1],cluster_centers[:,1:], 'rx')
elif USE_RB_CHROMATICITY:
    plt.scatter(illum_chroma[:,:1],illum_chroma[:,2:],c=labels,s=3**2)
    # plot centerpoints of clusters
    plt.plot(cluster_centers[:,:1],cluster_centers[:,2:], 'rx')
plt.savefig(f'cluster_result/{CAMERA}_{N_clusters}_cluster_plot.png')
plt.clf()

# # plot distributions of labels
# dist = []
# for i in range(N_clusters):
#     dist.append(len(labels[labels==i]))
# plt.plot(range(N_clusters),dist)
# plt.savefig(f'cluster_result/{CAMERA}_{N_clusters}_dist_labels.png')
# plt.clf()

# plot center point of clusters
if USE_G1_CHROMATICITY:
    plt.plot(cluster_centers[:,:1],cluster_centers[:,1:], 'bo')
elif USE_RB_CHROMATICITY:
    plt.plot(cluster_centers[:,:1],cluster_centers[:,2:], 'bo')
plt.savefig(f'cluster_result/{CAMERA}_{N_clusters}_centers_plot.png')
plt.clf()
# save center coordinates of clusters
np.save(os.path.join(CLUSTER_SAVE_DIR, f'{CAMERA}_{N_clusters}_cluster.npy'), cluster_centers)

# # insert illuminant cluster data to Json & save Json
# for i,idx in enumerate(index):
#     place,light = idx.split('_')
#     cluster_no = labels[i]
#     meta[place][light+'_cluster'] = int(cluster_no)

# with open(f'{CAMERA}_meta_{N_clusters}_clustered.json', 'w') as out_file:
#     json.dump(meta, out_file, indent=4)

# with open(f'cluster_result/{CAMERA}_{N_clusters}_kmeans', 'wb') as pickle_file:
#     pickle.dump({'kmeans':kmeans}, pickle_file)