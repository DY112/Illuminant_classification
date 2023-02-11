"""
Clustering illuminant chroma using K-means algorithm
Use euclidian distance as distance metric

Ignored place no (outliers) : 496,497,498,544,709,717
"""
import json,pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

N_clusters = 5
CAMERA = 'galaxy'

with open(f'meta_{CAMERA}.json','r') as json_file:
    meta = json.load(json_file)

index = []
illum_chroma = []
colors = []

# gather illuminant information
for place in meta:
    place_meta = meta[place]
    
    l1 = place_meta['Light1']
    if(l1[0]>10 or l1[2]>3): print(place,l1)
    illum_chroma.append([l1[0],l1[2]])
    color = [item / max(l1[0],1.,l1[2]) for item in [l1[0],1.,l1[2]]]
    colors.append(color)
    index.append(place+'_Light1')

    l2 = place_meta['Light2']
    if(l2[0]>10 or l2[2]>3): print(place,l2)
    illum_chroma.append([l2[0],l2[2]])
    color = [item / max(l2[0],1.,l2[2]) for item in [l2[0],1.,l2[2]]]
    colors.append(color)
    index.append(place+'_Light2')
    
    if place_meta['NumOfLights'] == 3:
        l3 = place_meta['Light3']
        if(l3[0]>10 or l3[2]>3): print(place,l3)
        illum_chroma.append([l3[0],l3[2]])
        color = [item / max(l3[0],1.,l3[2]) for item in [l3[0],1.,l3[2]]]
        colors.append(color)
        index.append(place+'_Light3')

# visualize initial illum chroma plot
illum_chroma = np.array(illum_chroma)
for i in range(len(illum_chroma)):
    plt.plot(illum_chroma[i,:1],illum_chroma[i,1:], c=colors[i], marker='o', alpha=0.5, markersize=5)
plt.savefig(f'cluster_result/{CAMERA}_initinal_plot.png')
plt.clf()

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
plt.savefig(f'cluster_result/{CAMERA}_{N_clusters}_centers_plot.png')
plt.clf()
# save center coordinates of clusters
np.save(f'cluster_result/{CAMERA}_{N_clusters}_cluster.npy',cluster_centers)

# insert illuminant cluster data to Json & save Json
for i,idx in enumerate(index):
    place,light = idx.split('_')
    cluster_no = labels[i]
    meta[place][light+'_cluster'] = int(cluster_no)

with open(f'{CAMERA}_meta_{N_clusters}_clustered.json', 'w') as out_file:
    json.dump(meta, out_file, indent=4)

with open(f'cluster_result/{CAMERA}_{N_clusters}_kmeans', 'wb') as pickle_file:
    pickle.dump({'kmeans':kmeans}, pickle_file)