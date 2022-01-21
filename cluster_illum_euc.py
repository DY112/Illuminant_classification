"""
Clustering illuminant chroma using K-means algorithm
Use euclidian distance as distance metric

Ignored place no (outliers) : 496,497,498,544,709,717
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

N_clusters = 20

with open('meta.json','r') as json_file:
    meta = json.load(json_file)

index = []
illum_chroma = []

# gather illuminant information
for place in meta:
    place_meta = meta[place]
    
    l1 = place_meta['Light1']
    if(l1[0]>10 or l1[2]>3): print(place,l1)
    illum_chroma.append([l1[0],l1[2]])
    index.append(place+'_Light1')

    l2 = place_meta['Light2']
    if(l2[0]>10 or l2[2]>3): print(place,l2)
    illum_chroma.append([l2[0],l2[2]])
    index.append(place+'_Light2')
    
    if place_meta['NumOfLights'] == 3:
        l3 = place_meta['Light3']
        if(l3[0]>10 or l3[2]>3): print(place,l3)
        illum_chroma.append([l3[0],l3[2]])
        index.append(place+'_Light3')

# visualize initial illum chroma plot
illum_chroma = np.array(illum_chroma)
plt.plot(illum_chroma[:,:1],illum_chroma[:,1:],'ro', alpha=0.5, markersize=3)
plt.savefig('initinal_plot.png')
plt.clf()

# K means clustering & plot
kmeans = KMeans(n_clusters=20).fit(illum_chroma)
cluster_centers = kmeans.cluster_centers_
labels = kmeans.labels_
plt.scatter(illum_chroma[:,:1],illum_chroma[:,1:],c=labels,s=3**2)
plt.plot(cluster_centers[:,:1],cluster_centers[:,1:], 'rx')
plt.savefig('cluster_plot.png')
plt.clf()

# plot center point of clusters
plt.plot(cluster_centers[:,:1],cluster_centers[:,1:], 'bo')
plt.savefig('centers_plot.png')
plt.clf()

# insert illuminant cluster data to Json & save Json
for i,idx in enumerate(index):
    place,light = idx.split('_')
    cluster_no = labels[i]
    meta[place][light+'_cluster'] = int(cluster_no)

with open('meta_clustered.json', 'w') as out_file:
    json.dump(meta, out_file, indent=4)