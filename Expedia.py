import os
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA
from sklearn.mixture import GMM
from sklearn.neighbors import KNeighborsClassifier
import Exploratory
import operator
import ml_metrics as metrics

## Working directory

os.chdir("~/Documents/My Office Briefcase/My Coworker's Documents/Kaggle/Expedia- Hotel Recommendations")

## Load data

print('Loading data')
destinations = pd.read_csv("destinations.csv")
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")

print('Converting date/time data')
train["date_time"] = pd.to_datetime(train["date_time"])
train["srch_ci"] = pd.to_datetime(train["srch_ci"],format='%Y-%m-%d', errors="coerce")
train["srch_co"] = pd.to_datetime(train["srch_co"],format='%Y-%m-%d', errors="coerce")

print('Processing date/time data')
train["year"] = train["date_time"].dt.year
train["month"] = train["date_time"].dt.month
train["hour"] = train["date_time"].dt.hour

## Test/train subsets
print('Setting up test set/training subset')
unique_users = train.user_id.unique()

t1 = train
t2 = test


##
# Exploratory analysis
# print('Getting all principal components')
# pcafull = PCA(n_components=149)
# dest_lrg = pcafull.fit_transform(destinations[["d{0}".format(i+1) for i in range(149)]])
# fh=Exploratory.cumeigenplot(pcafull)
# fh.savefig('cumeigenplot_destinations.png')
# fh=Exploratory.eigenplot(pcafull)
# fh.savefig('eigenplot_destinations.png')
# del dest_lrg
# del pcafull

### Dimensionality reduction
print('Getting first 5 components')
pca = PCA(n_components = 5)
dest_small = pca.fit_transform(destinations[["d{0}".format(i+1) for i in range(149)]])
# fh=Exploratory.pcaplot(dest_small)
# fh.savefig('pcaplot_destinations.png')

print('Gaussian mixture model on 4 vs 5')
Xgmm = dest_small[:,(3,4)]
dest_small = dest_small[:,(0,1,2)]

g = GMM(n_components=5)
g.fit(Xgmm)
pX = g.predict_proba(Xgmm)
# fh=Exploratory.gmmplot(Xgmm,g)
# fh.savefig('gmmplot_destinations.png')
Il = g.means_[:,1]==min(g.means_[:,1])
p = np.log(pX[:,Il] / (1-pX[:,Il]))

dest_small = np.concatenate((dest_small,p),axis=1)

dest_small = pd.DataFrame(dest_small)
dest_small.rename(columns = {0: "PC1", 1: "PC2", 2: "PC3", 3: "GMM"}, inplace = True)
print('Classifying destination ID\'s based on PC1/PC2/PC3/GMM')
rfc = KNeighborsClassifier(n_neighbors = 5, n_jobs = -1)
print('Testing predicted classification based on PC1/PC2/PC3/GMM')
RFC_destination1 = []
RFC_destination2 = []
RFC_destination3 = []
RFC_destination4 = []
RFC_destination5 = []
for dest in range(dest_small.shape[0]):
    if divmod(dest+1, 100)[1] == 0:
        print('\n', end = '', flush = True)
    if divmod(dest+1, 1000)[1] == 0:
        print(dest+1, 'destinations processed.', flush=True)
    print('.',end = '',flush=True)
    sel_destination = destinations.loc[dest, 'srch_destination_id']
    RFCfit = rfc.fit(dest_small[destinations.srch_destination_id!=sel_destination], destinations.srch_destination_id[destinations.srch_destination_id!=sel_destination])
    Neighbors = RFCfit.kneighbors(np.reshape(dest_small.loc[dest, :].as_matrix(), [1, -1]), return_distance = False)
    Neighbors = Neighbors[0]
    RFC_destination1.append(Neighbors[0])
    RFC_destination2.append(Neighbors[1])
    RFC_destination3.append(Neighbors[2])
    RFC_destination4.append(Neighbors[3])
    RFC_destination5.append(Neighbors[4])
print('Predicting nearby complete')

del dest_small
dest_tiny = pd.DataFrame()
dest_tiny['RFC_destination1'] = RFC_destination1
dest_tiny['RFC_destination2'] = RFC_destination2
dest_tiny['RFC_destination3'] = RFC_destination3
dest_tiny['RFC_destination4'] = RFC_destination4
dest_tiny['RFC_destination5'] = RFC_destination5
dest_tiny["srch_destination_id"] = destinations["srch_destination_id"]
t0 = np.ones((t2.shape[0], 5))*np.nan
print('Adding PC and GMM KNN-derived features to test set')
for it, row in dest_tiny.iterrows():
    if divmod(it+1, 100)[1]==0:
        print('\n', end = '', flush=True)
    if divmod(it+1, 1000)[1]==0:
        print(it+1, 'iterations complete.', flush=True)
    print('.',end='',flush=True)
    I = t2.srch_destination_id == row.srch_destination_id
    t0[I,0] = row.RFC_destination1
    t0[I,1] = row.RFC_destination2
    t0[I,2] = row.RFC_destination3
    t0[I,3] = row.RFC_destination4
    t0[I,4] = row.RFC_destination5
t2['RFC_destination1'] = t0[:, 0]
t2['RFC_destination2'] = t0[:, 1]
t2['RFC_destination3'] = t0[:, 2]
t2['RFC_destination4'] = t0[:, 3]
t2['RFC_destination5'] = t0[:, 4]

# Take advantage of data leak
match_cols = ['user_location_country', 'user_location_region', 'user_location_city', 'hotel_market',
              'orig_destination_distance']

groups = t1.groupby(match_cols)

def generate_exact_matches(row, match_cols):
    index = tuple([row[t] for t in match_cols])
    try:
        group = groups.get_group(index)
    except Exception:
        return []
    clus = list(set(group.hotel_cluster))
    return clus


exact_matches = []
for i in range(t2.shape[0]):
    exact_matches.append(generate_exact_matches(t2.iloc[i], match_cols))

# exact matches contains all the exact matches between training and testing data.

## most popular hotel clusters.
def make_key(items):
    return "_".join([str(i) for i in items])

most_common_clusters = list(train.hotel_cluster.value_counts().head().index)
## Next, search destination ID most popular
print('Matching search destination id with hotel cluster')

match_cols = ["srch_destination_id"]
cluster_cols = match_cols + ['hotel_cluster']
groups = t1.groupby(cluster_cols)

# Click-through rate: each click has some probability of leading to a booking
top_clusters = {}
totalClicks = 0
totalBookings = 0
for name,group in groups:
    totalClicks = totalClicks + len(group.is_booking[group.is_booking == False])
    totalBookings = totalBookings + len(group.is_booking[group.is_booking == True])
CTR = totalBookings/(totalBookings+totalClicks)

for name, group in groups:
    clicks = len(group.is_booking[group.is_booking == False])
    bookings = len(group.is_booking[group.is_booking == True])

    score = bookings + CTR*clicks

    clus_name = make_key(name[:len(match_cols)])
    if clus_name not in top_clusters:
        top_clusters[clus_name] = {}
    top_clusters[clus_name][name[-1]] = score

cluster_dict = {}
for n in top_clusters:
    tc = top_clusters[n]
    top = [l[0] for l in sorted(tc.items(), key=operator.itemgetter(1), reverse=True)[:5]]
    cluster_dict[n] = top

# Here's where we make predictions.
match_cols = ["srch_destination_id"]
near1_col = ["RFC_destination1"]
near2_col = ["RFC_destination2"]
near3_col = ["RFC_destination3"]
near4_col = ["RFC_destination4"]
near5_col = ["RFC_destination5"]

preds = []
near_preds = []
for index, row in t2.iterrows():
    # Direct prediction
    key = make_key([row[m] for m in match_cols])
    if key in cluster_dict:
        preds.append(cluster_dict[key])
    else:
        preds.append([])
    # Nearby (cluster-wise) prediction
    nearp = []
    key = make_key([row[m] for m in near1_col])
    if key in cluster_dict:
        nearp.append(cluster_dict[key][0])
    key = make_key([row[m] for m in near2_col])
    if key in cluster_dict:
        nearp.append(cluster_dict[key][0])
    key = make_key([row[m] for m in near3_col])
    if key in cluster_dict:
        nearp.append(cluster_dict[key][0])
    key = make_key([row[m] for m in near4_col])
    if key in cluster_dict:
        nearp.append(cluster_dict[key][0])
    key = make_key([row[m] for m in near5_col])
    if key in cluster_dict:
        nearp.append(cluster_dict[key][0])

    near_preds.append(nearp)



def f5(seq, idfun=None):
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        if marker in seen: continue
        seen[marker] = 1
        result.append(item)
    return result

print('Predicting', len(preds), 'rows of test set', flush=True)
full_preds = [f5(exact_matches[p] + preds[p] + near_preds[p] + most_common_clusters)[:5] for p in range(len(preds))]
#print('MAP@5:')
#print(metrics.mapk([[l] for l in t2["hotel_cluster"]], full_preds, k=5))

##
print('Exporting predictions')
import csv
with open('Predictions-2016-05-26.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = ['id', 'hotel_cluster'], delimiter=',')
    writer.writeheader()

    for k in range(len(full_preds)):
        s1 = '{0}'.format(test.id[k])
        fp0 = full_preds[k]
        s2 = ''
        for p in range(len(fp0)):
            s2 = s2 + ' {0}'.format(fp0[p])
        writer.writerow({'id': s1, 'hotel_cluster': s2})
