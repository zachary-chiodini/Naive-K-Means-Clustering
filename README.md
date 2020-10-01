# Naive K-Means Clustering

<p align="justify">
    This is a module for the repeated k-means clustering method for pattern recognition and classification of data written in Python.
</p>
<p align="left">
    <img src="photos/dependencies.png" width="244px">
</p>
<h1>Mathematical Model</h1>

<p align="center">
    <img src="photos/clusters.png">
</p>

<p align="center">
    <img src="photos/clusters1.png">
</p>

<p align="center">
    <img src="photos/clusters100.png">
</p>

<a href="https://www.sciencedirect.com/science/article/pii/S0031320319301608">Good Recourse</a>

<h1>Try It</h1>

```python
from kmeans import KMeans
import pandas as pd, numpy as np
```


```python
df = pd.read_csv( 'clusters.csv' )
```


```python
X = df.to_numpy() / 100000
kmeans = KMeans( X, 15 )
kmeans.classify( repeat = 100 )
```


```python
kmeans.C = np.sort( kmeans.C, axis = 0 ).round( 5 )
truth = pd.read_csv( 'centroids.csv', header = None ).to_numpy() / 100000
truth = np.sort( truth, axis = 0 )
```


```python
table = pd.DataFrame(
    {
        'Centroid' : [ '1', '2', '3', '4', '5', 
                       '6', '7', '8', '9', '10', 
                       '11', '12', '13', '14', '15'],
        'K-Means X' : kmeans.C[ :, 0 ],
        'True X'    : truth[ :, 0 ],
        'K-Means Y' : kmeans.C[ :, 1 ],
        'True Y'    : truth[ :, 1 ]
    }
)
```


```python
from table import render_table
import matplotlib.pyplot as plt

render_table( 
    table, header_columns = 0, col_width = 2.0, font_size = 15,
    title = 'Coordinates of K-Means Computed Centroids vs. Ground Truth Centroids' 
)
plt.show()
```

<p align="center">
    <img src="photos/table.png">
</p>
