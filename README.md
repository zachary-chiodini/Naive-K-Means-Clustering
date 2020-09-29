# Naive K-Means Clustering

<p align="center">
    <img src="photos/clusters.png">
</p>

<p align="center">
    <img src="photos/clusters1.png">
</p>

<p align="center">
    <img src="photos/clusters100.png">
</p>

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
truth = pd.read_csv( 'centroids.csv', header = None ).to_numpy() / 100000
```


```python
pd.concat( 
    [ pd.DataFrame( np.sort( kmeans.C, axis = 0 ), columns = [ 'K-Means X', 'K-Means Y' ] ),
      pd.DataFrame( np.sort( truth, axis = 0 ), columns = [ 'Truth X', 'Truth Y' ] ) ],
    axis = 1,
).set_axis(
    [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 ] 
).rename_axis( 'Cluster' )
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>K-Means X</th>
      <th>K-Means Y</th>
      <th>Truth X</th>
      <th>Truth Y</th>
    </tr>
    <tr>
      <th>Cluster</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1.393952</td>
      <td>1.576855</td>
      <td>1.39493</td>
      <td>1.57873</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.678561</td>
      <td>1.615219</td>
      <td>1.69274</td>
      <td>1.65319</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.446549</td>
      <td>1.756104</td>
      <td>2.41071</td>
      <td>1.74800</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.206026</td>
      <td>3.211233</td>
      <td>3.21801</td>
      <td>3.18382</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3.372648</td>
      <td>3.478127</td>
      <td>3.38586</td>
      <td>3.48574</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3.988700</td>
      <td>3.994159</td>
      <td>3.98934</td>
      <td>3.97671</td>
    </tr>
    <tr>
      <th>7</th>
      <td>4.177997</td>
      <td>4.049241</td>
      <td>4.16383</td>
      <td>4.04142</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5.078183</td>
      <td>5.462597</td>
      <td>5.08785</td>
      <td>5.46059</td>
    </tr>
    <tr>
      <th>9</th>
      <td>6.065750</td>
      <td>5.581439</td>
      <td>6.04328</td>
      <td>5.57352</td>
    </tr>
    <tr>
      <th>10</th>
      <td>6.179267</td>
      <td>5.621234</td>
      <td>6.19259</td>
      <td>5.63537</td>
    </tr>
    <tr>
      <th>11</th>
      <td>6.709291</td>
      <td>5.744552</td>
      <td>6.74365</td>
      <td>5.74379</td>
    </tr>
    <tr>
      <th>12</th>
      <td>8.016168</td>
      <td>7.311453</td>
      <td>8.01908</td>
      <td>7.32034</td>
    </tr>
    <tr>
      <th>13</th>
      <td>8.234213</td>
      <td>7.870020</td>
      <td>8.22771</td>
      <td>7.86204</td>
    </tr>
    <tr>
      <th>14</th>
      <td>8.520585</td>
      <td>8.476420</td>
      <td>8.50993</td>
      <td>8.44424</td>
    </tr>
    <tr>
      <th>15</th>
      <td>8.589480</td>
      <td>8.627657</td>
      <td>8.60858</td>
      <td>8.60464</td>
    </tr>
  </tbody>
</table>
</div>
