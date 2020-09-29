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
      pd.DataFrame( np.sort( truth, axis = 0 ), columns = [ 'True X', 'True Y' ] ) ],
    axis = 1,
).set_axis(
    [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 ] 
).rename_axis( 
    'Centroid' 
).style.set_caption(
    'K-Means Computed Centroids vs. Ground Truth Centroids'
)
```

<style  type="text/css" >
</style><table id="T_9f9747fe_0224_11eb_ac0b_18568081fac3" ><caption>K-Means Computed Centroids vs. Ground Truth Centroids</caption><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >K-Means X</th>        <th class="col_heading level0 col1" >K-Means Y</th>        <th class="col_heading level0 col2" >True X</th>        <th class="col_heading level0 col3" >True Y</th>    </tr>    <tr>        <th class="index_name level0" >Centroid</th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_9f9747fe_0224_11eb_ac0b_18568081fac3level0_row0" class="row_heading level0 row0" >1</th>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row0_col0" class="data row0 col0" >1.393952</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row0_col1" class="data row0 col1" >1.576855</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row0_col2" class="data row0 col2" >1.394930</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row0_col3" class="data row0 col3" >1.578730</td>
            </tr>
            <tr>
                        <th id="T_9f9747fe_0224_11eb_ac0b_18568081fac3level0_row1" class="row_heading level0 row1" >2</th>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row1_col0" class="data row1 col0" >1.678561</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row1_col1" class="data row1 col1" >1.615219</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row1_col2" class="data row1 col2" >1.692740</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row1_col3" class="data row1 col3" >1.653190</td>
            </tr>
            <tr>
                        <th id="T_9f9747fe_0224_11eb_ac0b_18568081fac3level0_row2" class="row_heading level0 row2" >3</th>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row2_col0" class="data row2 col0" >2.446549</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row2_col1" class="data row2 col1" >1.756104</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row2_col2" class="data row2 col2" >2.410710</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row2_col3" class="data row2 col3" >1.748000</td>
            </tr>
            <tr>
                        <th id="T_9f9747fe_0224_11eb_ac0b_18568081fac3level0_row3" class="row_heading level0 row3" >4</th>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row3_col0" class="data row3 col0" >3.206026</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row3_col1" class="data row3 col1" >3.211233</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row3_col2" class="data row3 col2" >3.218010</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row3_col3" class="data row3 col3" >3.183820</td>
            </tr>
            <tr>
                        <th id="T_9f9747fe_0224_11eb_ac0b_18568081fac3level0_row4" class="row_heading level0 row4" >5</th>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row4_col0" class="data row4 col0" >3.372648</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row4_col1" class="data row4 col1" >3.478127</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row4_col2" class="data row4 col2" >3.385860</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row4_col3" class="data row4 col3" >3.485740</td>
            </tr>
            <tr>
                        <th id="T_9f9747fe_0224_11eb_ac0b_18568081fac3level0_row5" class="row_heading level0 row5" >6</th>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row5_col0" class="data row5 col0" >3.988700</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row5_col1" class="data row5 col1" >3.994159</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row5_col2" class="data row5 col2" >3.989340</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row5_col3" class="data row5 col3" >3.976710</td>
            </tr>
            <tr>
                        <th id="T_9f9747fe_0224_11eb_ac0b_18568081fac3level0_row6" class="row_heading level0 row6" >7</th>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row6_col0" class="data row6 col0" >4.177997</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row6_col1" class="data row6 col1" >4.049241</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row6_col2" class="data row6 col2" >4.163830</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row6_col3" class="data row6 col3" >4.041420</td>
            </tr>
            <tr>
                        <th id="T_9f9747fe_0224_11eb_ac0b_18568081fac3level0_row7" class="row_heading level0 row7" >8</th>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row7_col0" class="data row7 col0" >5.078183</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row7_col1" class="data row7 col1" >5.462597</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row7_col2" class="data row7 col2" >5.087850</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row7_col3" class="data row7 col3" >5.460590</td>
            </tr>
            <tr>
                        <th id="T_9f9747fe_0224_11eb_ac0b_18568081fac3level0_row8" class="row_heading level0 row8" >9</th>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row8_col0" class="data row8 col0" >6.065750</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row8_col1" class="data row8 col1" >5.581439</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row8_col2" class="data row8 col2" >6.043280</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row8_col3" class="data row8 col3" >5.573520</td>
            </tr>
            <tr>
                        <th id="T_9f9747fe_0224_11eb_ac0b_18568081fac3level0_row9" class="row_heading level0 row9" >10</th>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row9_col0" class="data row9 col0" >6.179267</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row9_col1" class="data row9 col1" >5.621234</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row9_col2" class="data row9 col2" >6.192590</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row9_col3" class="data row9 col3" >5.635370</td>
            </tr>
            <tr>
                        <th id="T_9f9747fe_0224_11eb_ac0b_18568081fac3level0_row10" class="row_heading level0 row10" >11</th>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row10_col0" class="data row10 col0" >6.709291</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row10_col1" class="data row10 col1" >5.744552</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row10_col2" class="data row10 col2" >6.743650</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row10_col3" class="data row10 col3" >5.743790</td>
            </tr>
            <tr>
                        <th id="T_9f9747fe_0224_11eb_ac0b_18568081fac3level0_row11" class="row_heading level0 row11" >12</th>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row11_col0" class="data row11 col0" >8.016168</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row11_col1" class="data row11 col1" >7.311453</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row11_col2" class="data row11 col2" >8.019080</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row11_col3" class="data row11 col3" >7.320340</td>
            </tr>
            <tr>
                        <th id="T_9f9747fe_0224_11eb_ac0b_18568081fac3level0_row12" class="row_heading level0 row12" >13</th>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row12_col0" class="data row12 col0" >8.234213</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row12_col1" class="data row12 col1" >7.870020</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row12_col2" class="data row12 col2" >8.227710</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row12_col3" class="data row12 col3" >7.862040</td>
            </tr>
            <tr>
                        <th id="T_9f9747fe_0224_11eb_ac0b_18568081fac3level0_row13" class="row_heading level0 row13" >14</th>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row13_col0" class="data row13 col0" >8.520585</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row13_col1" class="data row13 col1" >8.476420</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row13_col2" class="data row13 col2" >8.509930</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row13_col3" class="data row13 col3" >8.444240</td>
            </tr>
            <tr>
                        <th id="T_9f9747fe_0224_11eb_ac0b_18568081fac3level0_row14" class="row_heading level0 row14" >15</th>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row14_col0" class="data row14 col0" >8.589480</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row14_col1" class="data row14 col1" >8.627657</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row14_col2" class="data row14 col2" >8.608580</td>
                        <td id="T_9f9747fe_0224_11eb_ac0b_18568081fac3row14_col3" class="data row14 col3" >8.604640</td>
            </tr>
    </tbody></table>
