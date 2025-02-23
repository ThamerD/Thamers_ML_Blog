# Machine Learning Models: Decision Trees
On this page, we explore Decision Trees and showcase one of their use cases.

## Imports


```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
```

We'll be using Kaggle's [Spotify Song Attributes](https://www.kaggle.com/geomack/spotifyclassification/home) dataset. The dataset contains a number of features of songs from 2017 and a binary variable `target` that represents whether the user liked the song (encoded as 1) or not (encoded as 0). See the documentation of all the features [here](https://developer.spotify.com/documentation/web-api/reference/get-audio-features). 

## 1. Read and split the dataset
<hr>

- We use `read_csv` from the pandas package to read the data.  
- We use `train_test_split` from sklearn to split the data into separate training and test sets. (Not to be confused with validation sets which will be created later from the training set).  
  - The `test_size` parameter determines the proportion of the test set to the training set. Generally, a larger training set results in a better model, a larger test set results in a more accurate assessment of the model. We must find a balance between these two.
  - Note that the dataset is sorted on the target. If we maintain this list sorting our model will simply predict the target based on the song's position in the sorted list, rather than its features. This will not help us make predictions for future unseen data. Therefore, we set the first column as the index so that our model does not learn the sorted order of our data.


```python
spotify_df = pd.read_csv("data/spotify.csv", index_col=0)

spotify_df.head() # to show a sample from the dataset
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
      <th>acousticness</th>
      <th>danceability</th>
      <th>duration_ms</th>
      <th>energy</th>
      <th>instrumentalness</th>
      <th>key</th>
      <th>liveness</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>valence</th>
      <th>target</th>
      <th>song_title</th>
      <th>artist</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0102</td>
      <td>0.833</td>
      <td>204600</td>
      <td>0.434</td>
      <td>0.021900</td>
      <td>2</td>
      <td>0.1650</td>
      <td>-8.795</td>
      <td>1</td>
      <td>0.4310</td>
      <td>150.062</td>
      <td>4.0</td>
      <td>0.286</td>
      <td>1</td>
      <td>Mask Off</td>
      <td>Future</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.1990</td>
      <td>0.743</td>
      <td>326933</td>
      <td>0.359</td>
      <td>0.006110</td>
      <td>1</td>
      <td>0.1370</td>
      <td>-10.401</td>
      <td>1</td>
      <td>0.0794</td>
      <td>160.083</td>
      <td>4.0</td>
      <td>0.588</td>
      <td>1</td>
      <td>Redbone</td>
      <td>Childish Gambino</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0344</td>
      <td>0.838</td>
      <td>185707</td>
      <td>0.412</td>
      <td>0.000234</td>
      <td>2</td>
      <td>0.1590</td>
      <td>-7.148</td>
      <td>1</td>
      <td>0.2890</td>
      <td>75.044</td>
      <td>4.0</td>
      <td>0.173</td>
      <td>1</td>
      <td>Xanny Family</td>
      <td>Future</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.6040</td>
      <td>0.494</td>
      <td>199413</td>
      <td>0.338</td>
      <td>0.510000</td>
      <td>5</td>
      <td>0.0922</td>
      <td>-15.236</td>
      <td>1</td>
      <td>0.0261</td>
      <td>86.468</td>
      <td>4.0</td>
      <td>0.230</td>
      <td>1</td>
      <td>Master Of None</td>
      <td>Beach House</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.1800</td>
      <td>0.678</td>
      <td>392893</td>
      <td>0.561</td>
      <td>0.512000</td>
      <td>5</td>
      <td>0.4390</td>
      <td>-11.648</td>
      <td>0</td>
      <td>0.0694</td>
      <td>174.004</td>
      <td>4.0</td>
      <td>0.904</td>
      <td>1</td>
      <td>Parallel Lines</td>
      <td>Junior Boys</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_df = None
test_df = None

train_df, test_df = train_test_split(
    spotify_df, test_size=0.2, random_state=123
)
```

## 2. Exploratory Data Analysis (EDA)
In this section, we want to take a closer look at the dataset so that we can make more informed decisions when designing the model later.
<hr>


```python
n_train_samples = train_df.shape[0]
n_test_samples = test_df.shape[0]

print(f"Number of training samples: {n_train_samples}")
print(f"Number of test samples: {n_test_samples}")

```

    Number of training samples: 1613
    Number of test samples: 404
    


```python
spotify_summary = train_df.describe()
spotify_summary
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
      <th>acousticness</th>
      <th>danceability</th>
      <th>duration_ms</th>
      <th>energy</th>
      <th>instrumentalness</th>
      <th>key</th>
      <th>liveness</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>valence</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1613.000000</td>
      <td>1613.000000</td>
      <td>1613.000000</td>
      <td>1613.000000</td>
      <td>1613.000000</td>
      <td>1613.000000</td>
      <td>1613.000000</td>
      <td>1613.000000</td>
      <td>1613.000000</td>
      <td>1613.000000</td>
      <td>1613.000000</td>
      <td>1613.000000</td>
      <td>1613.000000</td>
      <td>1613.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.185627</td>
      <td>0.616745</td>
      <td>247114.827650</td>
      <td>0.681296</td>
      <td>0.136862</td>
      <td>5.383137</td>
      <td>0.189189</td>
      <td>-7.112929</td>
      <td>0.621203</td>
      <td>0.091277</td>
      <td>121.979777</td>
      <td>3.964662</td>
      <td>0.497587</td>
      <td>0.507750</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.259324</td>
      <td>0.163225</td>
      <td>81177.300308</td>
      <td>0.211612</td>
      <td>0.277744</td>
      <td>3.620422</td>
      <td>0.153170</td>
      <td>3.838867</td>
      <td>0.485238</td>
      <td>0.087890</td>
      <td>26.965641</td>
      <td>0.255201</td>
      <td>0.247378</td>
      <td>0.500095</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000005</td>
      <td>0.122000</td>
      <td>16042.000000</td>
      <td>0.014800</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.018800</td>
      <td>-33.097000</td>
      <td>0.000000</td>
      <td>0.023100</td>
      <td>47.859000</td>
      <td>1.000000</td>
      <td>0.035900</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.009190</td>
      <td>0.511000</td>
      <td>200105.000000</td>
      <td>0.564000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.092300</td>
      <td>-8.388000</td>
      <td>0.000000</td>
      <td>0.037300</td>
      <td>100.518000</td>
      <td>4.000000</td>
      <td>0.295000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.062500</td>
      <td>0.629000</td>
      <td>230200.000000</td>
      <td>0.714000</td>
      <td>0.000071</td>
      <td>6.000000</td>
      <td>0.127000</td>
      <td>-6.248000</td>
      <td>1.000000</td>
      <td>0.054900</td>
      <td>121.990000</td>
      <td>4.000000</td>
      <td>0.496000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.251000</td>
      <td>0.738000</td>
      <td>272533.000000</td>
      <td>0.844000</td>
      <td>0.057300</td>
      <td>9.000000</td>
      <td>0.243000</td>
      <td>-4.791000</td>
      <td>1.000000</td>
      <td>0.107000</td>
      <td>137.932000</td>
      <td>4.000000</td>
      <td>0.690000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.995000</td>
      <td>0.984000</td>
      <td>849960.000000</td>
      <td>0.997000</td>
      <td>0.976000</td>
      <td>11.000000</td>
      <td>0.969000</td>
      <td>-0.307000</td>
      <td>1.000000</td>
      <td>0.816000</td>
      <td>219.331000</td>
      <td>5.000000</td>
      <td>0.992000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



In the following plots, we explore different features and analyze their relationship with our target. 1 means the user liked the song, 0 means they did not.


```python
# Histogram for loudness
feat = "loudness"
train_df.groupby("target")[feat].plot.hist(bins=50, alpha=0.5, legend=True, density = True, title = "Histogram of " + feat)
plt.xlabel(feat)
```




    Text(0.5, 0, 'loudness')




    
![png](decision_trees_files/decision_trees_10_1.png)
    



```python
for feat in ['acousticness', 'danceability', 'tempo', 'energy', 'valence']: # This loop creates a histogram for each of the features in the list
    train_df.groupby("target")[feat].plot.hist(bins=50, alpha=0.5, legend=True, density = True, title = "Histogram of " + feat)
    plt.xlabel(feat)
    plt.show()
```


    
![png](decision_trees_files/decision_trees_11_0.png)
    



    
![png](decision_trees_files/decision_trees_11_1.png)
    



    
![png](decision_trees_files/decision_trees_11_2.png)
    



    
![png](decision_trees_files/decision_trees_11_3.png)
    



    
![png](decision_trees_files/decision_trees_11_4.png)
    


Keep in mind that even if we see a feature with a histogram that has not discernable patterns with the target, it does not necessarily mean that the feature is not useful for predicting the target. As some patterns only appear when a feature is combined with another. For example: Valence on its own seems insignificant for predicting the target, but that can change when we look at Valence alongside Tempo.

Note that the dataset includes two text features labeled `song_title` and `artist`. For now, we will simply drop these text features as encoding text can be tricky and may derail us from our original goal here, which is to explore decision trees.

## 3. Select features
<hr>

In this section, we select the features we want our model to learn. In our case, we will take all the available features except for `song_title` and `artist`. Note that we also need to split our x and y (features and target respectively).


```python
X_train = train_df.drop(columns=['target', 'song_title', 'artist'])
y_train = train_df['target']
X_test = test_df.drop(columns=['target', 'song_title', 'artist'])
y_test = test_df['target']
```

## 4. Create and assess the baseline
<hr>

In this section, we create a very simple baseline model which we will use to measure our decision tree model against. In our case, the `DummyClassifier` will simply predict the most frequent case. Meaning if most songs in our dataset were liked, it will predict that they were all liked.  
We then use cross_val_score to assess our baseline model.


```python
dum = DummyClassifier(random_state=123, strategy='most_frequent')
dummy_score = np.mean(cross_val_score(dum, X_train, y_train, cv=10))
dummy_score
```




    np.float64(0.5077524729698643)



## 5. Create and assess the Decision Tree model
<hr>

In this section, we finally create the decision tree model, and we assess it using `cross_validate`. Note that this function fits the model to the dataset as its first step so we don't need to fit our model beforehand.


```python
spotify_tree = DecisionTreeClassifier(random_state=123)
```


```python
dt_scores_df = pd.DataFrame(cross_validate(spotify_tree, X_train, y_train, cv=10, return_train_score=True))
dt_scores_df

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
      <th>fit_time</th>
      <th>score_time</th>
      <th>test_score</th>
      <th>train_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.016004</td>
      <td>0.002000</td>
      <td>0.697531</td>
      <td>0.999311</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.014143</td>
      <td>0.002000</td>
      <td>0.660494</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.009999</td>
      <td>0.001000</td>
      <td>0.685185</td>
      <td>0.999311</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.011000</td>
      <td>0.001158</td>
      <td>0.639752</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.011091</td>
      <td>0.002000</td>
      <td>0.639752</td>
      <td>0.999311</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.010095</td>
      <td>0.002001</td>
      <td>0.658385</td>
      <td>0.999311</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.010088</td>
      <td>0.001002</td>
      <td>0.639752</td>
      <td>0.999311</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.013002</td>
      <td>0.001001</td>
      <td>0.608696</td>
      <td>0.999311</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.011600</td>
      <td>0.002000</td>
      <td>0.701863</td>
      <td>0.999311</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.012999</td>
      <td>0.001001</td>
      <td>0.695652</td>
      <td>0.999311</td>
    </tr>
  </tbody>
</table>
</div>



The main number we want to look at here is `test_score`. We ran 10 different tests on our model, let's take their mean value and compare it to our baseline.


```python
round(dt_scores_df['test_score'].mean(), 3)
```




    np.float64(0.663)



## 6. (Optional) Visualize the model
<hr>

In this section, we use the `tree` package to visualize our decision tree model to understand it better


```python
spotify_tree.fit(X_train, y_train) # We must fit (train) the model before we visualize it

feature_names = X_train.columns.tolist() # feature names 
class_names = ["Liked", "Disliked"] # unique class names 

toy_tree_viz = tree.plot_tree(spotify_tree, feature_names=feature_names, class_names=class_names, max_depth=1)
# The tree is too big and complicated to fully visualize, so we set max_depth=2 to visualize the first layers only
```


    
![png](decision_trees_files/decision_trees_24_0.png)
    


## 6. Hyperparameter optimization
<hr>

So far, we have used the decision tree model in its default configuration and got some decent results. But how can we make it perform better? We need to optimize its hyperparameters. In our case, the decision tree model has a single hyperparameter `depth` which determines the depths of the decision tree.  
Let's try out a number of different depths and see which one preforms best.


```python
depths = np.arange(1, 25, 2)
depths
```




    array([ 1,  3,  5,  7,  9, 11, 13, 15, 17, 19, 21, 23])




```python
results_dict = {
    "depth": [],
    "mean_train_score": [],
    "mean_cv_score": [],
}

for depth in depths: # Create a model for each depth in our list, assess it, and add it to our results_df
    model = DecisionTreeClassifier(max_depth=depth, random_state=123)
    scores = cross_validate(model, X_train, y_train, cv=10, return_train_score=True)
    results_dict["depth"].append(depth)
    results_dict["mean_cv_score"].append(np.mean(scores["test_score"]))
    results_dict["mean_train_score"].append(np.mean(scores["train_score"]))

results_df = pd.DataFrame(results_dict)
results_df = results_df.set_index("depth")
results_df
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
      <th>mean_train_score</th>
      <th>mean_cv_score</th>
    </tr>
    <tr>
      <th>depth</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.651030</td>
      <td>0.646032</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.733485</td>
      <td>0.692524</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.794035</td>
      <td>0.711713</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.858718</td>
      <td>0.703060</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.912930</td>
      <td>0.690610</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.955157</td>
      <td>0.680048</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.980850</td>
      <td>0.674457</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.993525</td>
      <td>0.658979</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.998278</td>
      <td>0.669538</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.999173</td>
      <td>0.665812</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.999449</td>
      <td>0.662706</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.999449</td>
      <td>0.662706</td>
    </tr>
  </tbody>
</table>
</div>



We can see that in our case, depth 5 yields the best result: `0.711713`. However, we must also consider the **fundamental tradeoff**. We want our model to have the highest test scores, but if its training score is too high it may suggest that it is overfitting on our particular dataset and will generalize poorly to future unseen data. To take a closer look at this, let's plot our model's scores and see how they change as depth changes.


```python
results_df[["mean_train_score", "mean_cv_score"]].plot()
```




    <Axes: xlabel='depth'>




    
![png](decision_trees_files/decision_trees_29_1.png)
    


<!-- END QUESTION -->

<br><br>

We can see that the `mean_cv_score` peaks at depth 5 then begins to decrease. Whereas the `mean_train_score` continuously increases. We can conclude that depth 5 is the ideal depth for our model in this use case. This is what we call "The sweet spot".

## 7. Final model and test
<hr>

In this section, we recreate our decision tree model using the optimized hyperparameter, then we test it and compare our results with out unoptimized and baseline models.


```python
best_model = DecisionTreeClassifier(max_depth=5, random_state=123)
best_model.fit(X_test, y_test)
test_score = best_model.score(X_test, y_test)
test_score
```




    0.8267326732673267



To recap:
- Baseline model score: ~0.51  
- Unoptimized decision tree model score: ~0.67  
- **Optimized decision tree model score**: ~0.83  
