# %% [markdown]
# # Machine Learning Models: Decision Trees
# On this page, we explore Decision Trees and showcase one of their use cases.

# %% [markdown]
# ## Imports

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# %% [markdown]
# We'll be using Kaggle's [Spotify Song Attributes](https://www.kaggle.com/geomack/spotifyclassification/home) dataset. The dataset contains a number of features of songs from 2017 and a binary variable `target` that represents whether the user liked the song (encoded as 1) or not (encoded as 0). See the documentation of all the features [here](https://developer.spotify.com/documentation/web-api/reference/get-audio-features). 
# 
# ## 1. Read and split the dataset
# <hr>
# 
# - We use `read_csv` from the pandas package to read the data.  
# - We use `train_test_split` from sklearn to split the data into separate training and test sets. (Not to be confused with validation sets which will be created later from the training set).  
#   - The `test_size` parameter determines the proportion of the test set to the training set. Generally, a larger training set results in a better model, a larger test set results in a more accurate assessment of the model. We must find a balance between these two.
#   - Note that the dataset is sorted on the target. If we maintain this list sorting our model will simply predict the target based on the song's position in the sorted list, rather than its features. This will not help us make predictions for future unseen data. Therefore, we set the first column as the index so that our model does not learn the sorted order of our data.

# %%
spotify_df = pd.read_csv("data/spotify.csv", index_col=0)

spotify_df.head() # to show a sample from the dataset

# %%
train_df = None
test_df = None

train_df, test_df = train_test_split(
    spotify_df, test_size=0.2, random_state=123
)

# %% [markdown]
# ## 2. Exploratory Data Analysis (EDA)
# In this section, we want to take a closer look at the dataset so that we can make more informed decisions when designing the model later.
# <hr>

# %%
n_train_samples = train_df.shape[0]
n_test_samples = test_df.shape[0]

print(f"Number of training samples: {n_train_samples}")
print(f"Number of test samples: {n_test_samples}")


# %%
spotify_summary = train_df.describe()
spotify_summary

# %% [markdown]
# In the following plots, we explore different features and analyze their relationship with our target. 1 means the user liked the song, 0 means they did not.

# %%
# Histogram for loudness
feat = "loudness"
train_df.groupby("target")[feat].plot.hist(bins=50, alpha=0.5, legend=True, density = True, title = "Histogram of " + feat)
plt.xlabel(feat)

# %%
for feat in ['acousticness', 'danceability', 'tempo', 'energy', 'valence']: # This loop creates a histogram for each of the features in the list
    train_df.groupby("target")[feat].plot.hist(bins=50, alpha=0.5, legend=True, density = True, title = "Histogram of " + feat)
    plt.xlabel(feat)
    plt.show()

# %% [markdown]
# Keep in mind that even if we see a feature with a histogram that has not discernable patterns with the target, it does not necessarily mean that the feature is not useful for predicting the target. As some patterns only appear when a feature is combined with another. For example: Valence on its own seems insignificant for predicting the target, but that can change when we look at Valence alongside Tempo.

# %% [markdown]
# Note that the dataset includes two text features labeled `song_title` and `artist`. For now, we will simply drop these text features as encoding text can be tricky and may derail us from our original goal here, which is to explore decision trees.

# %% [markdown]
# ## 3. Select features
# <hr>
# 
# In this section, we select the features we want our model to learn. In our case, we will take all the available features except for `song_title` and `artist`. Note that we also need to split our x and y (features and target respectively).

# %%
X_train = train_df.drop(columns=['target', 'song_title', 'artist'])
y_train = train_df['target']
X_test = test_df.drop(columns=['target', 'song_title', 'artist'])
y_test = test_df['target']

# %% [markdown]
# ## 4. Create and assess the baseline
# <hr>
# 
# In this section, we create a very simple baseline model which we will use to measure our decision tree model against. In our case, the `DummyClassifier` will simply predict the most frequent case. Meaning if most songs in our dataset were liked, it will predict that they were all liked.  
# We then use cross_val_score to assess our baseline model.

# %%
dum = DummyClassifier(random_state=123, strategy='most_frequent')
dummy_score = np.mean(cross_val_score(dum, X_train, y_train, cv=10))
dummy_score

# %% [markdown]
# ## 5. Create and assess the Decision Tree model
# <hr>
# 
# In this section, we finally create the decision tree model, and we assess it using `cross_validate`. Note that this function fits the model to the dataset as its first step so we don't need to fit our model beforehand.

# %%
spotify_tree = DecisionTreeClassifier(random_state=123)

# %%
dt_scores_df = pd.DataFrame(cross_validate(spotify_tree, X_train, y_train, cv=10, return_train_score=True))
dt_scores_df


# %% [markdown]
# The main number we want to look at here is `test_score`. We ran 10 different tests on our model, let's take their mean value and compare it to our baseline.

# %%
round(dt_scores_df['test_score'].mean(), 3)

# %% [markdown]
# ## 6. (Optional) Visualize the model
# <hr>
# 
# In this section, we use the `tree` package to visualize our decision tree model to understand it better

# %%
spotify_tree.fit(X_train, y_train) # We must fit (train) the model before we visualize it

feature_names = X_train.columns.tolist() # feature names 
class_names = ["Liked", "Disliked"] # unique class names 

toy_tree_viz = tree.plot_tree(spotify_tree, feature_names=feature_names, class_names=class_names, max_depth=1)
# The tree is too big and complicated to fully visualize, so we set max_depth=2 to visualize the first layers only

# %% [markdown]
# ## 6. Hyperparameter optimization
# <hr>
# 
# So far, we have used the decision tree model in its default configuration and got some decent results. But how can we make it perform better? We need to optimize its hyperparameters. In our case, the decision tree model has a single hyperparameter `depth` which determines the depths of the decision tree.  
# Let's try out a number of different depths and see which one preforms best.

# %%
depths = np.arange(1, 25, 2)
depths

# %%
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

# %% [markdown]
# We can see that in our case, depth 5 yields the best result: `0.711713`. However, we must also consider the **fundamental tradeoff**. We want our model to have the highest test scores, but if its training score is too high it may suggest that it is overfitting on our particular dataset and will generalize poorly to future unseen data. To take a closer look at this, let's plot our model's scores and see how they change as depth changes.

# %%
results_df[["mean_train_score", "mean_cv_score"]].plot()

# %% [markdown]
# <!-- END QUESTION -->
# 
# <br><br>

# %% [markdown]
# We can see that the `mean_cv_score` peaks at depth 5 then begins to decrease. Whereas the `mean_train_score` continuously increases. We can conclude that depth 5 is the ideal depth for our model in this use case. This is what we call "The sweet spot".

# %% [markdown]
# ## 7. Final model and test
# <hr>
# 
# In this section, we recreate our decision tree model using the optimized hyperparameter, then we test it and compare our results with out unoptimized and baseline models.

# %%
best_model = DecisionTreeClassifier(max_depth=5, random_state=123)
best_model.fit(X_test, y_test)
test_score = best_model.score(X_test, y_test)
test_score

# %% [markdown]
# To recap:
# - Baseline model score: ~0.51  
# - Unoptimized decision tree model score: ~0.67  
# - **Optimized decision tree model score**: ~0.83  


