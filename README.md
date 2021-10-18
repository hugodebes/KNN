# KNN

KNN algorithm implementation in Python

![image](https://user-images.githubusercontent.com/51094403/136416903-78eb3d71-e532-4c63-b1ba-b2b2bcfa194f.png)

## Understanding the Algorithm

The K-Nearest Neighbours (KNN) algorithms is a data classification methods. It comes from the assumption that an individual will most likely look like its closest individuals. We can summmarize like this : Birds of the same feather flock together".

From a mathematical perspective, we'll try to find, for a given point, the individuals with the minimal distance (eq. his neighbors). The factor K comes into play here as it represents the number of individuals that we'll select. For example, if K= 3, we'll select the 3 nearest neighbors of our individual. Finally, the predicted class will be the class with the most occurence in the neighbors list.

If you want more details, you can look at [this page](https://medium.com/@springboard_ind/knn-machine-learning-algorithm-explained-596d60336076).

## Implementation 

This classification Algorithm was used during a competition in my school to predict unknown data.
To avoid disruption linked to scale of our data, I decided to perform a normalization. The negative side is that we put every variable on equal terms whereas some might be more significant than others.

In order to implement our algorithm, I also needed to evaluate the distance between two points. Several methods exist like the Manhattan distance or the Tchebychev distance but I have selected the Euclidian distance for its performance rate.
Then, I simply assign the most represented class to the data that we are predicting.

```python

def distance_euclidienne(indData,indTest):
  dist_ind=0
  for j in range(len(indTest-1)):
    dist_ind+=sqrt((indTest[j]-indData[j])**2)
  return dist_ind

def knn(indTest,data,k):
  data["distance"] = [distance_euclidienne(data.iloc[i,:],indTest) for i in range(len(data))]
  return data.sort_values(by="distance")[:k]

def prediction_classification(voisins_proches):
  return max(set(voisins_proches["Class"].tolist()),key=voisins_proches["Class"].tolist().count)
```

## Assessing our Model

I created a confusion matrix to understand how successful my methods were on my dataset.
![image](https://user-images.githubusercontent.com/51094403/136416765-f62251c4-be86-40ed-b4b5-dc660a907a1f.png)

The KNN algorithms is a non-parametric method which means that the only unknown parameter is K. In order to find the best value I plotted how efficient was my algorithm depending on the number k to optimize it.
The best option seems to be K=4.

![image](https://user-images.githubusercontent.com/51094403/136416830-20815e99-d07e-4af9-a74a-4b77b4dc8240.png)


# Note
The report is in french but do not hesitate to contact me about this projetct whether in english or french.
