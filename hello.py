from sklearn import tree
features  =  [[140, 0],[130, 0],[150, 1],[170, 1]] #feeatures like weight of fruit in gms and their stucture value
labels = [0,0,1,1]       #0 means apple and 1 means orange

clf = tree.DecisionTreeClassifier()     #this functions are used to get train our model
clf = clf.fit(features, labels)

print(clf.predict([[150,0]]))

