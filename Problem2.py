from csv import reader
from math import sqrt

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
    
	return lookup
# Calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0
	for i in range(len(row1)-1):
		distance += (float(row1[i])-float(row2[i]))**2
	return sqrt(distance)

# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    dist = list()
    for train_row in train:
        distance = euclidean_distance(test_row, train_row)
        dist.append(distance)
        distances.append((train_row, distance))
    dist.sort()
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    tie  = 0
    allarray = list()
    for i in range(num_neighbors):
        if(dist.count(dist[i])==num_neighbors):
            tie = 1
        neighbors.append(distances[i][0])
        allarray.append(distances)
    return neighbors, tie , allarray

# Make a prediction with neighbors
def predict_classification(train, test,k):
    nn , tie, allarray =get_neighbors(train,test,k)
    classes = list()
    for row in nn:
        classes.append(row[-1])
    if(tie == 1):
        classes.sort()
        return classes[0]
    
    elements = list()
    counts = list()
    for i in range(len(classes)):
       if(classes[i] in elements):
           continue
       else:
           elements.append(classes[i])
           counts.append(classes.count(classes[i]))
    if(len(counts) == k):
        return nn[0][-1]
    maxcount = counts[0]
    maxelement = elements[0]
    for i in range(len(elements)):
        if(counts[i] > maxcount):
             maxcount = counts[i]
             maxelement = elements[i]
    return maxelement

# Test the kNN on the Iris Flowers dataset

filename1 = 'TrainData.txt'
filename2 = 'TestData.txt'
trainset = load_csv(filename1)
testset = load_csv(filename2)
for i in range(len(trainset[0])-1):
	str_column_to_float(trainset, i)
for i in range(len(testset[0])-1):
	str_column_to_float(testset, i)
# convert class column to integers
ll=str_column_to_int(trainset, len(trainset[0])-1)
l=str_column_to_int(testset, len(testset[0])-1)

# evaluate algorithm
#acc=[]
for k in range(1,10):    
    print("k value :",k)
    newclass = list()
    accuracy = list()
    sumofcor = 0
    for i in range(len(testset)):
      newclass.append(predict_classification(trainset,testset[i],k))
    for i in range(len(testset)):
        print("predicted class ",list(ll)[newclass[i]]," actual class ", list(l)[testset[i][len(testset[i])-1]])
        if(list(ll)[newclass[i]] == list(l)[testset[i][len(testset[i])-1]]):
            accuracy.append(1)
        else:
           accuracy.append(0)
    correct = accuracy.count(1)
    false = accuracy.count(0)
    print("Number of correctly classified instances :",correct, "Total number of instances",len(accuracy))
    print("Accuracy :",correct/len(accuracy))
    #acc.append(correct/len(accuracy))
#print(acc)