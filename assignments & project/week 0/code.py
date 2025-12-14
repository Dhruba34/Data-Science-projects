import time # to time the execution
import numpy as np
import matplotlib.pyplot as plt
#from decimal import Decimal

def load_data(data_path):
    arr=[]
    file=open(data_path,'r')
    while(True):
        line=file.readline()
        if not line:
            break
        arr.append([float(i) for i in line.split(',')])
    arr=np.array(arr)
    return arr

def initialise_centers(data, K, init_centers=None):
    cent=np.sort(np.random.choice(data.shape[0],size=K,replace=False))
    if not init_centers:
        return np.array([data[i] for i in cent])
    else:
        return init_centers

def initialise_labels(data):
    return np.arange(len(data))

def calculate_distances(data, centers):
    arr=np.array([np.array([np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) for p2 in centers]) for p1 in data])
    return arr
def update_labels(distances):
    arr=np.array([np.argmin(i) for i in distances])
    return arr
def update_centers(data, labels, K):
    centers=np.zeros((K,2))
    counts=np.zeros((K,1))
    np.add.at(centers,labels,data)
    np.add.at(counts,labels,1)
    centers/=counts
    return centers
def check_termination(labels1, labels2):
    return np.allclose(labels1,labels2)
def kmeans(data_path:str, K:int, init_centers):
    '''
    Input :
        data (type str): path to the file containing the data
        K (type int): number of clusters
        init_centers (type numpy.ndarray): initial centers. shape = (K, 2) or None
    Output :
        centers (type numpy.ndarray): final centers. shape = (K, 2)
        labels (type numpy.ndarray): label of each data point. shape = (N,)
        time (type float): time taken by the algorithm to converge in seconds
    N is the number of data points each of shape (2,)
    '''
    data = load_data(data_path)    
    centers = initialise_centers(data, K, init_centers)
    labels = initialise_labels(data)

    start_time = time.time() # Time stamp 

    while True:
        distances = calculate_distances(data, centers)
        labels_new = update_labels(distances)
        centers = update_centers(data, labels_new, K)
        if check_termination(labels, labels_new): break
        else: labels = labels_new
 
    end_time = time.time() # Time stamp after the algorithm ends
    return centers, labels, end_time - start_time
def visualise(data_path, labels, centers):
    arr=load_data(data_path)
    plt.figure()
    plt.scatter(arr[:,0],arr[:,1],c=labels,cmap='plasma',edgecolor='black')
    plt.scatter(centers[:,0],centers[:,1],c='#FF0000',marker='X')
    plt.title('K-means clustering')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig('kmeans.png')
    plt.show()
    return plt

data_path = 'D:\\wids project\\Data-Science-projects\\assignments & project\\week 0\\spice_locations.txt'
K, init_centers = 2, None
centers, labels, time_taken = kmeans(data_path, K, init_centers)
print('Time taken for the algorithm to converge:', time_taken)
visualise(data_path, labels, centers)
