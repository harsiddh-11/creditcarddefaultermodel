import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
from file_operations import file_methods

class KMeansClustering:
    """
    used to divide the data into clusters before training
    """

    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object

    def elbow_plot(self, data):
        """ Method saves the plot to decide the optium number of clusters to the files"""
        self.logger_object.log(self.file_object, 'Entered the elbow_plot method of the KMeansCLustering class')
        wcss = []
        try:
            for i in range(1,11):
                kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42) #inititalizing the KMeans object
                kmeans.fit(data)
                wcss.append(kmeans.inertia_)
            plt.plot(range(1,11),wcss)
            plt.title('The Elbow Method')
            plt.xlabel('Number of clusters')
            plt.ylabel('WCSS')
            # plt.show()
            plt.savefig('preprocessing_data/K-Means_Elbow.PNG')
            # finding the values of the optimum cluster programmatically
            self.kn = KneeLocator(range(1,11), wcss, curve = 'convex', direction = 'decreasing')
            self.logger_object.log(self.file_object, 'The optimum number of clusters is: '+str(self.kn.knee)+' . Exited the elbow_plot method of the KMeansClustering class')
            return self.kn.knee

        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in elbow_plot method of the KMeansClustering class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'Finding the number of clusters failed. Exited the elbow_plot method of the KMeansClustering class')
            raise Exception()

    def create_clusters(self, data, number_of_clusters):
        """ create datframe consisiting of the cluster information"""
        self.logger_object.log(self.file_object, 'Entered the create_clusters method of the KMeansClustering class')
        self.data = data
        try:
            self.kmeans = KMeans(n_clusters = number_of_clusters, init = 'k-means++', random_state = 42)
            self.y_kmeans = self.kmeans.fit_predict(data) #divide data into clusters

            self.file_op = file_methods.File_Operation(self.file_object, self.logger_object)
            self.save_model = self.file_op.save_model(self.kmeans, 'KMeans') #saving the KMeans model to directory
            self.data['Cluster'] = self.y_kmeans # create a new column in dataset for storing the cluster information
            self.logger_object.log(self.file_object, 'succesfully created '+str(self.kn.knee)+ 'clusters. Exited the create_clusters method of the KMeansClustering class')
            return self.data

        except Exception as e:
            self.logger_object.log(self.file_object, 'Exception occured in create_clusters method of the KMeansClustering class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'Fitting the data to clusters failed. Exited the create_clusters method of the KMeansClustering class')
            raise Exception()

