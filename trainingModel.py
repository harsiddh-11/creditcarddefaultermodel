from sklearn.model_selection import train_test_split
from data_ingestion import data_loader
from data_preprocessing import preprocessing
from data_preprocessing import clustering
from best_model_finder import tuner
from file_operations import file_methods
from application_logging import logger

class trainModel:

    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("Training_Logs/ModelTrainingLog.txt", 'a+')

    def trainingModel(self):
        self.log_writer.log(self.file_object, 'Start of Training')
        try:
            data_getter = data_loader.Data_Getter(self.file_object,self.log_writer)
            data = data_getter.get_data()
            """Doing data preprocessing"""

            preprocessor = preprocessing.Preprocessor(self.file_object,self.log_writer)

            # create seperate features and labels
            X,Y = preprocessor.separate_label_feature(data, label_column_name = 'default payment next month')

            # check if missing values are present in the dataset
            is_null_present, cols_with_missing_values = preprocessor.is_null_present(X)

            # if missing values are present in the data set
            if(is_null_present):
                X = preprocessor.impute_missing_values(X, cols_with_missing_values)

            """Applying the clustering approach"""

            kmeans = clustering.KMeansClustering(self.file_object, self.log_writer) # object initialization
            number_of_clusters = kmeans.elbow_plot(X) #using the elbow plot to find the number of optimum clusters

            # Divide the data into clusters
            X = kmeans.create_clusters(X, number_of_clusters)
            # Create a new column in the dataset consisting of the corresponding cluster assignment
            X['Labels'] = Y

            # getting the unique clusters from our datset
            list_of_clusters = X['Cluster'].unique()

            """Parsing all the clusters and looking for the best ML algorithm to fit on individual cluster"""

            for i in list_of_clusters:
                cluster_data = X[X['Cluster']==i] #filter the data for one cluster

                # Prepare the feature and label columns
                cluster_features = cluster_data.drop(['Labels', 'Cluster'], axis =1)
                cluster_label = cluster_data['Labels']

                # splitting the data into training and test set for each cluster one by one
                x_train, x_test, y_train, y_test = train_test_split(cluster_features, cluster_label, test_size = 1/3, random_state = 355)

                # pretending with more data pre-processing steps
                train_x = preprocessor.scale_numerical_columns(x_train)
                test_x = preprocessor.scale_numerical_columns(x_test)

                model_finder = tuner.Model_Finder(self.file_object, self.log_writer) # object initialization

                #getting the best model for each of the clusters
                best_model_name, best_model = model_finder.get_best_model(train_x, y_train, test_x, y_test)

                # getting the best model to the directory
                file_op = file_methods.file_Operation(self.file_object, self.log_writer)
                save_model = file_op.save_model(best_model, best_model_name + str(i))

            # logging the successful Training
            self.log_writer.log(self.file_object, 'Successful End of Training')
            self.file_object.close()

        except Exception as e:
            # logging the unsuccessful Training
            self.log_writer.log(self.file_object, 'Unsuccessful End of Training')
            self.file_object.close()
            raise Exception