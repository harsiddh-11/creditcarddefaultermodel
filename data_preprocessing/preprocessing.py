import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from feature_engine.imputation import CategoricalImputer
from imblearn.over_sampling import RandomOverSampler

class Preprocessor:
    """This class used to clean and tranfor the data"""
    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object

    def remove_unwanted_spaces(self, data, columns):
        """Remove unwanted space from pandas dataframe"""
        self.logger_object.log(self.file_object, 'Entered the remove_unwanted_spaces method of the Preprocessor class')
        self.data = data

        try:
            self.df_without_spaces = self.data.apply(lambda x: x.str.strip() if x.dtype == "object" else x) #drop the labels specififed in columns
            self.logger_object.log(self.file_object,'Unwanted spaces removal Successful.Exited the remove_unwanted_spaces method of the Preprocessor class')
            return self.df_without_spaces

        except Exception as e:
            self.logger_object.log(self.file_object, 'Exception occured in remove_unwanted_spaces method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'unwanted space removal Unsuccessful. Exited the remove_unwanted_spaces method of the Preprocessor class')
            raise Exception()

    def remove_columns(self, data, columns):
        """This method removes the given columns from the dataframe"""
        self.logger_object.log(self.file_object, 'Entered the remove_columns method of the Preprocessor class')
        self.data = data
        self.columns = columns

        try:
            self.useful_data = self.data.drop(labels = self.columns, axis = 1) #drop the labels specified in the columns
            self.logger_objet.log(self.file_object, 'Column removal Successful.Exited the remove_columns method of the Preprocessor class')
            return self.useful_data
        except Exception as e:
            self.logger_object.log(self.file_object, 'Exception occured in remove_columns method of the Preprocessor class. Exception message:  '+str(e))
            self.logger_object.log(self.file_object, 'Column removal Unsuccessful. Exited the remove_columns method of the Preprocessor class')
            raise Exception()

    def separate_label_feature(self, data , label_column_name):
        """This method separates the feature and target column"""
        self.logger_object.log(self.file_object, 'Entered the separate_label_feature method of the Preprocessor class')
        try:
            self.X = data.drop(labels = label_column_name, axis =1) #drop the column specified and separatethe feature column
            self.Y = data[label_column_name]
            self.logger_object.log(self.file_object, 'Label Separation Successful. Exited the separate_label_feature method of the Preprocessor class')
            return self.X, self.Y

        except Exception as e:
            self.logger_object.log(self.file_object, 'Exception occured in separate_label_feature method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'Label Separation Unsuccessful. Exited the separate_label_feature method of the Preprocessor class')
            raise Exception()

    def is_null_present(self, data):
        """method check whether there is null value or not"""
        self.logger_object.log(self.file_object, 'Entered the is_null_present method of the Preprocessor class')
        self.null_present = False
        self.cols_with_missing_values = []
        self.cols = data.columns
        try:
            self.null_counts = data.isna().sum()
            for i in range(len(self.null_counts)):
                if self.null_counts[i] > 0:
                    self.null_present = True
                    self.cols_with_missing_values.append(self.cols[i])
            if(self.null_present):
                self.dataframe_with_null = pd.DataFrame()
                self.dataframe_with_null['columns'] = data.columns
                self.dataframe_with_null['missing values count'] = np.asarray(data.isna().sum())
                self.dataframe_with_null.to_csv('prepreocessing_data/null_values.csv')
            self.logger_object.log(self.file_object, 'Finding missing values is a success.Data written to the null values file. Exited the is_null_present method of the Preprocessor class')
            return self.null_present, self.cols_with_missing_values

        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in is_null_present method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'Finding missing values failed. Exited the is_null_present method of the Preprocessor class')
            raise Exception()

    def impute_missing_values(self, data, cols_with_missing_values):
        """This method replace all the missing values in the dataframe using KNN imputer"""
        self.logger_object.log(self.file_object, 'Entered the impute missing values method of the preprocessor class')
        self.data = data
        self.cols_with_missing_values = cols_with_missing_values
        try:
            self.imputer = CategoricalImputer()
            for col in self.cols_with_missing_values:
                self.data[col] = self.imputer.fit_transform(self.data[col])
            self.logger_object.log(self.file_object, 'Imputing missing values Successful. Exited the impute_missing_values method of the Preprocessor class')
            return self.data

        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in impute_missing_values method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'Imputing missing values failed. Exited the impute_missing_values method of the Preprocessor class')
            raise Exception()

    def scale_numerical_columns(self, data):
        """
        this method scales the numerical values using the standard scaler
        """
        self.logger_object.log(self.file_object, 'Entered the scale_numerical_columns method of the Preprocessor class')
        self.data =data

        try:
            self.num_df = self.data.select_dtypes(include = ['int64']).copy()
            self.scaler = StandardScaler()
            self.scaled_data = self.scaler.fit_transform(self.num_df)
            self.scaled_num_df = pd.DataFrame(data = self.scaled_data, columns = self.num_df.columns)
            self.logger_object.log(self.file_object, 'scaling for numerical values successful. Exited the scale_numerical_columns method of the Preprocessor class')
            return self.scaled_num_df

        except Exception as e:
            self.logger_object.log(self.file_object, 'Exception occured in scale_numerical_columns method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'scaling for numerical columns Failed. Exited the scale_numerical_columns method of the Preprocessor class')
            raise Exception

    def encode_categorical_columns(self,data):
        """encode categorical values to numerical values"""
        self.logger_object.log(self.file_object, 'Entered the encode_categorical_columns method of the Preprocessor class')
        try:
            self.cat_df = data.select_dtypes(include = ['object']).copy()
            for col in self.cat_df.columns:
                self.cat_df = pd.get_dummies(self.cat_df, columns = [col], prefix = [col], drop_first = True)
                self.logger_object.log(self.file_object, 'encoding for categorical values successful. Exited the encode_categorical_columns method of the Preprocessor class')
                return self.cat_df

        except Exception as e:
            self.logger_object.log(self.file_object, 'Exception occured in encode_categorical_columns method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'encoding for categorical columns Failed.Exited the encode_categorical_columns method of the Preprocessor class')
            raise Exception

    def handle_imbalanced_dataset(self, X, Y):
        """This method handles the imbalanced dataset to make it balanced one"""
        self.logger_object.log(self.file_object, 'Entered the handle_imbalanced_dataset method of the preprocessor class')
        try:
            self.rdsmple = RandomOverSampler()
            self.x_sampled, self.y_sampled = self.rdsmple.fit_transform(x,y)
            self.logger_object.log(self.file_object,'dataset balancing successful. Exited the handle_imbalanced_dataset method of the Preprocessor class')
            return self.x_sampled, self.y_sampled
        except Exception as e:
            self.logger_object.log(self.file_object, 'Exception occured in handle_imbalanced_dataset method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'dataset balancing Failed. Exited the handle_imbalanced_dataset method of the Preprocessor class')
            raise Exception()




