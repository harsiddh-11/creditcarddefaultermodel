from datetime import datetime
from Prediction_Raw_Data_Validation.predictionDataValidation import Prediction_Data_validation
from DataTypeValidation_Insertion_Prediction.DataTypeValidationPrediction import dBOperation
from DataTranformation_Prediction.DataTransformationPrediction import dataTransformPredict
from application_logging import logger


class pred_validation:
    def P__init__(self, path):
        self.raw_data = Prediction_Data_validation(path)
        self.dataTransform = dataTransformPredict()
        self.dBOperation = dBOperation()
        self.file_object = open("Prediction_Logs/Prediction_Log.txt", 'a+')
        self.log_writer = logger.App_Logger()

    def prediction_validation(self):

        try:
            self.log_writer.log(self.file_object, 'Start of Validation on files for prediction!!')
            # extracting values from prediction schema
            LengthOfDateStampInFile, LengthOfTimeStampInFile, column_names, noofcolumns = self.raw_data.valuesFromSchema()
            regex = self.raw_data.manualRegexCreation()
            self.raw_data.validationFileNameRaw(regex, LengthOfDateStampInFile, LengthOfTimeStampInFile)
            self.raw_data.validateColumnLength(noofcolumns)
            self.raw_data.validateMissingValuesInWholeColumn()
            self.log_writer.log(self.file_object,"Raw Data Validation Complete!!")
            self.log_writer.log(self.file_object, ("Starting Data Transforamtion!!"))
            self.dataTransform.replaceMissingWithNull()
            self.log_writer.log(self.file_object, "DataTransformation Completed!!!")
            # create database with given name, if present open the connection! Create table with columns given in schema
            self.dBOperation.createTableDb('Prediction', column_names)
            self.log_writer.log(self.file_object,"Table creation Completed!!")
            self.log_writer.log(self.file_object, "Insertion of Data into Table started!!!!")
            #insert csv files in the table
            self.dBOperation.insertIntoTableGoodData('Prediction')
            self.log_writer.log(self.file_object, "Insertion in Table completed!!")
            self.log_writer.log(self.file_object, "Deleting Good Data Folder!!")
            # deleting the good data folder after loading files in table
            self.raw_data.deleteExistingGoodDataTrainingFolder()
            self.log_writer.log(self.file_object, "Good_Data folder deleted!!!")
            self.log_writer.log(self.file_object, "Moving bad files to Archive and deleting Bad_Data folder!!!")
            # move thebad files to archive folder
            self.raw_data.moveBadFilesToArchiveBad()
            self.log_writer.log(self.file_object, "Bad files moved to archive!! Bad folder Deleted!!")
            self.log_writer.log(self.file_object, "Validation Operation completed!!")
            self.log_writer.log(self.file_object, "Extracting csv file from table")
            #export data in table to csvfile
            self.dBOperation.selectingDatafromtableintocsv('Prediction')

        except Exception as e:
            raise e



