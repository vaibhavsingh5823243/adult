from log import Log
log_obj=Log("Adult Census Income Prediction").log()

import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
#from pandas_profiling import ProfileReport
from sklearn.naive_bayes import CategoricalNB
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


class IncomePrediction:

    def __init__(self):
        self.data=pd.read_csv('adult.csv')

    def visualization(self):
        """This function will create profile report of data"""
        try:
           # html_page=ProfileReport(self.data)
           # html_page.to_file('templates\profile_report.html')
            return True
        except Exception as e:
            return log_obj.error(e)

    def accuracy(self):

        """This function will return the accuracy
           score,precision value and recall"""

        try:

            x_train, y_train, x_test, y_test=self.split_train_test()
            y_pred=self.predict(x_test)
            matrix=confusion_matrix(y_test,y_pred)
            TP,FP,FN,TN=matrix[0][0],matrix[0][1],matrix[1][0],matrix[1][1]
            accuracy=round((TP+TN)/(TP+FP+FN+TN),4)*100
            precision=round(TP/(TP+FP),4)*100
            recall=round(TP/(TP+FN),4)*100
            f1_score=round(2*(precision*recall)/(precision+recall),2)
            result={'Accuracy':accuracy,'Precision':precision,'Recall':recall, 'F1 Score':f1_score}
            return matrix,result

        except Exception as e:
            return log_obj.error(e)

    def cat_to_num(self,data):

        """This function will convert categorical feature to
         numeric feature.Like male:1 and female:0,or <=50K:0
         and >50K:1"""

        try:
            if data==' Male' or data==' <=50K':
                return 0
            elif data==' Female' or data==' >50K':
                return 1
        except Exception as e:
            return log_obj.error(e)

    def select_feature_nd_target(self):

        """After analyzing the dataset i found that categorical variable
        does't contibute in model building so i decided to drop it.And
        create a model only on numeric features"""

        try:
            newdata=self.data
            target=newdata['salary'].apply(self.cat_to_num)
            newdata['sex']=newdata['sex'].apply(self.cat_to_num)
            features=newdata.select_dtypes(include='int64')
            features=features.drop(columns=['education-num'])
            return features,target
        except Exception as e:
            return log_obj.error(e)

    def split_train_test(self,train_size=.70,test_size=.3):

        """
        This function will takes three paremeters and split
        the dataset into train and test set.
        :param train_size:
        :param test_size:
        :return:
        """
        try:
            features,target=self.select_feature_nd_target()
            x_train,x_test,y_train,y_test=train_test_split(features,target,test_size=test_size,random_state=58)
            return x_train,y_train,x_test,y_test
        except Exception as e:
            return log_obj.error(e)

    def model(self):

        """
        This function will create the model
        :return:
        """

        try:
            x_train, y_train, x_test, y_test=self.split_train_test()
            model=CategoricalNB()
            model.fit(x_train,y_train)
            with open('model.sav','wb') as f:
                 pickle.dump(model,f)
            return True
        except Exception as e:
            return log_obj.error(e)

    def predict(self,feature):

        """
        This function will take data and return the
        predicted value.
        :param feature:
        :return:
        """

        try:
            with open('model.sav','rb') as f:
               model=pickle.load(f)
               result=model.predict(feature)
            return result
        except Exception as e:
            return log_obj.error(e)

    def score(self,x,y):
        """
        This function will take dataset and feature as parameter
        and return the accuracy of the model.
        :param x:
        :param y:
        :return:
        """
        try:
            with open('model.sav','rb') as f:
               model=pickle.load(f)
               r=model.score(x,y)
               return r
        except Exception as e:
            return log_obj.error(e)
obj=IncomePrediction()
print(obj.model())