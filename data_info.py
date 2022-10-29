import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit,train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from warnings import simplefilter
from sklearn.exceptions import DataConversionWarning,ConvergenceWarning,FitFailedWarning
simplefilter('ignore',category=DataConversionWarning)
simplefilter('ignore',category=ConvergenceWarning)
simplefilter('ignore',category=FitFailedWarning)
simplefilter('ignore',category=UserWarning)
#simplefilter('ignore',category=SettingWithCopyWarning)
pd.options.display.max_columns=100
data=pd.read_csv('heart.csv')
features=['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
       'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
target=['HeartDisease']
X=data[features]
cat_features=list(set(features)-set(X._get_numeric_data().columns))
categorical_features=['Sex','ChestPainType','RestingECG', 'ExerciseAngina','ST_Slope']
#X['Male']=X['Sex']=='M'
#categorical to numeric 'ChestPainType' , 'RestingBP' ,'ExerciseAngina','ST_Slope'
encoder=OneHotEncoder(handle_unknown='ignore')
df2=pd.DataFrame(encoder.fit_transform(X[cat_features]).todense())
X.drop(cat_features,axis=1,inplace=True)
scaler=StandardScaler()
scaler.fit(X)
X=pd.DataFrame(scaler.transform(X))
df2.columns=encoder.get_feature_names()
df=pd.concat([X,df2],axis=1)
df.to_csv('heartdata.csv')
print(df)

y=data[target]
x_train,x_test,y_train,y_test=train_test_split(df,y,test_size=0.2,random_state=1)
LR=LogisticRegression()
params={'C':[0.01,0.1,0.2,0.4,0.6,0.8,1,10,100],
        'penalty':['l2','l1','none','elasticnet']}
grid=GridSearchCV(LR,params,cv=5)
history=grid.fit(x_train,y_train)
print(grid.best_estimator_)
print(grid.best_estimator_.score(x_test,y_test))

