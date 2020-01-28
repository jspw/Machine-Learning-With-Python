# Machine-Learning-With-Python-and-R
<h4>I am working with Anaconda version 2019.10 and python version 3.7.4 64bit ! <br></h4>
<h4>and for R , i am using  R version 3.6.1 and Rstudio version 1.2.5033 . <br></h4>

There i find some issues and problems in anaconda and Rstudio .I have listed them below with their solutions . I hope this will help others.

<b>#problems </b>

- If you find kernel error in spyder then try to remove spyder from anaconda and thn again install spyder 
- If there is a error called "ModuleError : pandas not found then use cmd :pip install wheel then 
pip install pandas
- I had to use,from sklearn.impute import SimpleImputer imputer = SimpleImputer(missing_values=np.nan, strategy='mean') as from sklearn.preprocessing import Imputer is not working 

- problem with OneHotEncoder : <br>
       from sklearn.preprocessing import OneHotEncoder
      from sklearn.compose import ColumnTransformer
      ct = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder = 'passthrough')
      X = ct.fit_transform(X)
