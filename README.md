# Machine-Learning-With-Python-and-R
**I am working with Anaconda version 2019.10 and python version 3.7.4 64bit !
and for R , i am using  R version 3.6.1 and Rstudio version 1.2.5033**

## Install Anaconda in Linux (Debian) :

To install prerequisites for Anaconda : <br>
`sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6`

---

After installing the prerequisites, download latest Anagonda Installer depending upon your Linux distribution’s architecture from the [ Official download page](https://www.anaconda.com/distribution/#linux) or for 64b-bit OS follow the below command :<br>

`wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh`


Check the integrity of the downloaded file:<br>


`sha256sum Downloads/Anaconda3-2019.10-Linux-x86_64.sh
46d762284d252e51cd58a8ca6c8adc9da2eadc82c342927b2f66ed011d1d8b53 Downloads/Anaconda3-2019.10-Linux-x86_64.sh`


Now, install Anaconda for Python 3.7 using command:<br>

`bash ~/Downloads/Anaconda3-2019.10-Linux-x86_64.sh` <br>

**Replace ~/Downloads/ with the path to the file you downloaded**

<br>

The installer prompts you to press ENTER to continue installation  and then watch carefully and type 'yes' 'yes' ..... etc !

<br>

if you get `Thank you for installing Anaconda3!` then congratulation you have installed Anaconda3 successfully.

<br>

For the installation to take effect, close and re-open your Terminal. Alternatively, run the following command:<br>

<br>

You will now see the prefix **(base)** in front of your shell prompt. It means that the conda’s base environment is 
activated.


To launch Anaconda Navigator graphical interface, run the following command from your Terminal:<br>

`anaconda-navigator`


There i find some issues and problems in anaconda and Rstudio .I have listed them below with their solutions . I hope this will help others.


<b>#problems </b>


- If you find kernel error in spyder then try to remove spyder from anaconda and thn again install spyder 
- If there is a error called "ModuleError : pandas not found then use cmd :pip install wheel then 
pip install pandas
- I had to use,from sklearn.impute import SimpleImputer imputer = SimpleImputer(missing_values=np.nan, strategy='mean', verbose = 0) as from sklearn.preprocessing import Imputer is not working 

- problem with OneHotEncoder : <br>
       from sklearn.preprocessing import OneHotEncoder
      from sklearn.compose import ColumnTransformer
      ct = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder = 'passthrough')
      X = ct.fit_transform(X)
- " from sklearn.cross_validation import train_test_split " the "cross_validation" name is now deprecated and was replaced by "model_selection" inside the new anaconda versions.  So we have to use "from sklearn.model_selection import train_test_split"
