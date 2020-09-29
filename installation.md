## Install Anaconda in Linux (Debian) :

- Prerequisites (Debian) :

      apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6

- Download latest Anagonda Installer depending upon your Linux distribution’s architecture from the [ Official download page](https://www.anaconda.com/distribution/#linux) or for 64b-bit OS follow the below command :

      wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh

- Go to download location, Open termial and command :

      bash Anaconda3-2020.07-Linux-x86_64.sh

- The installer prompts you to press ENTER to continue installation and then watch carefully and type 'yes' 'yes' ..... etc and do what it prompts !
  If you get `Thank you for installing Anaconda3!` then congratulation you have installed Anaconda3 successfully.
  For the installation to take effect, close and re-open your Terminal.
  You will now see the prefix **(base)** in front of your shell prompt. It means that the conda’s base environment is
  activated.

- To launch Anaconda Navigator graphical interface, run the following command from your Terminal:

      anaconda-navigator

<hr>

#### There i find some issues and problems in anaconda.I have listed them below with their solutions . I hope this will help others.

**Problems and issues**

- If you find kernel error in spyder then try to remove spyder from anaconda and thn again install spyder
- If there is a error called "ModuleError : pandas not found then use cmd :

  - `pip install wheel`
  - `pip install pandas`

- I had to use

  ```py
  from sklearn.impute import SimpleImputer
  imputer = SimpleImputer(missing_values=np.nan, strategy='mean', verbose = 0)
  ```

  as `from sklearn.preprocessing import Imputer` is not working

* the latest build of sklearn library removed **categorical_features** parameter for onehotencoder class :

```py
  from sklearn.preprocessing import OneHotEncoder
  from sklearn.compose import ColumnTransformer
  transformer = ColumnTransformer(
            transformers=[
                  (
                              "OneHot",        # Just a name
                              OneHotEncoder(), # The transformer class
                              [1]              # The column(s) to be applied on.
                        )
                  ],
            remainder='passthrough' # donot apply anything to the remaining columns
      )
  X = transformer.fit_transform(X)
```

- in `from sklearn.cross_validation import train_test_split` the `cross_validation` name is now deprecated and was replaced by `model_selection` . So we have to use "from sklearn.model_selection import train_test_split"

- **AttributeError: module 'statsmodels.formula.api' has no attribute 'OLS'**
  Use `import statsmodels.api as sm` instade of `import statsmodels.formula.api as sm`.
  and use `x_opt = np.array(x[:, [0, 1, 2, 3, 4, 5]], dtype=float)` instade of `x_opt = x[:, [0, 1, 2, 3, 4, 5]]`

```py
    import statsmodels.api as sm
    x_opt = np.array(x[:, [0, 1, 2, 3, 4, 5]], dtype=float)
    regressor_ols = sm.OLS(endog=y,exog=x_opt).fit()
```
