#!/usr/bin/env python
# coding: utf-8

# In[5]:


##import the necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


# In[151]:


pwd


# In[20]:


##load the data
df_gog=pd.read_csv(r'C:\\Users\\fatem\\dataset\\googleplaystore.csv')


# In[21]:


df_gog.head()


# In[7]:


df_gog.shape


# In[8]:


df_gog.tail()


# In[9]:


df_gog.describe(include='all')


# Statistical Summery:
# 
# 1.Unique apps are 9660 out of 10841 which means there are 1181 duplicate values in apps column.<br>
# 2.Apps are divide into 34 unique category of apps with 120 unique genres.<br>
# 3.Average rating: 4.19, minimum rating: 1,while avg 1st quartile ratings is 4 and avg 3rd quartile is 4.50 with maximum valus of 19.<br>
# 4.Out of 10841,10039  apps are free to install.<br>
# 

# In[10]:



df_gog.dtypes


# Columns like Reviews,Installs and Price contains numeric data but are in object form

# # Data cleaning:

# In[580]:


##Dealing with duplicate values:
df_gog.drop_duplicates(keep='first',inplace=True)
df_gog.shape


# In[75]:


#check for null values
df_gog.isnull().sum()


# In[22]:


#drop the null values of content rating:
df_gog.dropna(subset=['Content Rating'],axis=0,inplace=True)
df_gog.reset_index(drop=True,inplace=True)


# In[23]:


#replace the null values of rating with mean category wise 
df_gog['Rating']=df_gog.groupby('Category')['Rating'].transform(lambda grp:grp.fillna(np.mean(grp)))


# In[24]:


df_gog['Rating']=df_gog['Rating'].round(2)


# In[25]:


df_gog[(df_gog['Type'].isnull())]


# We will replace the null type with free as its price is 0.

# In[26]:


df_gog['Type'].replace(np.nan,'Free',inplace=True)


# In[27]:


df_gog['Current Ver'].describe()


# In[28]:


##top value of current ver is 'varies with device', lets replace the nan value of the column with it.
df_gog['Current Ver'].replace(np.nan,'Varies with device',inplace=True)


# In[29]:


df_gog['Android Ver'].describe()


# In[30]:


##replace nan values of android ver with '4.1 and up'.
df_gog['Android Ver'].replace(np.nan,'4.1 and up',inplace=True)


# In[31]:


df_gog.shape


# In[32]:


df_gog.isnull().sum()


# In[33]:


##Size column has sizes in Kb as well as Mb. To analyze, we need to convert these to numeric.
df_gog['Size'].value_counts()


# In[34]:


df_gog['Size'].replace('Varies with device',np.nan,inplace=True)


# In[35]:


df_gog.dropna(subset=['Size'],axis=0,inplace=True)
df_gog.reset_index(drop=True,inplace=True)


# In[36]:


df_gog['Size'].value_counts()


# In[37]:


df_gog['Right']=df_gog['Size'].str[-1]


# In[38]:


split=df_gog['Size'].str.split('M')


# In[39]:


df_gog['Size']=split.str[0]


# In[40]:


split1=df_gog['Size'].str.split('k')


# In[41]:


df_gog['Size']=split1.str[0]


# In[42]:


df_gog['Size']=df_gog['Size'].astype('float')


# In[43]:


subset=df_gog.loc[df_gog['Right']=='M',['Size']]*1000


# In[44]:


subset1=df_gog.loc[df_gog['Right']=='k',['Size']]


# In[45]:


df_gog['Size']=pd.concat([subset,subset1])


# In[46]:


df_gog.drop(['Right'],axis=1,inplace=True)


# In[47]:


df_gog.head()


# In[48]:


### Price field is a string and has $ symbol. Remove ‘$’ sign, and convert it to numeric.
df_gog['Price']=df_gog['Price'].apply(lambda x:str(x).replace('$',''))
df_gog['Price']=df_gog['Price'].apply(lambda x:float(x))


# In[49]:


df_gog['Price'].unique()


# In[50]:


##Installs field is currently stored as string and has values like 1,000,000+. 
df_gog['Installs'].value_counts()


# In[52]:


df_gog['Installs']=df_gog['Installs'].replace('[\+\,]','',regex=True).astype('int')


# In[53]:


##Reviews is a numeric field that is loaded as a string field. Convert it to numeric (int/float).
df_gog['Reviews']=pd.to_numeric(df_gog['Reviews'])


# In[54]:


df_gog.dtypes


# In[55]:


df_gog.info()


# # Sanity Checks:

# In[612]:


##Average rating should be between 1 and 5 as only these values are allowed on the play store. 
#Drop the rows that have a value outside this range.
df_gog[(df_gog['Rating']>5)|(df_gog['Rating']<1)].shape


# In[56]:


##Reviews should not be more than installs as only those who installed can review the app. 
##If there are any such records, drop them.
df_gog[(df_gog['Reviews']>df_gog['Installs'])].shape


# In[57]:


np.where(df_gog['Reviews']>df_gog['Installs'])


# In[58]:


df_gog.drop([1704, 3089, 4190, 4287, 4829, 4998, 5650, 6758, 7219, 8695],inplace=True,axis=0)


# In[59]:


##For free apps (type = “Free”), the price should not be >0. Drop any such rows.
df_gog[(df_gog['Type']=='Free')&(df_gog['Price']!=0)].shape


# In[60]:


df_gog.info()


# In[61]:


df_gog.head()


# In[62]:


df_gog.to_csv(r'C:\\Users\\fatem\\dataset\\clean_gooleplaystore.csv',index=False)


# # Univariate Analysis

# In[64]:


df_gog=pd.read_csv(r'C:\\Users\\fatem\\dataset\\clean_gooleplaystore.csv')


# Price

# In[65]:


sns.boxplot(data=df_gog,y='Price');
plt.title('Price univariate analysis')


# NOTE: We can see that median and mean in this plot is same which is 0 because
# Free apps are very as compared to the paid applications,hence making other data points  outliers
# Therefore we will only look at the price of paid appliacation

# In[66]:


sns.boxplot(data=df_gog,y=df_gog.loc[(df_gog['Type']=='Paid')&(df_gog['Price']>0),['Price']]);

plt.title('Price-paid univariate analysis')


# Almost 80% of the apps are in the range of 1 dollar to 70 dollars.

# In[67]:


print(df_gog['Price' ].skew())
df_gog['Price'].describe()


# The skewness value of 21.53 shows that the variable 'Price' has a right-skewed distribution, 
# indicating the presence of extreme higher values.The maximum 'Price' value of 400 proves this point.

# Reviews

# In[68]:


sns.boxplot(data=df_gog,y='Reviews');
plt.title('Price univariate analysis')


# In[69]:


print(df_gog['Price' ].skew())
df_gog['Price'].describe()


# There are apps with very high number of reviews.

# Rating

# In[70]:


sns.distplot(df_gog['Rating'],kde=True,hist=True,bins=10);
plt.title('Rating univariate analysis')


# In[71]:


print(df_gog['Rating' ].skew())
df_gog['Rating'].describe()


# Rating is left skewed(negative skew),median is 4.3 from descriptive analysis. Most of the apps have higher ratings above 4.0

# Size

# In[72]:


Size=df_gog['Size']
sns.distplot(Size,kde=True,hist=True);
plt.title('Size univariate analysis')


# In[73]:


print(df_gog['Size' ].skew())
df_gog['Size'].describe()


# The distribution is right skewed with skew score of 1.44 showing most of the apps are ligher in size.

# # Outlier Treatment:

# Price:

# In[74]:


##From the box plot, it seems like there are some apps with very high price. 
##A price of $200 for an application on the Play Store is very high and suspicious!
z=np.abs(stats.zscore(df_gog['Price']))
threshold=3
outlier_price=df_gog.loc[(z>3)]
outlier_price


# There are very few apps with the price above 200$.After looking these apps on google app it turns out their developer keep changing the prices and the apps are there to show others you are rich.

# In[75]:


##Lets drop these as most seem to be junk apps
df_gog=df_gog[(df_gog['Price']<=200)]


# In[76]:


df_gog.shape


# Reviews:

# In[77]:


##Very few apps have very high number of reviews.
df_gog.loc[(df_gog['Reviews']>2_000_000),['App']]


# In[78]:


####These are all star apps that don’t help with the analysis and, in fact, will skew it. 
##Drop records having more than 2 million reviews.
df_gog.drop(df_gog[df_gog['Reviews'] >2000000].index, inplace=True)


# Installs:

# In[79]:


##There seems to be some outliers in this field too. 

df_gog['Installs'].quantile([0.10,0.25,0.50,0.70,0.90,0.95,0.99])


# In[80]:


##Decide a threshold as cutoff for outlier 
z=np.abs(stats.zscore(df_gog['Installs']))
threshold=3 
outlier_installs=df_gog.loc[(z>3)]
outlier_installs


# In[81]:


##Apps having very high number of installs should be dropped from the analysis.
df_gog.drop(df_gog[df_gog['Installs'] >50000000.0].index, inplace=True)


# # Bivariate Analysis:
# 
# Let’s look at how the available predictors relate to the variable of interest, i.e., our target variable rating. Make scatter plots (for numeric features) and box plots (for character features) to assess the relations between rating and the other features.

# In[82]:


##find the correaltion between the variables
df_gog.corr()


# In[91]:


sns.set(rc={'figure.figsize':(10,8)})
sns.heatmap(df_gog.corr(),annot=True,cmap='winter') 
plt.title('Correlation between the variables')


# Installs vs Reviews

# In[678]:


### As installs has the highest correlation with Reviews:
sns.scatterplot(x='Installs',y='Reviews',data=df_gog);
plt.title('Installs vs Reviews bivariate analysis')


# The Installs and Reviews seem to be moderately correalted(0.75).This relation ship could explain why more populer categories have more installs and more reviews.We can use only one of the two for model.

# Rating vs. Price

# In[337]:


sns.scatterplot(x='Price',y='Rating',data=df_gog);
plt.title('Price vs Rating bivariate analysis')


# Free apps  have the range of ranting between 1 to 5.While paid apps semms to have almost higher rating.There sems no pattern here.

# In[284]:


sns.scatterplot(x='Size',y='Rating',data=df_gog);
plt.title('Size vs Rating bivariate analysis')


# Most top rated apps are optimalle sized between 2000k to 40000k neither too light nor too heavy.
# bulky apps are fairly high rated indicating they are bulky for a purpose.

# In[285]:


sns.scatterplot(x='Reviews',y='Rating',data=df_gog)
plt.title('Reviews vs Rating bivariate analysis')


# The higher rated apps have more reviews than lower rated apps. Some of the higher rated apps have significantly more reviews than others. This could be caused through in-app pop ups or in-app incentives.

# In[83]:


plt.figure(figsize=(8, 6))
sns.boxplot(data=df_gog,x='Content Rating',y='Rating');
plt.title('Content Rating vs Rating bivariate analysis')


# The content Rting does not really affect the rating despite most apps being in the Everyone category.

# In[84]:


plt.figure(figsize=(12, 7))
sns.boxplot(data=df_gog,x='Category',y='Rating');
plt.xticks(rotation=90)
plt.title('Category vs Rating bivariate analysis')


# The categories with the highest range of ratings are Art & Design,Events,and Parenting.
# The Dating category has the lowest range of ratings.Almost all the categories has median rating between 4.0 to 4.5.

# In[87]:


df_gog.to_csv(r'C:\\Users\\fatem\\dataset\\treated_gooleplaystore.csv',index=False)


# # Data Processing:

# In[6]:


df=pd.read_csv(r'C:\\Users\\fatem\\dataset\\treated_gooleplaystore.csv')


# For the categorial column,i converted each category into an invidiual number. in the later section when we do apply machine learning, two methods will be applied to the code, being integer encoding and one-hot encoding aka dummy variables. This is because integer encoding relies on the fact that there is a relationshio between each category to provide better predictive accuracy.
# 

# In[7]:


## cleaning categories into integers
categorystring=df['Category']
categoryval=df['Category'].unique()
categoryvalcount=len(categoryval)
category_dict={}
for i in range(0,categoryvalcount):
    category_dict[categoryval[i]]=i
df['Category_c']=df['Category'].map(category_dict).astype(int)


# In[8]:


### converting type classification into binary
def type_cat(types):
    if types=='Free':
        return 0
    else:
        return 1
    
df['Type']=df['Type'].map(type_cat)


# In[9]:


#cleaning of content rating classification
Rating_list=df['Content Rating'].unique()
Rating_dict={}
for i in range(len(Rating_list)):
    Rating_dict [Rating_list[i]]=i
    
df['Content Rating']=df['Content Rating'].map(Rating_dict).astype(int)


# In[10]:


#dropping of unrelated and unnecessary items
df.drop(['App','Last Updated', 'Current Ver', 'Android Ver'],axis=1,inplace=True)


# In[11]:


#cleaning of genres
Genres_list=df.Genres.unique()
Genres_dict={}
for i in range(len(Genres_list)):
    Genres_dict[Genres_list[i]]=i
    
df['Genres_c']=df['Genres'].map(Genres_dict).astype(int)


# In[12]:


df.info()


# Creating another dataframe that specifically creates dummy variables for each categorical instance in the dataframe.

# In[13]:


#for dummy variables encoding for categories
df2=pd.get_dummies(df,columns=['Category'])


# In[14]:


df2.head()


# I chose 2 most common models i.e linear regression and randome forst regressor. We will technically run 4 regressions for each model used as we consider one-hot vs integer encoded results for the category section,as well as including/excluding the genres section.

# # Model Building

# The following is the code to obtain the error terms for the various models,for comparability.

# In[38]:


from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[39]:


#For evaluation of error term
def Evaluationmatrix(y_true, y_predict):
    print ('Mean Squared Error: '+ str(metrics.mean_squared_error(y_true,y_predict)))
    print ('Mean absolute Error: '+ str(metrics.mean_absolute_error(y_true,y_predict)))
    print ('R2 Score:'+ str(metrics.r2_score(y_true_t,y_predict)))
    


# In[40]:


#to add into results_index for evaluation of error term
def Evaluationmatrix_dict(y_true, y_predict, name = 'Linear - Integer'):
    dict_matrix = {}
    dict_matrix['Series Name'] = name
    dict_matrix['Mean Squared Error'] = metrics.mean_squared_error(y_true,y_predict)
    dict_matrix['Mean Absolute Error'] = metrics.mean_absolute_error(y_true,y_predict)
    dict_matrix['R2 Score']=metrics.r2_score(y_true,y_predict)
    return dict_matrix


# Linear model - Excluding Genres

# In[41]:


#excluding Genre label
from sklearn.linear_model import LinearRegression 

#Integer encoding
X = df.drop(labels = ['Category','Rating','Genres','Genres_c'],axis = 1)
y = df.Rating
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
model = LinearRegression()
model.fit(X_train,y_train)
Results = model.predict(X_test)

#Creation of results dataframe and addition of first entry
resultsdf = pd.DataFrame()
resultsdf = resultsdf.from_dict(Evaluationmatrix_dict(y_test,Results),orient = 'index')
resultsdf = resultsdf.transpose()

#dummy encoding

X_d = df2.drop(labels = ['Rating','Genres','Category_c','Genres_c'],axis = 1)
y_d = df2.Rating
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y_d, test_size=0.30)
model_d = LinearRegression()
model_d.fit(X_train_d,y_train_d)
Results_d = model_d.predict(X_test_d)

#adding results into results dataframe
resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test_d,Results_d, name = 'Linear - Dummy'),ignore_index = True)


# In[42]:


plt.figure(figsize=(12,7))
sns.regplot(Results,y_test,color='teal', label = 'Integer', marker = 'x')
sns.regplot(Results_d,y_test_d,color='orange',label = 'Dummy')
plt.legend()
plt.title('Linear model - Excluding Genres')
plt.xlabel('Predicted Ratings')
plt.ylabel('Actual Ratings')
plt.show()


# In[43]:


print ('Actual mean of population:' + str(y.mean()))
print ('Integer encoding(mean) :' + str(Results.mean()))
print ('Dummy encoding(mean) :'+ str(Results_d.mean()))
print ('Integer encoding(std) :' + str(Results.std()))
print ('Dummy encoding(std) :'+ str(Results_d.std()))


# If we look at the actual mean of the predictive results, both are approximately the same, however the dummy encoded results have a much larger standard deviation as compared to the integer encoded model.
# 
# Next is looking at the linear model including the genre label as a numeric value.

#  Linear model - Including Genres

# In[44]:


#Including genre label

#Integer encoding
X = df.drop(labels = ['Category','Rating','Genres'],axis = 1)
y = df.Rating
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
model = LinearRegression()
model.fit(X_train,y_train)
Results = model.predict(X_test)

resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test,Results, name = 'Linear(inc Genre) - Integer'),ignore_index = True)

#dummy encoding

X_d = df2.drop(labels = ['Rating','Genres','Category_c'],axis = 1)
y_d = df2.Rating
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y_d, test_size=0.30)
model_d = LinearRegression()
model_d.fit(X_train_d,y_train_d)
Results_d = model_d.predict(X_test_d)

resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test_d,Results_d, name = 'Linear(inc Genre) - Dummy'),ignore_index = True)


# In[45]:


plt.figure(figsize=(12,7))
sns.regplot(Results,y_test,color='teal', label = 'Integer', marker = 'x')
sns.regplot(Results_d,y_test_d,color='orange',label = 'Dummy')
plt.legend()
plt.title('Linear model - Including Genres')
plt.xlabel('Predicted Ratings')
plt.ylabel('Actual Ratings')
plt.show()


# In[46]:


print ('Integer encoding(mean) :' + str(Results.mean()))
print ('Dummy encoding(mean) :'+ str(Results_d.mean()))
print ('Integer encoding(std) :' + str(Results.std()))
print ('Dummy encoding(std) :'+ str(Results_d.std()))


# When including the genre data, we see a slight difference in the mean between the integer and dummy encoded linear models. The dummy encoded model's std is still higher than the integer encoded model.
# 
# 

# RFR model - excluding Genres

# In[47]:


from sklearn.ensemble import RandomForestRegressor

#Integer encoding
X = df.drop(labels = ['Category','Rating','Genres','Genres_c'],axis = 1)
y = df.Rating
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
model3 = RandomForestRegressor()
model3.fit(X_train,y_train)
Results3 = model3.predict(X_test)

#evaluation
resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test,Results3, name = 'RFR - Integer'),ignore_index = True)

#dummy encoding

X_d = df2.drop(labels = ['Rating','Genres','Category_c','Genres_c'],axis = 1)
y_d = df2.Rating
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y_d, test_size=0.30)
model3_d = RandomForestRegressor()
model3_d.fit(X_train_d,y_train_d)
Results3_d = model3_d.predict(X_test_d)

#evaluation
resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test,Results3_d, name = 'RFR - Dummy'),ignore_index = True)


# In[48]:


plt.figure(figsize=(12,7))
sns.regplot(Results3,y_test,color='teal', label = 'Integer', marker = 'x')
sns.regplot(Results3_d,y_test_d,color='orange',label = 'Dummy')
plt.legend()
plt.title('RFR model - excluding Genres')
plt.xlabel('Predicted Ratings')
plt.ylabel('Actual Ratings')
plt.show()


# In[49]:


print ('Integer encoding(mean) :' + str(Results3.mean()))
print ('Dummy encoding(mean) :'+ str(Results3_d.mean()))
print ('Integer encoding(std) :' + str(Results3.std()))
print ('Dummy encoding(std) :'+ str(Results3_d.std()))


# At first glance, I would say that the RFR model produced the best predictive results, just looking at the scatter graph plotted. Overall both models, the integer and the dummy encoded models seem to perform relatively similar, although the dummy encoded model has a higher overall predicted mean.

# In[50]:


#for integer
Feat_impt = {}
for col,feat in zip(X.columns,model3.feature_importances_):
    Feat_impt[col] = feat

Feat_impt_df = pd.DataFrame.from_dict(Feat_impt,orient = 'index')
Feat_impt_df.sort_values(by = 0, inplace = True)
Feat_impt_df.rename(index = str, columns = {0:'Pct'},inplace = True)

plt.figure(figsize= (14,10))
Feat_impt_df.plot(kind = 'barh',figsize= (14,10),legend = False)
plt.show()


# If we look at what influences the ratings, the top 4 being reviews, size, category, and number of installs seem to have the highest influence. This is quite an interesting observation, while also rationalizable.

# In[51]:


#for dummy
Feat_impt_d = {}
for col,feat in zip(X_d.columns,model3_d.feature_importances_):
    Feat_impt_d[col] = feat

Feat_impt_df_d = pd.DataFrame.from_dict(Feat_impt_d,orient = 'index')
Feat_impt_df_d.sort_values(by = 0, inplace = True)
Feat_impt_df_d.rename(index = str, columns = {0:'Pct'},inplace = True)

plt.figure(figsize= (14,10))
Feat_impt_df_d.plot(kind = 'barh',figsize= (14,10),legend = False)
plt.show()


# Looking at the breakdown even further, it would seem that indeed Reviews, size and number of install remain as a significant contributer to the predictiveness of app ratings. What's interesting to me is that how the Family category of apps have such a high level of predictiveness in terms of ratings, as say compared to the and Weather category.

# RFR model - including Genres

# In[52]:


#Including Genres_C

#Integer encoding
X = df.drop(labels = ['Category','Rating','Genres'],axis = 1)
y = df.Rating
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
model3a = RandomForestRegressor()
model3a.fit(X_train,y_train)
Results3a = model3a.predict(X_test)

#evaluation
resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test,Results3a, name = 'RFR(inc Genres) - Integer'),ignore_index = True)

#dummy encoding

X_d = df2.drop(labels = ['Rating','Genres','Category_c'],axis = 1)
y_d = df2.Rating
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y_d, test_size=0.30)
model3a_d = RandomForestRegressor()
model3a_d.fit(X_train_d,y_train_d)
Results3a_d = model3a_d.predict(X_test_d)

#evaluation
resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test,Results3a_d, name = 'RFR(inc Genres) - Dummy'),ignore_index = True)


# In[53]:


plt.figure(figsize=(12,7))
sns.regplot(Results3a,y_test,color='teal', label = 'Integer', marker = 'x')
sns.regplot(Results3a_d,y_test_d,color='orange',label = 'Dummy')
plt.legend()
plt.title('RFR model - including Genres')
plt.xlabel('Predicted Ratings')
plt.ylabel('Actual Ratings')
plt.show()


# In[54]:


print ('Integer encoding(mean) :' + str(Results3.mean()))
print ('Dummy encoding(mean) :'+ str(Results3_d.mean()))
print ('Integer encoding(std) :' + str(Results3.std()))
print ('Dummy encoding(std) :'+ str(Results3_d.std()))


# Again with the inclusion of the genre variable, the results do not seem to defer significantly as compared to the previous results.

# In[55]:


#for integer
Feat_impt = {}
for col,feat in zip(X.columns,model3a.feature_importances_):
    Feat_impt[col] = feat

Feat_impt_df = pd.DataFrame.from_dict(Feat_impt,orient = 'index')
Feat_impt_df.sort_values(by = 0, inplace = True)
Feat_impt_df.rename(index = str, columns = {0:'Pct'},inplace = True)

plt.figure(figsize= (14,10))
Feat_impt_df.plot(kind = 'barh',figsize= (14,10),legend = False)
plt.show()


# From the results, it would seem that the Reviews section actually plays an important part in the decision tree making. Yet the exclusion of it dosent seem to significantly impact results. This to me is quite interesting.

# In[56]:


#for dummy
Feat_impt_d = {}
for col,feat in zip(X_d.columns,model3a_d.feature_importances_):
    Feat_impt_d[col] = feat

Feat_impt_df_d = pd.DataFrame.from_dict(Feat_impt_d,orient = 'index')
Feat_impt_df_d.sort_values(by = 0, inplace = True)
Feat_impt_df_d.rename(index = str, columns = {0:'Pct'},inplace = True)

plt.figure(figsize= (14,10))
Feat_impt_df_d.plot(kind = 'barh',figsize= (14,10),legend = False)
plt.show()


# In[57]:


resultsdf


# In[58]:


resultsdf.set_index('Series Name', inplace = True)


# In[60]:



plt.figure(figsize = (10,12))
plt.subplot(3,1,1)
resultsdf['Mean Squared Error'].sort_values(ascending = False).plot(kind = 'barh',color=(0.3, 0.4, 0.6, 1), title = 'Mean Squared Error')
plt.subplot(3,1,2)
resultsdf['Mean Absolute Error'].sort_values(ascending = False).plot(kind = 'barh',color=(0.5, 0.4, 0.6, 1), title = 'Mean Absolute Error')
plt.subplot(3,1,3)
resultsdf['R2 Score'].sort_values(ascending = False).plot(kind = 'barh',color=(0.7, 0.4, 0.6, 1), title = 'R2 Score')
plt.show()


# Finally, looking at the results, itcan be concluded that integer encode model performed better with RFR and Dummy encoded model performed better with Liner.Using this round of data as a basis, the integer encoded RFR including Genres has the lowest overall error rates, followed by the integer encoded RFR excluding Genres . Yet, all models seem to be very close in terms of it's error term, but thr R2 for dummy encoded linear model seems to be with minus values.
# What is very surprising to me is how the RFR dummy model has such a significantly more error term compared to all the other models,even though on the surface it seemed to perform very similar to the RFR integer model.
# 
# 

# In[ ]:




