#!/usr/bin/env python
# coding: utf-8

# # Project Overview  
#   
# This project has two parts that demonstrate the importance and value of data visualization techniques in the data analysis process. In the first part, you will use Python visualization libraries to systematically explore a selected dataset, starting from plots of single variables and building up to plots of multiple variables. In the second part, you will produce a short presentation that illustrates interesting properties, trends, and relationships that you discovered in your selected dataset. The primary method of conveying your findings will be through transforming your exploratory visualizations from the first part into polished, explanatory visualizations.

# ## Table of Contents
# - [Introduction](#intro)
# - [Part I - Data Wrangling and Data Exploration](#data_wrangling)
# - [Part II - Explanatory Data Analysis](#eda)
# - [Part III - Conclusions](#conclusions)

# <a id='intro'></a>
# ## Why this project?  
#   
# Data visualization is an important skill that is used in many parts of the data analysis process.  
#   
# __Exploratory__ data visualization generally occurs during and after the data wrangling process, and is the main method that you will use to understand the patterns and relationships present in your data. This understanding will help you approach any statistical analyses and will help you build conclusions and findings. This process might also illuminate additional data cleaning tasks to be performed. 
#   
# __Explanatory__ data visualization techniques are used after generating your findings, and are used to help communicate your results to others. Understanding design considerations will make sure that your message is clear and effective. In addition to being a good producer of visualizations, going through this project will also help you be a good consumer of visualizations that are presented to you by others.

# For this project I choose to analyse the results of the OECD __Programme for International Student Assessment (PISA)__ in 2012.  
# 
# From OECD website:
# _PISA is an international study that was launched by the OECD in 1997, first administered in 2000 and now covers over 80 countries. Every 3 years the PISA survey provides comparative data on 15-year-olds’ performance in reading, mathematics, and science. In addition, each cycle explores a distinct “innovative domain” such as Collaborative Problem Solving (PISA 2015) and Global Competence (PISA 2018). The results have informed education policy discussions at the national and global level since its inception._  
# https://www.oecd.org/pisa/aboutpisa/pisa-based-test-for-schools-faq.htm  
# 
# __The PISA goals are:__  
# - Empower school leaders and teachers by providing them with evidence-based analysis of their students’ performance.  
# - Measure students’ knowledge, skills and competencies that will equip them for success in education and the world of work.  
# - Provide valuable information on the learning climate within a school, students’ socioeconomic background and motivation for learning.  
# - Help schools measure a wider range of 21st century skills beyond maths, reading and science.  
# - Provide opportunities for global peer-learning among teachers and school leaders.  
# 
# __Based on the objectives of the PISA, using the data, the following questions can be answered:__  
# 1. What is students’ performance at schools in different countries (including whether country is a OECD member).  
# 2. What are the characteristics of students participated in PICA 2012:  
#     * gender,  
#     * age,  
#     * whether a student passed the test in the country of birth or not,  
#     * international grade and grade compared to modal grade in country.  
# 3. What's a relationship between students performance and highest parental education measured in years as well as mother's and father's highest schooling?  
# 4. Whether there exist a correlation between family wealth (measured in the number of telephones, computers, etc.) and students performance?  
# 5. How do student possessions such as own room and desk, etc. affect his/her performance?  
# 6. Last but not least, whether total time learning and out of school lessons on math, science, and reading affect student performance?

# In[1]:


# Import libraries

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option('display.max_columns', 700)


# In[2]:


FOLDER = 'C:/Sasha/udacity/Data Analyst Nanodegree Program/5. Data Visualization/8. Communicate Data Findings'

FILE_NAME = 'pisa2012.csv'
DICT_NAME = 'pisadict2012.csv'

file_path = os.path.join(FOLDER, FILE_NAME)
dict_path = os.path.join(FOLDER, DICT_NAME)


# In[3]:


DTYPES = {'SCHOOLID': 'str', 'STIDSTD': 'str'}


# In[4]:


# Load in data 

df = pd.read_csv(file_path, dtype=DTYPES, encoding='ISO-8859-1')
df.head(2)


# Let's first drop unused columns that could slow down our work.

# In[5]:


cols_to_keep = ['CNT', 'OECD', 'SCHOOLID', 'STIDSTD', 'AGE', 'ST04Q01', 'ST20Q01', 'ST01Q01', 'GRADE',
                'ST11Q01', 'ST11Q02', 'ST11Q05', 'PARED', 'ST13Q01', 'ST17Q01', 'ST27Q01', 'ST27Q02', 'ST27Q03',
                'ST27Q04', 'ST27Q05', 'ST28Q01', 'ST26Q01', 'ST26Q02', 'ST26Q03', 'ST26Q04', 'ST26Q05', 'ST26Q06',
                'ST55Q01', 'ST55Q02', 'ST55Q03', 'LMINS', 'SMINS', 'MMINS',
                'PV1MATH', 'PV2MATH', 'PV3MATH', 'PV4MATH', 'PV5MATH', 'PV1SCIE', 'PV2SCIE', 'PV3SCIE',
                'PV4SCIE', 'PV5SCIE', 'PV1READ', 'PV2READ', 'PV3READ', 'PV4READ', 'PV5READ']

df = df[cols_to_keep]
df.head(2)


# We can look at meaning of columns using PISA dictionary data.

# In[6]:


# Load in dict data

dict_df = pd.read_csv(dict_path, header=None, names=['value', 'meaning'], skiprows=1, encoding='ISO-8859-1')
dict_df.head()


# In[7]:


dict_df[dict_df['value'].isin(df.columns)]


# ### General Properties

# In[8]:


print('Shape:', df.shape[0], 'rows and', df.shape[1], 'columns')


# There're 1471 schools with 33806 students in the dataset.

# In[9]:


df.SCHOOLID.nunique(), df.STIDSTD.nunique()


# Schools and students identifiers were loaded as integers first time, so their types were specified directly when loading data.

# In[10]:


df.dtypes


# In[11]:


# First, I think, "ST55Q.." clumns should be numeric

df.ST55Q01.unique()


# Most rows have some missing values.

# In[12]:


df.info()


# In[13]:


# Check missing data

df_missing = df.isnull().sum().sort_values(ascending=False)
df_missing[df_missing > 0] / df.shape[0]


# In[14]:


df[['SMINS', 'LMINS', 'MMINS']].head(10)


# Columns SMINS, LMINS, and MMINS (that's Learning time (minutes per week)) have more than 40% missing values. So, last 7th question could be modified as follows:  
# Whether absense of data about learning time on math, science, and reading, affect students performance? And if a student report info about total time learning, how these influence each grade in assesment?

# The same could be applied to ST55Q03, ST55Q01, and ST55Q02 (Out of school lessons) because of high percent (about 36%) of missingvalues.

# Finally, it's also more than a quater missig values in ST11Q05 column (At Home - Grandparents). He we can asume, NaN in ST11Q05 could be also considered as missing data.

# In[15]:


df.ST11Q05.unique()


# In all other columns share of missing data isn't exceed than 10%. So, for explanatory data analysis, we could ignore this distortions, and drop missing values.

# In[16]:


for column in df_missing.index:
    if column not in ('SMINS', 'LMINS', 'MMINS', 'ST55Q03', 'ST55Q01', 'ST55Q02', 'ST11Q05'):
        df[column].fillna(df[column].mode()[0], inplace=True)


# Check NaN
df_missing = df.isnull().sum().sort_values(ascending=False)
df_missing[df_missing > 0] / df.shape[0]


# In[17]:


# Check duplicated rows

df.duplicated().sum() # => there's no duplicated rows


# In[18]:


# Summary statistics of numeric columns

df.describe()


# <a id='data_wrangling'></a>
# ## Part I - Data Wrangling and Data Exploration  

# In[19]:


# Check values of object columns , and simplify where possible

cat_cols = df.select_dtypes(include='object').columns
cat_cols = cat_cols.drop(['SCHOOLID', 'STIDSTD', 'CNT'])
print('Number of string columns', cat_cols.shape[0])
cat_cols


# There's a lot of categories in CNT column (countries). Look at them seperately.

# In[20]:


for column in cat_cols:
    print(column, df[column].nunique(), df[column].unique())


# For column "How many properties are at home?" replace "None" and NaN values with zeros.

# In[21]:


cols_with_none = 'ST27Q01', 'ST27Q02', 'ST27Q03', 'ST27Q04', 'ST27Q05'
for column in cols_with_none:
    df[column] = df[column].replace(['None', np.nan], 'Zero')


# In[22]:


print('Number of unique countries', df['CNT'].nunique())
df.CNT.unique()


# Some change could be made:
# 1. Hong Kong-China -> Hong Kong
# 2. China-Shanghai -> China
# 3. Perm(Russian Federation) -> Russian Federation (since Perm is just a city in RF)
# 4. Florida (USA) -> United States of America
# 5. Connecticut (USA) -> United States of America
# 6. Massachusetts (USA) -> United States of America
# 7. Chinese Taipei -> Taiwan
# 8. Macao-China -> Macao

# In[23]:


# Implement changes in NCT column

df['CNT'] = (df['CNT'].replace('Hong Kong-China', 'Hong Kong')
                        .replace('China-Shanghai', 'China')
                        .replace('Perm(Russian Federation)', 'Russian Federation')
                        .replace('Florida (USA)', 'United States of America')
                        .replace('Connecticut (USA)', 'United States of America')
                        .replace('Connecticut (USA)', 'United States of America')
                        .replace('Massachusetts (USA)', 'United States of America')
                        .replace('Chinese Taipei', 'Taiwan')
                        .replace('Macao-China', 'Macao'))


# In[24]:


# Check changes

print('Number of unique countries', df['CNT'].nunique())
df['CNT'].unique()


# Finally, the independent variables "PV..." - plausible values in math, science, and reading - will be summed and diveded by 5 (number of column of each subject). 

# In[25]:


df['PV_MATH'] = (df.PV1MATH + df.PV2MATH + df.PV3MATH + df.PV4MATH + df.PV5MATH) / 5
df['PV_SCIE'] = (df.PV1SCIE + df.PV2SCIE + df.PV3SCIE + df.PV4SCIE + df.PV5SCIE) / 5
df['PV_READ'] = (df.PV1READ + df.PV2READ + df.PV3READ + df.PV4READ + df.PV5READ) / 5

df[['PV_MATH', 'PV_SCIE', 'PV_READ']].describe()


# In[26]:


# Drop initial "PV..." columns

df.drop(['PV1MATH', 'PV2MATH', 'PV3MATH', 'PV4MATH', 'PV5MATH', 'PV1SCIE', 'PV2SCIE', 'PV3SCIE', 'PV4SCIE',
         'PV5SCIE', 'PV1READ', 'PV2READ', 'PV3READ', 'PV4READ', 'PV5READ'], axis=1, inplace=True)


# <a id='eda'></a>
# ## Part II - Explanatory Data Analysis

# In[27]:


df.head(3)


# #### 1. What is students’ performance at schools in different countries (including whether country is a OECD member).

# In[28]:


base_color = sns.color_palette()[0]


# In[29]:


fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(21,7))
df['PV_MATH'].hist(ax=ax[0])
df['PV_SCIE'].hist(ax=ax[1])
df['PV_READ'].hist(ax=ax[2])
ax[0].set_xlabel('Math scores')
ax[1].set_xlabel('Science scores')
ax[2].set_xlabel('Reading scores')
plt.suptitle('Distribution of grades')
plt.show()


# In[30]:


df['PV_MATH'].mean(), df['PV_SCIE'].mean(), df['PV_READ'].mean()


# If we plot all the grades by subject, then scores in each subject looks normally distributed. Mean scores of science are about 3 points higher than average reading scores. In its turn, avearge reading scores are about 3 poits higher than avearge math scores. So scores in those 3 subjects are very similar.  
# So, let's look at their boxplots. 

# In[31]:


fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(21,7))
df['PV_MATH'].plot(kind='box', ax=ax[0])
df['PV_SCIE'].plot(kind='box', ax=ax[1])
df['PV_READ'].plot(kind='box', ax=ax[2])
ax[0].set_xlabel('Math scores')
ax[1].set_xlabel('Science scores')
ax[2].set_xlabel('Reading scores')
plt.suptitle('Boxplots of grades')
plt.show()


# In general, there're outliers in every Series of scores. Moreover, math scores have approximately equal tails of outliers, but science and reading scores have outliers with lower scores more, than outliers with higher scores.
# let's go deepper, and look at students perormance in the context of countries, OECD membership, and other columns.

# In[32]:


df_plot1 = df.groupby('CNT').agg({'PV_MATH': 'mean', 'PV_SCIE': 'mean', 'PV_READ': 'mean', 'STIDSTD': 'nunique'})

print('Shape', df_plot1.shape)
df_plot1.head()


# In[33]:


df_plot1[['PV_MATH']].sort_values('PV_MATH').iloc[-15:].plot.barh(figsize=(21,7), width=.9, legend=False,
                                                                  cmap='Blues_r')
plt.yticks(fontsize=12)
plt.xlabel('Average math scores')
plt.ylabel('Countries')
plt.title('Top-15 Average math scores by Countries')
plt.show()


# In[34]:


df[df.CNT == 'China'].PV_MATH.mean(), df[df.CNT == 'Peru'].PV_MATH.mean()


# Except Liechtenstein which is on the 6th position, on average, students from Asia countries receive the highest scores on math. China, Singapore, Hong Kong, Taiwan, and Korea are in Top-5. Macao and Japan follow immediately behind Liechtenstein.  
# Chinese students receive on average 611 points. In comparison, in Peru average math scores are equal 368. This's 1.7 times less than in China.

# In[35]:


df_plot1[['PV_SCIE']].sort_values('PV_SCIE').iloc[-15:].plot.barh(figsize=(21,7), width=.9, legend=False,
                                                                  cmap='Blues_r')
plt.yticks(fontsize=12)
plt.xlabel('Average science scores')
plt.ylabel('Countries')
plt.title('Top-15 Average science scores by Countries')
plt.show()


# In[36]:


(df[df.CNT == 'China'].PV_SCIE.mean(), df[df.CNT == 'Hong Kong'].PV_SCIE.mean(),
            df[df.CNT == 'Singapore'].PV_SCIE.mean())


# Average science scores are less than math scores by about 6 points. And this is becoming noticeable for countries with the highest average scores in science. China, Gang Kong and Singapore are also in the Top-3 with an average score of 547 to 579. For China, this difference is 32 points or 5.2%. 

# In[37]:


df_plot1[['PV_READ']].sort_values('PV_READ').iloc[-15:].plot.barh(figsize=(21,7), width=.9, legend=False,
                                                                  cmap='Blues_r')
plt.yticks(fontsize=12)
plt.xlabel('Average reading scores')
plt.ylabel('Countries')
plt.title('Top-15 Average reading scores by Countries')
plt.show()


# In[38]:


df[df.CNT == 'China'].PV_READ.mean()


# For average reading scores, China, Hong Kong, Singapore, Japan, Korea and Taiwan continue to be the leaders with a maximum average of 569 points for China. This average score is the lowest for China in three subjects, possibly also because English is not a native language for a large population of the country.

# In[39]:


df_plot2 = df.groupby('OECD').agg({'PV_MATH': 'mean', 'PV_SCIE': 'mean', 'PV_READ': 'mean', 'STIDSTD': 'nunique'})
print(df_plot2)


# In[40]:


df_plot2[['PV_MATH', 'PV_SCIE', 'PV_READ']].sort_values('PV_READ', ascending=False).plot.bar(
                    figsize=(21,7), width=.9, legend=True, cmap='viridis', rot=0)
plt.legend(['Math', 'Science', 'Reading'])
plt.yticks(fontsize=12)
plt.xlabel('Average reading scores')
plt.ylabel('Countries')
plt.title('Average reading scores by OECD membership')
plt.show()


# The difference is noticeable in all three subjects at once: average scores in mathematics, science and reading are higher in OSCE countries than in non-OSCE countries. The difference is about 48 points for each subject.

# #### 2. What are the characteristics of students participated in PICA 2012:
# * gender,  
# * age,  
# * whether a student passed the test in the country of birth or not,  
# * international grade and grade compared to modal grade in country.

# In[41]:


df.groupby('ST04Q01').STIDSTD.nunique().sort_index().plot.bar(figsize=(21,7), width=.9, legend=False,
                                                                  cmap='Blues_r', rot=0)
plt.xlabel('Gender')
plt.title('Student\'s by Gender')
plt.show()


# Number of female students is little more (by 0.6%) than number of male students. Let's look at gender by countries.

# The largest number of students are in Mexico, Italy, Spain, Canada and Brazil. Except for Italy, the number of the females is greater than that of the male. In Brazil, there are 8% fewer males than females. The number of students in Mexico is 1.8 times higher than in Brazil, which is in 5th place, the number of males is 2.8 times less and the number of females is 2.7.

# In[42]:


fig, ax = plt.subplots(figsize=(21,35))
df.sort_values(by=['ST04Q01', 'CNT'])
sns.countplot(data=df, y='CNT', hue='ST04Q01', ax=ax, order=df['CNT'].value_counts().index)
ax.legend(title='Gender', loc='best')
plt.xlabel('Students')
plt.ylabel('Country')
plt.title('Student\'s by Gender by Countries')
plt.show()


# In[43]:


fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(21,7))
df[df.ST04Q01 == 'Female']['PV_MATH'].hist(ax=ax[0], alpha=.5, label='Female')
df[df.ST04Q01 == 'Male']['PV_MATH'].hist(ax=ax[0], alpha=.5, label='Male')
ax[0].set_xlabel('Math scores')
ax[0].legend()

df[df.ST04Q01 == 'Female']['PV_SCIE'].hist(ax=ax[1], alpha=.5, label='Female')
df[df.ST04Q01 == 'Male']['PV_SCIE'].hist(ax=ax[1], alpha=.5, label='Male')
ax[1].set_xlabel('Science scores')
ax[1].legend()

df[df.ST04Q01 == 'Female']['PV_READ'].hist(ax=ax[2], alpha=.5, label='Female')
df[df.ST04Q01 == 'Male']['PV_READ'].hist(ax=ax[2], alpha=.5, label='Male')
ax[2].set_xlabel('Reading scores')
ax[2].legend()

plt.suptitle('Distribution of grades by Gender')
plt.show()


# Distriution of males and females math and science score are distributed approximately normal. However, there's slight difference of reading scores: female have aslightly higher grades than males.

# Since student age is between 15 and 16 year old, and number of students who are 15 years old are twice larger than students who are 16 years old, there would be interesting to compare whether there's some biases due to the different age.

# In[44]:


df.AGE.astype(int).value_counts().sort_index().plot.bar(figsize=(21,7), width=.9, legend=False,
                                                                  cmap='Blues_r', rot=0)
plt.xlabel('Age')
plt.title('Student\'s age')
plt.show()


# Distribution of scores of students from 15 and 16 years old groups is distributed normally, and I think, there's no significant difference between these students.

# In[45]:


fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(21,7))
df[df.AGE < 16]['PV_MATH'].hist(ax=ax[0], alpha=.5, label='15 yers')
df[df.AGE > 15]['PV_MATH'].hist(ax=ax[0], alpha=.5, label='16 yers')
ax[0].set_xlabel('Math scores')
ax[0].legend()

df[df.AGE < 16]['PV_SCIE'].hist(ax=ax[1], alpha=.5, label='15 yers')
df[df.AGE > 15]['PV_SCIE'].hist(ax=ax[1], alpha=.5, label='16 yers')
ax[1].set_xlabel('Science scores')
ax[1].legend()

df[df.AGE < 16]['PV_READ'].hist(ax=ax[2], alpha=.5, label='15 yers')
df[df.AGE > 15]['PV_READ'].hist(ax=ax[2], alpha=.5, label='16 yers')
ax[2].set_xlabel('Reading scores')
ax[2].legend()

plt.suptitle('Distribution of grades by Age (year old)')
plt.show()


# In[46]:


df.groupby('ST20Q01').STIDSTD.nunique().sort_index().plot.bar(figsize=(21,7), width=.9, legend=False,
                                                                  cmap='Blues_r', rot=0)
plt.xlabel('Country')
plt.title('Whether a student passed the test in the country of birth or not')
plt.show()


# In[47]:


# Whether a student passed the test in the country of birth or not

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(21,7))
df[df.ST20Q01 == 'Country of test']['PV_MATH'].hist(ax=ax[0], alpha=.5, label='Country of test')
df[df.ST20Q01 == 'Other country']['PV_MATH'].hist(ax=ax[0], alpha=.5, label='Other country')
ax[0].set_xlabel('Math scores')
ax[0].legend()

df[df.ST20Q01 == 'Country of test']['PV_SCIE'].hist(ax=ax[1], alpha=.5, label='Country of test')
df[df.ST20Q01 == 'Other country']['PV_SCIE'].hist(ax=ax[1], alpha=.5, label='Other country')
ax[1].set_xlabel('Science scores')
ax[1].legend()

df[df.ST20Q01 == 'Country of test']['PV_READ'].hist(ax=ax[2], alpha=.5, label='Country of test')
df[df.ST20Q01 == 'Other country']['PV_READ'].hist(ax=ax[2], alpha=.5, label='Other country')
ax[2].set_xlabel('Reading scores')
ax[2].legend()

plt.suptitle('Distribution of grades by Age (year old)')
plt.show()


# An average international grade of students is 9.8 points, and on the same time, the mean grade compared to modal grade in country is equal -0.16 points.  
# Among all 64 countries represented in the dataset, students from Canada, Italy, Mexico, and Spain have the highest average international rate.

# In[48]:


# International grade and grade compared to modal grade in country

df[['ST01Q01', 'GRADE']].mean()


# In[49]:


group_means = df.groupby(['CNT']).mean()
group_order = group_means.sort_values(['ST01Q01'], ascending = False).index

g = sns.FacetGrid(data = df, col = 'CNT', col_wrap = 5, height = 2,
                 col_order = group_order)
g.map(plt.hist, 'ST01Q01', bins = np.arange(5, 15+1, 1))
g.set_titles('{col_name}')
plt.show()


# #### 3. What's a relationship between students performance and highest parental education measured in years as well as mother's and father's highest schooling?

# In[50]:


plt.subplots(figsize=(21,7))
sns.regplot(df['PARED'], df['PV_MATH'], fit_reg=True, x_jitter=0.1, y_jitter=0.1, scatter_kws={'alpha': 1/3})
plt.xlabel('Highest parental education in years')
plt.ylabel('Math scores')
plt.title('Relationship between math scores and highest parental education in years')
plt.show()


# There exist a positive weak relationship between highest parental education in years and students math scores. To check whether this relationship is significant, linear regression can be fitted to determine if increase in parental education affects increases students math scores.

# #### 4. Whether there exist a correlation between family wealth (measured in the number of telephones, computers, etc.) and students performance?

# In[51]:


# How many - computers (here we'll look at 2 extreme - no computer vs. 3 or more computers)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(21,7))
df[df.ST27Q03 == 'Zero']['PV_MATH'].hist(ax=ax[0], alpha=.5, label='Zero')
df[df.ST27Q03 == 'Three or more']['PV_MATH'].hist(ax=ax[0], alpha=.5, label='Three or more')
ax[0].set_xlabel('Math scores')
ax[0].legend()

df[df.ST27Q03 == 'Zero']['PV_SCIE'].hist(ax=ax[1], alpha=.5, label='Zero')
df[df.ST27Q03 == 'Three or more']['PV_SCIE'].hist(ax=ax[1], alpha=.5, label='Three or more')
ax[1].set_xlabel('Science scores')
ax[1].legend()

df[df.ST27Q03 == 'Zero']['PV_READ'].hist(ax=ax[2], alpha=.5, label='Zero')
df[df.ST27Q03 == 'Three or more']['PV_READ'].hist(ax=ax[2], alpha=.5, label='Three or more')
ax[2].set_xlabel('Reading scores')
ax[2].legend()

plt.suptitle('Distribution of grades of those who does and doesn\'t own computers')
plt.show()


# More than half of all students don't have a computer at all. Therefore, we can observe, that distribution of score of those students who doesn't have a computer is skewed to the right for two subjects - mathematics and science.

# In[52]:


# How many - cars (here we'll look at 2 extreme - no car vs. 3 or more cars)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(21,7))
df[df.ST27Q04 == 'Zero']['PV_MATH'].hist(ax=ax[0], alpha=.5, label='Zero')
df[df.ST27Q04 == 'Three or more']['PV_MATH'].hist(ax=ax[0], alpha=.5, label='Three or more')
ax[0].set_xlabel('Math scores')
ax[0].legend()

df[df.ST27Q04 == 'Zero']['PV_SCIE'].hist(ax=ax[1], alpha=.5, label='Zero')
df[df.ST27Q04 == 'Three or more']['PV_SCIE'].hist(ax=ax[1], alpha=.5, label='Three or more')
ax[1].set_xlabel('Science scores')
ax[1].legend()

df[df.ST27Q04 == 'Zero']['PV_READ'].hist(ax=ax[2], alpha=.5, label='Zero')
df[df.ST27Q04 == 'Three or more']['PV_READ'].hist(ax=ax[2], alpha=.5, label='Three or more')
ax[2].set_xlabel('Reading scores')
ax[2].legend()

plt.suptitle('Distribution of grades of those who does and doesn\'t own cars')
plt.show()


# A half of all students don't have a car in family. And we can observe, that distribution of score of those students who has no car in the family is skewed to the right for all 3 subjects - math, science, and reading.

# In[53]:


# How many - cellular phones (here we'll look at 2 extreme - no cellular phone vs. 3 or more cellular phones)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(21,7))
df[df.ST27Q01 == 'Zero']['PV_MATH'].hist(ax=ax[0], alpha=.5, label='Zero')
df[df.ST27Q01 == 'Three or more']['PV_MATH'].hist(ax=ax[0], alpha=.5, label='Three or more')
ax[0].set_xlabel('Math scores')
ax[0].legend()

df[df.ST27Q01 == 'Zero']['PV_SCIE'].hist(ax=ax[1], alpha=.5, label='Zero')
df[df.ST27Q01 == 'Three or more']['PV_SCIE'].hist(ax=ax[1], alpha=.5, label='Three or more')
ax[1].set_xlabel('Science scores')
ax[1].legend()

df[df.ST27Q01 == 'Zero']['PV_READ'].hist(ax=ax[2], alpha=.5, label='Zero')
df[df.ST27Q01 == 'Three or more']['PV_READ'].hist(ax=ax[2], alpha=.5, label='Three or more')
ax[2].set_xlabel('Reading scores')
ax[2].legend()

plt.suptitle('Distribution of grades of those who does and doesn\'t own cellular phones')
plt.show()


# In[54]:


df[df.ST27Q01 == 'Zero'].shape[0] / df.shape[0]


# Almost every student in the dataset has at least one cellular phone. And it is almost impossible to determine what the distribution of grades looks like for those students who do not have a cell phone, since the number of such guys in the dataset is very small (about 1.5%).

# #### 5. How do student possessions such as own room and desk, etc. affect his/her performance?

# In[55]:


# Possessions - own room

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(21,7))
df[df.ST26Q02 == 'Yes']['PV_MATH'].hist(ax=ax[0], alpha=.5, label='Own room', color='green')
df[df.ST26Q02 == 'No']['PV_MATH'].hist(ax=ax[0], alpha=.5, label='No room', color='black')
ax[0].set_xlabel('Math scores')
ax[0].legend()

df[df.ST26Q02 == 'Yes']['PV_SCIE'].hist(ax=ax[1], alpha=.5, label='Own room', color='green')
df[df.ST26Q02 == 'No']['PV_SCIE'].hist(ax=ax[1], alpha=.5, label='No room', color='black')
ax[1].set_xlabel('Science scores')
ax[1].legend()

df[df.ST26Q02 == 'Yes']['PV_READ'].hist(ax=ax[2], alpha=.5, label='Own room', color='green')
df[df.ST26Q02 == 'No']['PV_READ'].hist(ax=ax[2], alpha=.5, label='No room', color='black')
ax[2].set_xlabel('Reading scores')
ax[2].legend()

plt.suptitle('Distribution of grades of those who does and doesn\'t have own room')
plt.show()


# About a quater of students doesn't have their own rooms. This affects their preparation to the exam. And as the result, the distribution of math and science scores of those students who don't have their own room is skewed to the right 

# In[56]:


# Possessions - has desk

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(21,7))
df[df.ST26Q01 == 'Yes']['PV_MATH'].hist(ax=ax[0], alpha=.5, label='Has desk', color='green')
df[df.ST26Q01 == 'No']['PV_MATH'].hist(ax=ax[0], alpha=.5, label='No desk', color='black')
ax[0].set_xlabel('Math scores')
ax[0].legend()

df[df.ST26Q01 == 'Yes']['PV_SCIE'].hist(ax=ax[1], alpha=.5, label='Has room', color='green')
df[df.ST26Q01 == 'No']['PV_SCIE'].hist(ax=ax[1], alpha=.5, label='No desk', color='black')
ax[1].set_xlabel('Science scores')
ax[1].legend()

df[df.ST26Q01 == 'Yes']['PV_READ'].hist(ax=ax[2], alpha=.5, label='Has room', color='green')
df[df.ST26Q01 == 'No']['PV_READ'].hist(ax=ax[2], alpha=.5, label='No desk', color='black')
ax[2].set_xlabel('Reading scores')
ax[2].legend()

plt.suptitle('Distribution of grades of those who does and doesn\'t have desk')
plt.show()


# In[57]:


df[df.ST26Q01 == 'No'].shape[0] / df.shape[0]


# 11.1% of students don't have a desk, therefore, on average their math and science scores are lower than scores of students who has a table. Both, the distribution of reading scores of those who have and who doesn't have a desk ia normally distributed without any skewednes.

# In[58]:


# Possessions - has study place

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(21,7))
df[df.ST26Q03 == 'Yes']['PV_MATH'].hist(ax=ax[0], alpha=.5, label='Has study place', color='green')
df[df.ST26Q03 == 'No']['PV_MATH'].hist(ax=ax[0], alpha=.5, label='No study place', color='black')
ax[0].set_xlabel('Math scores')
ax[0].legend()

df[df.ST26Q03 == 'Yes']['PV_SCIE'].hist(ax=ax[1], alpha=.5, label='Has study place', color='green')
df[df.ST26Q03 == 'No']['PV_SCIE'].hist(ax=ax[1], alpha=.5, label='No study place', color='black')
ax[1].set_xlabel('Science scores')
ax[1].legend()

df[df.ST26Q03 == 'Yes']['PV_READ'].hist(ax=ax[2], alpha=.5, label='Has study place', color='green')
df[df.ST26Q03 == 'No']['PV_READ'].hist(ax=ax[2], alpha=.5, label='No study place', color='black')
ax[2].set_xlabel('Reading scores')
ax[2].legend()

plt.suptitle('Distribution of grades of those who does and doesn\'t have study place')
plt.show()


# It's really difficult to prepare to the assessment if you dont have study place at home. As a result the distribution of scores of those students who don't have a study place on average receive lower scores on math and science.

# In[59]:


# Possessions - has computer

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(21,7))
df[df.ST26Q04 == 'Yes']['PV_MATH'].hist(ax=ax[0], alpha=.5, label='Has computer', color='green')
df[df.ST26Q04 == 'No']['PV_MATH'].hist(ax=ax[0], alpha=.5, label='No computer', color='black')
ax[0].set_xlabel('Math scores')
ax[0].legend()

df[df.ST26Q04 == 'Yes']['PV_SCIE'].hist(ax=ax[1], alpha=.5, label='Has computer', color='green')
df[df.ST26Q04 == 'No']['PV_SCIE'].hist(ax=ax[1], alpha=.5, label='No computer', color='black')
ax[1].set_xlabel('Science scores')
ax[1].legend()

df[df.ST26Q04 == 'Yes']['PV_READ'].hist(ax=ax[2], alpha=.5, label='Has computer', color='green')
df[df.ST26Q04 == 'No']['PV_READ'].hist(ax=ax[2], alpha=.5, label='No computer', color='black')
ax[2].set_xlabel('Reading scores')
ax[2].legend()

plt.suptitle('Distribution of grades of those who does and doesn\'t have computer')
plt.show()


# Absense of computer significantly complicates the preparation not only for the exam, but also for the homework. Because for example, not all students have large-screen tablets or smartphones that can partially replace a computer. As a result, the distribution of math scores is significantly skewed to the right. Distributions of reading and science scores are also slightly skewed to theright.

# In[60]:


# Possessions - has software

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(21,7))
df[df.ST26Q05 == 'Yes']['PV_MATH'].hist(ax=ax[0], alpha=.5, label='Has software', color='green')
df[df.ST26Q05 == 'No']['PV_MATH'].hist(ax=ax[0], alpha=.5, label='No software', color='black')
ax[0].set_xlabel('Math scores')
ax[0].legend()

df[df.ST26Q05 == 'Yes']['PV_SCIE'].hist(ax=ax[1], alpha=.5, label='Has software', color='green')
df[df.ST26Q05 == 'No']['PV_SCIE'].hist(ax=ax[1], alpha=.5, label='No software', color='black')
ax[1].set_xlabel('Science scores')
ax[1].legend()

df[df.ST26Q05 == 'Yes']['PV_READ'].hist(ax=ax[2], alpha=.5, label='Has software', color='green')
df[df.ST26Q05 == 'No']['PV_READ'].hist(ax=ax[2], alpha=.5, label='No software', color='black')
ax[2].set_xlabel('Reading scores')
ax[2].legend()

plt.suptitle('Distribution of grades of those who does and doesn\'t have software')
plt.show()


# I can assume that the lack of software does not affect the distribution of grades in any way, since not all students pay money for software, thus, the lack of a computer worsens the average grade for the test more significantly.

# In[61]:


# Possessions - has Internet

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(21,7))
df[df.ST26Q06 == 'Yes']['PV_MATH'].hist(ax=ax[0], alpha=.5, label='Has Internet', color='green')
df[df.ST26Q06 == 'No']['PV_MATH'].hist(ax=ax[0], alpha=.5, label='No Internet', color='black')
ax[0].set_xlabel('Math scores')
ax[0].legend()

df[df.ST26Q06 == 'Yes']['PV_SCIE'].hist(ax=ax[1], alpha=.5, label='Has Internet', color='green')
df[df.ST26Q06 == 'No']['PV_SCIE'].hist(ax=ax[1], alpha=.5, label='No Internet', color='black')
ax[1].set_xlabel('Science scores')
ax[1].legend()

df[df.ST26Q06 == 'Yes']['PV_READ'].hist(ax=ax[2], alpha=.5, label='Has Internet', color='green')
df[df.ST26Q06 == 'No']['PV_READ'].hist(ax=ax[2], alpha=.5, label='No Internet', color='black')
ax[2].set_xlabel('Reading scores')
ax[2].legend()

plt.suptitle('Distribution of grades of those who does and doesn\'t have Internet')
plt.show()


# The lack of the Internet affects the distribution of grades in mathematics and science, since the process of obtaining the necessary information is either very slow or not at all. But it is worth noting that the presence of the Internet affects the distribution of reading grades in such a way that for those students who have the Internet, the opportunity to obtain additional information leads to a distortion of the distribution of grades to the left.

# Summing up, I would like to say that for a student who prepares and takes the exam, any help, whether it be a computer, the Internet, a place for preparation, or his own team, positively correlates with higher grades in both mathematics and science and reading.
# This is a very interesting study that can be done by collecting additional missing data and adding information from other sources, for example, information on income and / or expenses of students' families.

# #### 6. Whether total time learning and out of school lessons on math, science, and reading affect student performance?

# In[62]:


plt.subplots(figsize=(21,7))
plt.scatter(df['MMINS'], df['PV_MATH'], alpha=.1, cmap='Blues_r')
plt.xlabel('')
plt.ylabel('')
plt.title('')
plt.show()


# In[63]:


plt.subplots(figsize=(21,7))
plt.scatter(df['SMINS'], df['PV_SCIE'], alpha=.1, cmap='Blues_r')
plt.xlabel('')
plt.ylabel('')
plt.title('')
plt.show()


# In[64]:


plt.subplots(figsize=(21,7))
plt.scatter(df['LMINS'], df['PV_READ'], alpha=.1, cmap='Blues_r')
plt.xlabel('')
plt.ylabel('')
plt.title('')
plt.show()


# In[65]:


print(df[['MMINS', 'PV_MATH']].corr().iloc[0,1])
print(df[['SMINS', 'PV_SCIE']].corr().iloc[0,1])
print(df[['LMINS', 'PV_READ']].corr().iloc[0,1])


# Probably, the answer is obvious to the question whether there is a positive correlation between the number of hours of preparation for a particular subject and the grade for the exam after such preparation. But according to the schedules of preparing students for both mathematics and science and literature, one cannot say that there is a moderated relationship between this action and the result.  
# In order to understand whether there really is no relationship between the preparation time for the exam and the grade for it. I calculated the Pearson correlation coefficient. And indeed the highest correlation coefficient is 15%.

# <a id='conclusions'></a>
# ## Part III - Conclusions 

# 1. What is students’ performance at schools in different countries (including whether country is a OECD member)
# If we plot all the grades by subject, then scores in each subject looks normally distributed. Mean scores of science are about 3 points higher than average reading scores. In its turn, avearge reading scores are about 3 poits higher than avearge math scores. So scores in those 3 subjects are very similar.  
# So, let's look at their boxplots.  
# In general, there're outliers in every Series of scores. Moreover, math scores have approximately equal tails of outliers, but science and reading scores have outliers with lower scores more, than outliers with higher scores. let's go deepper, and look at students perormance in the context of countries, OECD membership, and other columns.  
# Except Liechtenstein which is on the 6th position, on average, students from Asia countries receive the highest scores on math. China, Singapore, Hong Kong, Taiwan, and Korea are in Top-5. Macao and Japan follow immediately behind Liechtenstein.  
# Chinese students receive on average 611 points. In comparison, in Peru average math scores are equal 368. This's 1.7 times less than in China.  
# Average science scores are less than math scores by about 6 points. And this is becoming noticeable for countries with the highest average scores in science. China, Gang Kong and Singapore are also in the Top-3 with an average score of 547 to 579. For China, this difference is 32 points or 5.2%.  
# For average reading scores, China, Hong Kong, Singapore, Japan, Korea and Taiwan continue to be the leaders with a maximum average of 569 points for China. This average score is the lowest for China in three subjects, possibly also because English is not a native language for a large population of the country.  
# The difference is noticeable in all three subjects at once: average scores in mathematics, science and reading are higher in OSCE countries than in non-OSCE countries. The difference is about 48 points for each subject.

# 2. What are the characteristics of students participated in PICA 2012:  
#   * gender:  
# Number of female students is little more (by 0.6%) than number of male students. Let's look at gender by countries.
# The largest number of students are in Mexico, Italy, Spain, Canada and Brazil. Except for Italy, the number of the females is greater than that of the male. In Brazil, there are 8% fewer males than females. The number of students in Mexico is 1.8 times higher than in Brazil, which is in 5th place, the number of males is 2.8 times less and the number of females is 2.7.  
# Distriution of males and females math and science score are distributed approximately normal. However, there's slight difference of reading scores: female have aslightly higher grades than males.  
#   * age:  
# Since student age is between 15 and 16 year old, and number of students who are 15 years old are twice larger than students who are 16 years old, there would be interesting to compare whether there's some biases due to the different age.  
# Distribution of scores of students from 15 and 16 years old groups is distributed normally, and I think, there's no significant difference between these students.  
#   * international grade and grade compared to modal grade in country:  
# An average international grade of students is 9.8 points, and on the same time, the mean grade compared to modal grade in country is equal -0.16 points.    
# Among all 64 countries represented in the dataset, students from Canada, Italy, Mexico, and Spain have the highest average international rate.

# 3. What's a relationship between students performance and highest parental education measured in years as well as mother's and father's highest schooling?  
# There exist a positive weak relationship between highest parental education in years and students math scores. To check whether this relationship is significant, linear regression can be fitted to determine if increase in parental education affects increases students math scores.

# 4. Whether there exist a correlation between family wealth (measured in the number of telephones, computers, etc.) and students performance?  
# More than half of all students don't have a computer at all. Therefore, we can observe, that distribution of score of those students who doesn't have a computer is skewed to the right for two subjects - mathematics and science.  
# A half of all students don't have a car in family. And we can observe, that distribution of score of those students who has no car in the family is skewed to the right for all 3 subjects - math, science, and reading.  
# Almost every student in the dataset has at least one cellular phone. And it is almost impossible to determine what the distribution of grades looks like for those students who do not have a cell phone, since the number of such guys in the dataset is very small (about 1.5%).

# 5. How do student possessions such as own room and desk, etc. affect his/her performance?  
# About a quater of students doesn't have their own rooms. This affects their preparation to the exam. And as the result, the distribution of math and science scores of those students who don't have their own room is skewed to the right   
# 11.1% of students don't have a desk, therefore, on average their math and science scores are lower than scores of students who has a table. Both, the distribution of reading scores of those who have and who doesn't have a desk ia normally distributed without any skewednes.  
# It's really difficult to prepare to the assessment if you dont have study place at home. As a result the distribution of scores of those students who don't have a study place on average receive lower scores on math and science.  
# Absense of computer significantly complicates the preparation not only for the exam, but also for the homework.   Because for example, not all students have large-screen tablets or smartphones that can partially replace a computer.   As a result, the distribution of math scores is significantly skewed to the right. Distributions of reading and science scores are also slightly skewed to theright.  
# I can assume that the lack of software does not affect the distribution of grades in any way, since not all students pay money for software, thus, the lack of a computer worsens the average grade for the test more significantly.  
#   
# Summing up, I would like to say that for a student who prepares and takes the exam, any help, whether it be a computer, the Internet, a place for preparation, or his own team, positively correlates with higher grades in both mathematics and science and reading.  
# This is a very interesting study that can be done by collecting additional missing data and adding information from other sources, for example, information on income and / or expenses of students' families.

# 6. Whether total time learning and out of school lessons on math, science, and reading affect student performance?  
# Probably, the answer is obvious to the question whether there is a positive correlation between the number of hours of preparation for a particular subject and the grade for the exam after such preparation. But according to the schedules of preparing students for both mathematics and science and literature, one cannot say that there is a moderated relationship between this action and the result.  
# In order to understand whether there really is no relationship between the preparation time for the exam and the grade for it. I calculated the coefficient of Pearson's correlation. And indeed the highest correlation coefficient is 15%.

# In[ ]:


get_ipython().system(' jupyter nbconvert *.ipynb --to slides --post serve --template output_toggle.tpl')


# In[ ]:




