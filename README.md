# Communicate Data Findings  
  
  For this project I analyzed the results of The OECD Programme for International Student Assessment (PISA) as a part of completing Data Visualization course.   
  
From OECD website:  
_PISA is an international study that was launched by the OECD in 1997, first administered in 2000 and now covers over 80 countries. Every 3 years the PISA survey provides comparative data on 15-year-olds’ performance in reading, mathematics, and science. In addition, each cycle explores a distinct “innovative domain” such as Collaborative Problem Solving (PISA 2015) and Global Competence (PISA 2018). The results have informed education policy discussions at the national and global level since its inception._  
https://www.oecd.org/pisa/aboutpisa/pisa-based-test-for-schools-faq.htm    
  
__The PISA goals are:__    
- Empower school leaders and teachers by providing them with evidence-based analysis of their students’ performance.    
- Measure students’ knowledge, skills and competencies that will equip them for success in education and the world of work.   
- Provide valuable information on the learning climate within a school, students’ socioeconomic background and motivation for learning.   
- Help schools measure a wider range of 21st century skills beyond maths, reading and science.   
- Provide opportunities for global peer-learning among teachers and school leaders.   
  
__Based on the objectives of the PISA, using the data, the following questions can be answered:__    
1. What is students’ performance at schools in different countries (including whether country is a OECD member).    
2. What are the characteristics of students participated in PICA 2012:    
    * gender,   
    * age,    
    * whether a student passed the test in the country of birth or not,   
    * international grade and grade compared to modal grade in country.   
3. What's a relationship between students performance and highest parental education measured in years as well as mother's and father's highest schooling?   
4. Whether there exist a correlation between family wealth (measured in the number of telephones, computers, etc.) and students performance?   
5. How do student possessions such as own room and desk, etc. affect his/her performance?   
6. Last but not least, whether total time learning and out of school lessons on math, science, and reading affect student performance?  
   
__Installations:__   
In this project Python 3.x and the following Python libraries were installed:  
 
Os https://docs.python.org/3/library/os.html  
Pandas https://pandas.pydata.org/   
Numpy https://numpy.org/   
Seaborn https://seaborn.pydata.org/   
Matplotlib.pyplot https://matplotlib.org/api/_as_gen/matplotlib.pyplot.html   
  
__Data:__  
  
PISA 2012 assessment results on Mathematics, Reading, Science, Problem Solving and Financial Literacy - pisa2012.csv   
Explanation of columns in PISA 2012 dataset - pisadict2012.csv  
  
__There are 4 parts in this project:__   
  
Introduction   
Part I - Data Wrangling and Data Exploration  
Part II - Explanatory Data Analysis   
Part III - Conclusions   


__Project Details (Steps I needed to do, from Udacity)__  
  
This project is divided into two major parts. In the first part, you will conduct an exploratory data analysis on a dataset of your choosing. You will use Python data science and data visualization libraries to explore the dataset’s variables and understand the data’s structure, oddities, patterns and relationships. The analysis in this part should be structured, going from simple univariate relationships up through multivariate relationships, but it does not need to be clean or perfect. There is no one single answer that needs to come out of a given dataset. This part of the project is your opportunity to ask questions of the data and make your own discoveries. It’s important to keep in mind that sometimes exploration can lead to dead ends, and that it can take multiple steps to dig down to what you’re truly looking for. Be patient with your steps, document your work carefully, and be thorough in the perspective that you choose to take with your dataset.  
  
In the second part, you will take your main findings from your exploration and convey them to others through an explanatory analysis. To this end, you will create a slide deck that leverages polished, explanatory visualizations to communicate your results. This part of the project should make heavy use of the first part of the project. Select one or two major paths in your exploration, choose relevant visualizations along that path, and then polish them to construct a story for your readers to understand what you found.  
 
___Step 1.1: Choose your Dataset___  
  
___Step 1.2: Explore Your Data___  
It’s time to get to the interesting bits. Explore your data and document your findings in a report. The report should briefly introduce the dataset, then systematically walk through the points of exploration that you conducted. You should have headers and text that organize your thoughts and findings. Visualizations in this part of the project need not be completely polished: this is just your own exploration at this point. However, you should still make sure that you adhere to principles of using appropriate plot types and encodings so that accurate conclusions can be drawn, and that you have enough comments and labeling so that when you return to your work, you can quickly grasp your analysis steps.  
  
___Step 2.1: Document your Story___  
At the end of your exploration, you probably have a bunch of things that you’ve discovered. Now it’s time to organize your findings and select a story that you will convey to others. In your readme document, you should summarize your main findings and reflect on the steps you took in your data exploration. You should also lay out the key insights that you want to convey in your explanatory report as well as any changes to visualizations, or note new visualizations that will be created to bridge between your insights.  
  
___Step 2.2: Create your Slide Deck___  
Follow the plans you laid out in the previous step and create a slide deck with explanatory data visualizations to tell a story about the data you explored. You can start with code that you used in your exploration, but you should make sure that the code is revised so that your plots are polished. Make sure that you also pay attention to aspects of design integrity in your revisions.  

__Conclusion:__  
1. What is students’ performance at schools in different countries (including whether country is a OECD member) If we plot all the grades by subject, then scores in each subject looks normally distributed. Mean scores of science are about 3 points higher than average reading scores. In its turn, avearge reading scores are about 3 poits higher than avearge math scores. So scores in those 3 subjects are very similar.  
In general, there're outliers in every Series of scores. Moreover, math scores have approximately equal tails of outliers, but science and reading scores have outliers with lower scores more, than outliers with higher scores. let's go deepper, and look at students perormance in the context of countries, OECD membership, and other columns.  
Except Liechtenstein which is on the 6th position, on average, students from Asia countries receive the highest scores on math. China, Singapore, Hong Kong, Taiwan, and Korea are in Top-5. Macao and Japan follow immediately behind Liechtenstein.  
Chinese students receive on average 611 points. In comparison, in Peru average math scores are equal 368. This's 1.7 times less than in China.  
Average science scores are less than math scores by about 6 points. And this is becoming noticeable for countries with the highest average scores in science. China, Gang Kong and Singapore are also in the Top-3 with an average score of 547 to 579. For China, this difference is 32 points or 5.2%.  
For average reading scores, China, Hong Kong, Singapore, Japan, Korea and Taiwan continue to be the leaders with a maximum average of 569 points for China. This average score is the lowest for China in three subjects, possibly also because English is not a native language for a large population of the country.  
The difference is noticeable in all three subjects at once: average scores in mathematics, science and reading are higher in OSCE countries than in non-OSCE countries. The difference is about 48 points for each subject.  
  
2. What are the characteristics of students participated in PICA 2012:  
- gender:  
Number of female students is little more (by 0.6%) than number of male students. Let's look at gender by countries. The largest number of students are in Mexico, Italy, Spain, Canada and Brazil. Except for Italy, the number of the females is greater than that of the male. In Brazil, there are 8% fewer males than females. The number of students in Mexico is 1.8 times higher than in Brazil, which is in 5th place, the number of males is 2.8 times less and the number of females is 2.7.  
Distriution of males and females math and science score are distributed approximately normal. However, there's slight difference of reading scores: female have aslightly higher grades than males.  
- age:  
Since student age is between 15 and 16 year old, and number of students who are 15 years old are twice larger than students who are 16 years old, there would be interesting to compare whether there's some biases due to the different age.  
Distribution of scores of students from 15 and 16 years old groups is distributed normally, and I think, there's no significant difference between these students.  
- international grade and grade compared to modal grade in country:  
An average international grade of students is 9.8 points, and on the same time, the mean grade compared to modal grade in country is equal -0.16 points.  
Among all 64 countries represented in the dataset, students from Canada, Italy, Mexico, and Spain have the highest average international rate.  
  
3. What's a relationship between students performance and highest parental education measured in years as well as mother's and father's highest schooling?  
There exist a positive weak relationship between highest parental education in years and students math scores. To check whether this relationship is significant, linear regression can be fitted to determine if increase in parental education affects increases students math scores.  
  
4. Whether there exist a correlation between family wealth (measured in the number of telephones, computers, etc.) and students performance?  
More than half of all students don't have a computer at all. Therefore, we can observe, that distribution of score of those students who doesn't have a computer is skewed to the right for two subjects - mathematics and science.  
A half of all students don't have a car in family. And we can observe, that distribution of score of those students who has no car in the family is skewed to the right for all 3 subjects - math, science, and reading.  
Almost every student in the dataset has at least one cellular phone. And it is almost impossible to determine what the distribution of grades looks like for those students who do not have a cell phone, since the number of such guys in the dataset is very small (about 1.5%).   
  
 5. How do student possessions such as own room and desk, etc. affect his/her performance?  
About a quater of students doesn't have their own rooms. This affects their preparation to the exam. And as the result, the distribution of math and science scores of those students who don't have their own room is skewed to the right.  
11.1% of students don't have a desk, therefore, on average their math and science scores are lower than scores of students who has a table. Both, the distribution of reading scores of those who have and who doesn't have a desk ia normally distributed without any skewednes.  
It's really difficult to prepare to the assessment if you dont have study place at home. As a result the distribution of scores of those students who don't have a study place on average receive lower scores on math and science.  
Absense of computer significantly complicates the preparation not only for the exam, but also for the homework. Because for example, not all students have large-screen tablets or smartphones that can partially replace a computer. As a result, the distribution of math scores is significantly skewed to the right. Distributions of reading and science scores are also slightly skewed to theright.  
I can assume that the lack of software does not affect the distribution of grades in any way, since not all students pay money for software, thus, the lack of a computer worsens the average grade for the test more significantly.  
Summing up, I would like to say that for a student who prepares and takes the exam, any help, whether it be a computer, the Internet, a place for preparation, or his own team, positively correlates with higher grades in both mathematics and science and reading.  
This is a very interesting study that can be done by collecting additional missing data and adding information from other sources, for example, information on income and / or expenses of students' families.  
  
6. Whether total time learning and out of school lessons on math, science, and reading affect student performance?  
Probably, the answer is obvious to the question whether there is a positive correlation between the number of hours of preparation for a particular subject and the grade for the exam after such preparation. But according to the schedules of preparing students for both mathematics and science and literature, one cannot say that there is a moderated relationship between this action and the result.  
In order to understand whether there really is no relationship between the preparation time for the exam and the grade for it. I calculated the coefficient of Pearson's correlation. And indeed the highest correlation coefficient is 15%.  
