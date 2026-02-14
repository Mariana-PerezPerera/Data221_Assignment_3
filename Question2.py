#importing pandas for dataframes and matplotlib to create the histogram:
import pandas as pd
import matplotlib.pyplot as plt

#loading in the data set and assigning it to a data_frame variable:
data_frame = pd.read_csv("crime1.csv")

#focussing on the specific column (violent crimes per pop.):
violent_crimes_per_population = data_frame["ViolentCrimesPerPop"]

bins_width = 18

#now we have to create the histogram:
plt.figure()
#assinging the data and making it a histogram:
plt.hist(violent_crimes_per_population, bins=bins_width)
#giving it a title:
plt.title("Distribution of Violent Crimes Per Population")
#labeling the x and y-axis:
plt.xlabel("The Violent Crimes Per Population")
plt.ylabel("Frequency")
#making/showing the histogram:
plt.show()

#now we will create a box plot:
plt.figure()
#assinging the data and making it a boxplot:
plt.boxplot(violent_crimes_per_population)
#giving it a title:
plt.title("Distribution of The Violent Crimes Per Population")
#labeling the x and y-axis:
plt.xlabel("The Violent Crimes Per Population")
plt.ylabel("Value")
#creating/showing the boxplot:
plt.show()

#==========================================================================
#Explanation:
'''
Histogram:
The histogram is using a bin width of 18 and it ultimately shows that most violent crime values
tend to be more concentrated in the lower-middle area. Moreover, there seems to be fewer near the higher end
of the scale. Therefore, we can say that the data is more right-skewed and this means that some communities 
have much higher crime rates than most communities. 
Box Plot:
The box plot depicts the median of the data which is represented by the orange line inside the box, 
which signifies the middle value of the given crime dataset. The median is depicted below the center area of the box as well. 
The overall length of the box is representing the interquartile range which ultimately shows where the middle 50% of the 
values from the dataset are. Any points lie outside of the box are known as outliers. The boxplot shows outliers, meaning that few 
communities either experience high violent crime rates or low violent crime rates compared to the majority.
'''

