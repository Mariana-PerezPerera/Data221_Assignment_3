import pandas as pd

#now we have load the data set
data_frame = pd.read_csv("crime1.csv")

#Nowe we have to focus on the violent crimes per pop column
violent_crimes_per_population = data_frame["ViolentCrimesPerPop"]

#computing the statistical measures: (mean, median, standard deviation, minimum value and maximum value:
mean_value = violent_crimes_per_population.mean()  #computes the mean
#now well print the mean:
print("Mean:", mean_value)

median_value = violent_crimes_per_population.median() #computes the median
#now well print the median:
print("Median:", median_value)

standard_deviation_value = violent_crimes_per_population.std()  #computes the standard deviation
#now well print the standard deviation:
print("Standard Deviation:", standard_deviation_value)

minimum_value = violent_crimes_per_population.min()  #computes the minimum
#now well print the minimum:
print("Minimum:", minimum_value)

maximum_value = violent_crimes_per_population.max()  #computes the maximum
#now well print the maximum:
print("Maximum:", maximum_value)

#---------------------------------------------------------------------
#Explanation:
'''
The mean is bigger than the median meaning that the distribution is most likely right-skewed. 
This means that there will be extreme values that ultimately pull the mean upward. Therefore, since there are
extreme values the statistic that is more affected would be the mean. The mean would be more affected than the median
because when the mean is being computed it is using all numerical values and therefore, extreme values (really big or 
really small) values will drastically pull the mean up or down. Since the median is simply just the middle position of 
the data is ultimately more resistant to any possible outliers. 
'''

