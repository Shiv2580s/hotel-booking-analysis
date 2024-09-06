Import Libraries

#Importing necessary libraries needed in EDA
import numpy as np
import pandas as pd

# for visualisation
import seaborn as sns
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

Dataset Loading
# Load Dataset
hotel_df= pd.read_csv('/content/drive/MyDrive/Almab/Hotel Bookings.csv')

Dataset First View
# Dataset First Look
hotel_df


# Dataset Rows & Columns count
hotel_df.shape

# Dataset Info
hotel_df.info()


# Dataset Duplicate Value Count
hotel_df.duplicated().sum()

# Missing Values/Null Values Count
hotel_df.isnull().sum()


# Visualizing the missing values using seaborn heatmap
# Setting the figure size to 20x8 inches
plt.figure(figsize=(20, 8))

# Creating a heatmap to visualize missing data in the dataframe
# hb_df.isna().transpose() - Transposes the DataFrame and checks for missing values
# cmap="YlGnBu" - Sets the colormap to 'Yellow-Green-Blue'
# cbar_kws={'label': 'Missing Data'} - Adds a label to the color bar
sns.heatmap(hotel_df.isna().transpose(), cmap="YlGnBu", cbar_kws={'label': 'Missing Data'})

# Adding a title to the heatmap with a font size of 18
plt.title('Missing Values', fontsize=18)

# Displaying the plot
plt.show()


# Dataset Columns
hotel_df.columns

# Dataset Describe
hotel_df.describe()

# Check Unique Values for each variable
pd.Series({col:hotel_df[col].unique() for col in hotel_df })


3. Data Wrangling
# Write your code to make your dataset analysis ready.
hotel_df1 = hotel_df.copy()
hotel_df1
hotel_df1.columns
hotel_df1.head()

# replacing null values in children column with 0 assuming that family had 0 children

hotel_df1['children' ].fillna(0, inplace = True)

# replacing null values in company and agent columns with 0 assuming those rooms were booked without company/agent

hotel_df1['company' ].fillna(0, inplace = True)
hotel_df1['agent' ].fillna(0, inplace = True)

# replacing null values in country column as 'Others'

hotel_df1['country'].fillna('Others', inplace = True)

# checking for null values after replacing them
hotel_df1.isnull().sum()

# dropping rows where no adults , children and babies are available because no bookings were made that day

no_guest=hotel_df1[hotel_df1['adults']+hotel_df1['babies']+hotel_df1['children']==0]
hotel_df1.drop(no_guest.index, inplace=True)

# adding some new columns to make our data analysis ready

# creating total people column by adding all the people in that booking

hotel_df1['total_people'] = hotel_df1['adults'] + hotel_df1['babies'] + hotel_df1['children']

# creating a column to check total stay by people in that booking
hotel_df1['total_stay'] = hotel_df1['stays_in_weekend_nights'] + hotel_df1['stays_in_week_nights']

hotel_df1

# checking the unique values which is to be analysed
pd.Series({col:hotel_df1[col].unique() for col in hotel_df1})



Chart - 1
Which type of hotel is most preffered by the guests?
# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Assuming hotel_df1 is your DataFrame containing hotel booking data

# Count the number of bookings for each hotel type (e.g., 'City Hotel' and 'Resort Hotel')
hotel_count = hotel_df1['hotel'].value_counts()

# Explanation: The value_counts() method is used here to count the occurrences of each unique value in the 'hotel' column.
# This will give us a Series where the index is the hotel type and the values are the counts of bookings.

# Plotting the counts in a pie chart
# figsize=(9,7) sets the size of the figure
# autopct='%1.2f%%' formats the percentages on the pie chart
# shadow=True adds a shadow to the pie chart
# fontsize=15 sets the font size of the text on the pie chart
# startangle=50 rotates the start of the pie chart by 50 degrees
hotel_count.plot.pie(figsize=(9,7), autopct='%1.2f%%', shadow=True, fontsize=15, startangle=50)

# Explanation: The plot.pie() method is used to create a pie chart from the Series.
# figsize determines the size of the chart.
# autopct='%1.2f%%' displays the percentage value on the pie slices with 2 decimal places.
# shadow=True adds a shadow effect to the chart for better visual appeal.
# fontsize adjusts the size of the text labels.
# startangle rotates the start of the pie chart to the specified angle for better visualization.

# Setting the title of the chart
plt.title('Hotel Booking Percentage', fontsize=20)

# Explanation: The plt.title() function sets the title of the pie chart.
# Here, 'Hotel Booking Percentage' is the title text and fontsize=18 sets the font size of the title.

# Ensuring the pie chart is a circle (equal aspect ratio)
plt.axis('equal')

# Explanation: The plt.axis('equal') function ensures that the pie chart is drawn as a circle.
# Without this, the pie chart might appear as an ellipse depending on the aspect ratio of the plot.

# Displaying the pie chart
plt.show()

# Explanation: The plt.show() function displays the pie chart.




Chart - 2
What is perecentage of hotel booking cancellation?

# Chart - 2 visualization code

import matplotlib.pyplot as plt

# Extracting and storing unique values of hotel cancellation status
# This counts the occurrences of cancellation (1) and non-cancellation (0) in the 'is_canceled' column
cancelled_hotel = hotel_df1.is_canceled.value_counts()

# Creating a pie chart to visualize the cancellation status
# 'figsize' sets the size of the figure
# 'explode' creates a small gap between the pie slices for better visualization
# 'autopct' adds the percentage of each slice on the pie chart
# 'shadow' adds a shadow to the pie chart for better visual effect
# 'fontsize' sets the font size of the text in the chart
# 'startangle' rotates the start of the pie chart for a better angle
cancelled_hotel.plot.pie(figsize=(9,7), explode=(0.05,0.05), autopct='%1.2f%%', shadow=True, fontsize=15, startangle=50)

# Adding a title to the pie chart
plt.title('Percentage of Hotel Cancellation and Non-Cancellation')

# Ensuring the pie chart is drawn as a circle
plt.axis('equal')

# Displaying the pie chart
plt.show()


Chart - 3
Which type of meal is most preffered by guests?

# Chart - 3 visualization code

# Counting the frequency of each meal type in the dataset
meal_count = hotel_df1.meal.value_counts()

# Extracting unique meal types and storing them in a variable
meal_name = hotel_df1['meal'].unique()

# Creating a DataFrame to store meal types and their corresponding counts
meal_df = pd.DataFrame(zip(meal_name, meal_count), columns=['Meal Name', 'Meal Count'])

# Visualizing the meal count data using a bar chart
plt.figure(figsize=(15, 5))  # Setting the size of the plot
g = sns.barplot(data=meal_df, x='Meal Name', y='Meal Count')  # Creating the bar plot

# Adding title to the plot
plt.title('Most Preferred Meal Types', fontsize=25)

# Displaying the plot
plt.show()

Chart - 4
Which year has the most bookings ?
# Chart - 4 visualization code

# Import necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Set the figure size to make the plot larger and more readable
plt.figure(figsize=(10, 4))

# Create a count plot to visualize the number of bookings across different years
# 'x' specifies the data for the x-axis, which is the arrival year
# 'hue' adds a layer of grouping based on hotel type (City Hotel or Resort Hotel)
sns.countplot(x=hotel_df1['arrival_date_year'], hue=hotel_df1['hotel'])

# Add a title to the plot with a larger font size for better visibility
plt.title("Number of bookings across year", fontsize=25)

# Display the plot
plt.show()


Chart - 5
Which month has the most bookings in each hotel type?
# Chart - 5 visualization code

# Importing necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Setting the figure size for the plot
plt.figure(figsize=(15,5))

# Creating a count plot to visualize the number of bookings across months
# Using seaborn's countplot function, we specify the x-axis as arrival_date_month and hue as hotel
# This plots the number of bookings for each month, differentiated by hotel type
sns.countplot(x=hotel_df1['arrival_date_month'], hue=hotel_df1['hotel'])

# Adding a title to the plot
plt.title("Number of bookings across months", fontsize=25)

# Displaying the plot
plt.show()



Chart - 6
From which country most guests come?

# Chart - 6 visualization code
# Counting the number of guests from various countries and storing the result in a DataFrame
country_df = hotel_df1['country'].value_counts().reset_index()[:10]

# Visualizing the values on a bar chart
# Setting the size of the graph
plt.figure(figsize=(15,4))

# Creating a bar plot using seaborn
sns.barplot(x=country_df['country'], y=country_df['count'])

# Adding a title to the chart
plt.title('Number of guests from each country', fontsize=20)

# Displaying the chart
plt.show()

dist_df = hotel_df1['distribution_channel'].value_counts().reset_index()
dist_df

Chart - 7
Which distribution channel is most used in booking?

# Chart - 7 visualization code
# Importing necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Creating a dataset of distribution channel name and count
dist_df = hotel_df1['distribution_channel'].value_counts().reset_index()

# Renaming the columns to appropriate names for better understanding
dist_df = dist_df.rename(columns={'distribution_channel': 'Channel name', 'count': 'channel count'})

# Creating an explode data to slightly separate each slice of the pie chart for better visibility
my_explode = (0.05, 0.05, 0.05, 0.05, 0.05)

# Adding a percentage column to the distribution channel dataframe to show the percentage of each channel
dist_df['percentage'] = round(dist_df['channel count'] * 100 / hotel_df1.shape[0], 1)

# Deciding the figure size of the pie chart
plt.figure(figsize=(15, 6))

# Plotting the pie chart with the distribution channel counts
plt.pie(dist_df['channel count'], labels=None, explode=my_explode, startangle=50)

# Adding legends with percentage using list comprehension
labels = [f'{l}, {s}%' for l, s in zip(dist_df['Channel name'], dist_df['percentage'])]
plt.legend(bbox_to_anchor=(0.85, 1), loc='upper left', labels=labels)

# Setting the title of the pie chart
plt.title('Most Used Booking Distribution Channels by Guests', fontsize=16)

# Ensuring the pie chart is drawn as a circle
plt.axis('equal')

# Displaying the pie chart
plt.show()


Chart - 8
Which room type is most preffered by guests?

# Import necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Step 1: Setting the size of the figure (chart) to make it more readable
plt.figure(figsize=(15, 5))

# Step 2: Plotting a count plot using seaborn
# - x specifies the data to be plotted along the x-axis, which is the 'reserved_room_type' column from hotel_df1
# - order specifies the order of the bars based on the count of each room type
sns.countplot(x=hotel_df1['reserved_room_type'], order=hotel_df1['reserved_room_type'].value_counts().index)

# Step 3: Adding a title to the chart to explain what it represents
plt.title('Preferred Room Type by Guests', fontsize=20)

# Step 4: Displaying the chart
plt.show()

Chart - 9
Which room type is most assigned?
# Chart - 9 visualization code

# Import necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Set the size of the figure for the chart
plt.figure(figsize=(15, 5))

# Create a count plot to visualize the number of times each room type was assigned to guests
# The 'order' parameter ensures that the room types are displayed in descending order of their counts
sns.countplot(x=hotel_df1['assigned_room_type'], order=hotel_df1['assigned_room_type'].value_counts().index)

# Add a title to the chart with a font size of 20
plt.title('Assigned Room Type to Guests', fontsize=20)

# Display the chart
plt.show()

Chart - 10
Top 5 agents in terms of most bookings?

# Chart - 10 visualization code
# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Group the dataset by the 'agent' column and count the number of bookings for each agent
agents = hotel_df1.groupby(['agent'])['agent'].agg({'count'}).reset_index().rename(columns={'count': 'Booking Count'}).sort_values(by='Booking Count', ascending=False)

# Step 2: Extract the top 5 agents based on the booking count
top_5 = agents[:5]

# Step 3: Define the explode values to slightly separate each slice of the pie chart for better visibility
explode = (0.02, 0.02, 0.02, 0.02, 0.02)

# Step 4: Define the colors to be used for each slice of the pie chart
colors = ("orange", "cyan", "brown", "indigo", "beige")

# Step 5: Define wedge properties for the pie chart slices
wp = {'linewidth': 1, 'edgecolor': "green"}

# Step 6: Define a function to format the percentage and absolute booking count on the pie chart
def func(pct, allvalues):
    absolute = int(pct / 100. * np.sum(allvalues))
    return "{:.1f}%\n({:d})".format(pct, absolute)

# Step 7: Create a pie chart to visualize the top 5 agents' booking counts
fig, ax = plt.subplots(figsize=(15, 7))
wedges, texts, autotexts = ax.pie(top_5['Booking Count'],
                                  autopct=lambda pct: func(pct, top_5['Booking Count']),
                                  explode=explode,
                                  shadow=False,
                                  colors=colors,
                                  startangle=50,
                                  wedgeprops=wp)

# Step 8: Add a legend to the pie chart
ax.legend(wedges, top_5['agent'],
          title="Agents",
          loc="upper left",
          bbox_to_anchor=(1, 0, 0.5, 1))

# Step 9: Customize the appearance of the text in the pie chart
plt.setp(autotexts, size=8, weight="bold")

# Step 10: Set the title of the pie chart
ax.set_title("Top 5 Agents in Terms of Booking", fontsize=17)

# Step 11: Ensure the pie chart is drawn as a circle
plt.axis('equal')

# Step 12: Display the pie chart
plt.show()

Chart - 11
What is the percentage of repeated guests?

# Chart - 11 visualization code

# Counting the number of repeated and non-repeated guests
rep_guests = hotel_df1['is_repeated_guest'].value_counts()

# Plotting the values in a pie chart
rep_guests.plot.pie(autopct='%1.2f%%', explode=(0.00, 0.09), figsize=(15, 6), shadow=False)

# Adding a title to the pie chart
plt.title('Percentage of Repeated Guests', fontsize=20)

# Ensuring the pie chart is drawn as a circle
plt.axis('equal')

# Displaying the pie chart
plt.show()


Chart - 12
Which customer type has the most booking?
# Chart - 12 visualization code

# Counting the number of bookings for each customer type
cust_type = hotel_df1['customer_type'].value_counts()

# Plotting the values in a line chart
cust_type.plot(figsize=(15,5))

# Setting the x-axis label
plt.xlabel('Count', fontsize=8)

# Setting the y-axis label
plt.ylabel('Customer Type', fontsize=10)

# Setting the title of the chart
plt.title('Customer Type and their booking count', fontsize=20)

# Displaying the chart
plt.show()


Chart - 13
How long people stay in the hotel?

# Chart - 13 visualization code

# Creating a DataFrame containing only non-cancelled bookings
not_cancelled_df = hotel_df1[hotel_df1['is_canceled'] == 0]

# Creating a DataFrame for hotel stays with a maximum of 15 days
hotel_stay = not_cancelled_df[not_cancelled_df['total_stay'] <= 15]

# Setting the size of the plot
plt.figure(figsize=(15, 5))

# Creating a count plot to visualize the total number of stays in each hotel for stays up to 15 days
sns.countplot(x=hotel_stay['total_stay'], hue=hotel_stay['hotel'])

# Adding a title to the chart
plt.title('Total number of stays in each hotel (Up to 15 days)', fontsize=20)

# Adding labels to the x and y axes
plt.xlabel('Total stay')
plt.ylabel('Count of days')

# Displaying the chart
plt.show()


Chart - 14 - Correlation Heatmap
# Correlation Heatmap visualization code

# Importing necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Setting the chart size
plt.figure(figsize=(15,10))

# Creating a heatmap to visualize the correlation between columns
# Using the corr() function to calculate the correlation matrix
# Setting numeric_only=True to include only numeric columns to avoid warnings
# annot=True to display the correlation values on the heatmap
sns.heatmap(hotel_df1.corr(numeric_only=True), annot=True)

# Setting the title of the heatmap
plt.title('Correlation of the columns', fontsize=20)

# Displaying the heatmap
plt.show()


Chart - 15 - Pair Plot
# Pair Plot visualization code

sns.pairplot(hotel_df1)
plt.show()


Conclusion
To achieve the business objective of increasing bookings and retaining customers, I recommend implementing dynamic pricing and introducing attractive offers and packages to entice new customers. A loyalty points program should be introduced to incentivize repeat bookings, allowing customers to redeem points for discounts or special perks on their next stays.

Additionally, providing amenities such as ample parking spaces, designated kids' areas, and complimentary internet access can enhance the overall guest experience and attract more bookings. These efforts not only help in increasing customer satisfaction but also contribute to building long-term relationships with guests, ultimately leading to positive business growth.

