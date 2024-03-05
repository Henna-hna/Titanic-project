
DAY-2
"""

import numpy as np

import pandas as pd
from pandas import Series, DataFrame

titanic_df = pd.read_csv("/content/train.csv")
titanic_df.head()

#for visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
# %matplotlib inline

"""So we can see that the Cabin column has information on the deck, but it has several NaN values, so we'll have to drop them.

"""

# First we'll drop the NaN values and create a new object, deck
deck = titanic_df['Cabin'].dropna()

# Quick preview of the decks
deck.head()

"""Notice we only need the first letter of the deck to classify its level (e.g. A,B,C,D,E,F,G)"""

# So let's grab that letter for the deck level with a simple for loop

#set an empty set:
levels = []

#for loop to grab first letter:
for level in deck:
  levels.append(level[0])

# Reset DataFrame and use factor plot
cabin_df = DataFrame(levels)
cabin_df.columns = ['Cabin']
sns.catplot(data=cabin_df, x='Cabin', palette='winter_d', kind='count')

"""Interesting to note we have a 'T' deck value there which doesn't make sense, we can drop it out with the following code:"""

# Redefine cabin_df as everything but where the row was equal to 'T'

cabin_df = cabin_df[cabin_df.Cabin != 'T']
#Replot
sns.catplot(data=cabin_df, x='Cabin', palette='summer', kind='count')

"""Great now that we've analyzed the distribution by decks, let's go ahead and answer our third question:

3.) Where did the passengers come from?
"""

# Let's take another look at our original data
titanic_df.head()

"""Note here that the Embarked column has C,Q,and S values. Reading about the project on Kaggle you'll note that these stand for Cherbourg, Queenstown, Southhampton."""

# Now we can make a quick factorplot to check out the results, note the x_order argument, used to deal with
sns.catplot(data=titanic_df, x='Embarked', hue='Pclass', kind='count', order=['C', 'Q', 'S'])

"""An interesting find here is that in Queenstown, almost all the passengers that boarded there were 3rd class. It would be intersting to look at the economics of that town in that time period for further investigation.

Now let's take a look at the 4th question:

4.) Who was alone and who was with family?
"""

# Let's start by adding a new column to define alone
# We'll add the parent/child column with the sibsp column

titanic_df['Alone'] = titanic_df.Parch + titanic_df.SibSp
titanic_df['Alone']

titanic_df.info()

"""Now we know that if the Alone column is anything but 0, then the passenger had family aboard and wasn't alone. So let's change the column now so that if the value is greater than 0, we know the passenger was with his/her family, otherwise they were alone."""

# Look for >0 or ==0 to set alone status

titanic_df['Alone'].loc[titanic_df['Alone'] >0 ] = 'With family'
titanic_df['Alone'].loc[titanic_df['Alone'] ==0] = 'Alone'

# Let's check to make sure it worked
titanic_df

# Now let's get a simple visualization!
sns.catplot(data=titanic_df, x='Alone', palette='Blues', kind='count')

Great work! Now that we've throughly analyzed the data let's go ahead and take a look at the most interesting (and open-ended) question: What factors helped someone survive the sinking?

# Let's start by creating a new column for legibility purposes through mapping (Lec 36)
titanic_df["Survivor"] = titanic_df.Survived.map({0: "no", 1: "yes"})

# Let's just get a quick overall view of survived vs died. 
sns.catplot(data=titanic_df, x='Survivor', palette='Set1', kind='count')

So quite a few more people died than those who survived. Let's see if the class of the passengers had an effect on their survival rate, since the movie Titanic popularized the notion that the 3rd class passengers did not do as well as their 1st and 2nd class counterparts.



