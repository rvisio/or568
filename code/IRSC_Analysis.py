
#source: https://www.kaggle.com/joshwilkins2013/examining-irsc-data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

df_songs_extra = pd.read_csv('song_extra_info.csv')
#print df_songs_extra.head()


#IRSC split into CC-XXX-YY-NNNNN
#CC is Country Code
#XXX is IRSC Issuer (Record Label Company)
#YY is the year the IRSC was assigned (not necessarily the year it was recorded)
#NNNNN is just a 5 digit id number



#Before we proceed, lets look at how much duplicated IRSC data we have and how much is missing


print ("%.2f%% of IRSCs are duplicates" % (100 - float(100*len(df_songs_extra.isrc.unique())) / float(len(df_songs_extra.isrc))))
print ("%.2f%% of IRSCs are missing" % (100 * float(df_songs_extra.isrc.isnull().sum()) / float(len(df_songs_extra.song_id))))

#
#Lets split the data into three new columns; CC, XXX, YY.
#Note the NNNNN column is just an id and not worth keeping


# Spliting IRSC Data into CC-XXX-YY
x = pd.Series(df_songs_extra.isrc.values)
df_songs_extra['cc'] = x.str.slice(0,2)  # Country Code column
df_songs_extra['xxx'] = x.str.slice(2,5) # IRSC Issuer
df_songs_extra['yy'] = x.str.slice(5,7).astype(float)  # IRSC issue date
del df_songs_extra['isrc']  # Remove isrc column

# Convert to 4 digit year
df_songs_extra.loc[df_songs_extra['yy'] > 17, 'yy'] += 1900  # 1900's songs
df_songs_extra.loc[df_songs_extra['yy'] < 18, 'yy'] += 2000  # 2000's songs
df_songs_extra.rename(columns={'yy': 'yyyy'}, inplace=True)

print df_songs_extra.head()



def count(col, data):
    plt.figure()
    plt.figure(figsize=(10,7))
    groups = data.groupby(col)['song_id', 'name'].count()
    groups.reset_index(inplace=True)
    groups.columns = [col, 'num_songs', 'placeholder']
    sbn.barplot(groups[col], groups['num_songs'])
    plt.title("Number of Songs per group in " + col.upper())

count('yyyy', df_songs_extra)
count('cc', df_songs_extra)
count('xxx', df_songs_extra[:2000])


# Loading in the training set
df_train = pd.read_csv('train.csv')
train = df_train.merge(df_songs_extra, how='left', on='song_id')

def chance(col, data):
    plt.figure()
    plt.figure(figsize=(10,7))
    groups = data.groupby(col)
    x_axis = [] # Sort by type
    repeat = [] # % of time repeated
    for name, group in groups:
        count0 = float(group[group.target == 0][col].count())
        count1 = float(group[group.target == 1][col].count())
        percentage = count1/(count0 + count1)
        x_axis = np.append(x_axis, name)
        repeat = np.append(repeat, percentage)
    plt.title("Repeat Chance by Group in " + col)
    plt.ylabel('Repeat Chance')
    sbn.barplot(x_axis, repeat)

chance('yyyy', train)
chance('cc', train)
chance('xxx', train[:2000])




plt.show()

#
#There looks to be a pretty good correlation of the chance a user repeats a song to that of the year the IRSC was assigned. Although this kinda starts to fall apart for earlier years.
#Although there are some country codes that have either a 0% or a 100% chance of repeat, you should note this would strongly be affected by the sample size for each group. And not all songs are found within the training set.
#There appears to be no correlation between xxx and repeat chance.

