

## source: https://www.kaggle.com/kunstmord/exploring-the-songs


#Conclusion
#
#Summing this all up, some features that might be useful:
#
#Number of genres a song has
#Number of composers and lyricists a song has
#Number of plays a song has
#Number of tracks and plays the song's artist has
#The songs language
#The number of languages an artist sings in
#The song's length
#The genres a song has (these can be one-hot encoded, something has to be done for the genres that are in the test set but do not appear in the train set

                       

##hypothesis - looking at what user is listening to what song WHEN, we can decide if the song has a better
## chance of getting repeated

# if user A signs up on 12/31 and user B signs up 9/1, is there a difference in the % of listening to the song
## again based on when the user signed up


### lets look at when users register based on month/year/date seperatly and see if there is any correlation

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


train = pd.read_csv('train.csv')
songs = pd.read_csv('songs.csv')
test = pd.read_csv('test.csv')


print('Song stats: ')
songs_in_train_and_test = np.intersect1d(train['song_id'].unique(), test['song_id'].unique())
print "\n"
print "Unique songs: Train/Test"
print(train['song_id'].nunique(), test['song_id'].nunique())
print "\n"
print "number of songs in the test data set that are not in the training dataset"
print((test['song_id'].nunique() - songs_in_train_and_test.shape[0]))
print "\n"
print "% of songs in test set not in the training set"
print(float((test['song_id'].nunique() - songs_in_train_and_test.shape[0])) / float(test['song_id'].nunique()))

print "We have 359966 different songs in the train set and 224753 different songs in the test set. Also, 59873 of the songs in the test set do no appear in the train set - that is 26.64 % of the test set size."

#
print('User stats: ')
print "# of unique users: Training/Test"
users_in_train_and_test = np.intersect1d(train['msno'].unique(), test['msno'].unique())
print(train['msno'].nunique(), test['msno'].nunique())
print "# of users in test set not in training set"
print((test['msno'].nunique() - users_in_train_and_test.shape[0]))
print "% of users in test set not in training set"
print(float((test['msno'].nunique() - users_in_train_and_test.shape[0])) / float(test['msno'].nunique()))
print "30755 unique users in the train set and 25131 unique users in the test set; () of the users which are in the test set do no appear in the train set - that is 14.52 % of the test set size."




train_merged = train.merge(songs[['song_id', 'artist_name', 'genre_ids',
                                  'language']], on='song_id')
test_merged = test.merge(songs[['song_id', 'artist_name', 'genre_ids',
                                'language']], on='song_id')

print('Artists stats: ')
artists_in_train_and_test = np.intersect1d(train_merged['artist_name'].unique(),
                                           test_merged['artist_name'].unique())
print "# of artists: training/test"
print(train_merged['artist_name'].nunique(), test_merged['artist_name'].nunique())
print "#number of artists in test set not in training set"
print((test_merged['artist_name'].nunique() - artists_in_train_and_test.shape[0]))
print "% of artists in test set not in training set"
print(float((test_merged['artist_name'].nunique() - artists_in_train_and_test.shape[0])) / float(test_merged['artist_name'].nunique()))

print('Language stats: ')
langs_in_train_and_test = np.intersect1d(train_merged['language'].unique(),
                                         test_merged['language'].unique())
print "# of languages: training/test"
print(train_merged['language'].nunique(), test_merged['language'].nunique())
print "# of languages in test set not in training set"
print((test_merged['language'].nunique() - langs_in_train_and_test.shape[0]))
print "% of languages in test set not in training set"
print(float((test_merged['language'].nunique() - langs_in_train_and_test.shape[0])) / float(test_merged['language'].nunique()))

print('Genre stats: ')
genres_in_train_and_test = np.intersect1d(train_merged['genre_ids'].apply(str).unique(),
                                          test_merged['genre_ids'].apply(str).unique())
print "# of genres: test/train"
print(train_merged['genre_ids'].nunique(), test_merged['genre_ids'].nunique())
print "# of genres in test set not in training set"
print((test_merged['genre_ids'].nunique() - genres_in_train_and_test.shape[0]))
print "% of test genres not in training genre"
print(float((test_merged['genre_ids'].nunique() - genres_in_train_and_test.shape[0])) / float(test_merged['genre_ids'].nunique()))



### merging the training dataset with the song data set on song ID
listen_log = train[['msno','song_id','target']].merge(songs,on='song_id')

## take song ID and target(if repeat target = 1) and grouping that by each song and finding the mean of the target (between 0 and 1)
listen_log_groupby = listen_log[['song_id', 'target']].groupby(['song_id']).agg(['mean',
                                                                                 'count'])


listen_log_groupby.reset_index(inplace=True)
listen_log_groupby.columns = list(map(''.join, listen_log_groupby.columns.values))
listen_log_groupby.columns = ['song_id', 'repeat_play_chance', 'plays']  #rename columns

song_data = listen_log_groupby.merge(songs, on='song_id') # merge song data with computed values

song_data['repeat_events'] = song_data['repeat_play_chance'] * song_data['plays']


print "most listened to song was listened to this many times"
print song_data['plays'].max()


x_plays = []
y_repeat_chance = []

for i in range(1,song_data['plays'].max()+1):
    plays_i = song_data[song_data['plays']==i]
    count = plays_i['plays'].sum()
    if count > 0:
        x_plays.append(i)
        y_repeat_chance.append(plays_i['repeat_events'].sum() / count)


f,axarray = plt.subplots(1,1,figsize=(15,10))
plt.xlabel('Number of song plays')
plt.ylabel('Chance of repeat listens')
plt.plot(x_plays, y_repeat_chance)
#plt.show()

## number of song plays has some sort of correlation with how much more likely the song will get repeated

def count_vals(x):
    # count number of values (since we can have mutliple values separated by '|')
    if type(x) != str:
        return 1
    else:
        return 1 + x.count('|')

# count number of genres, composers, lyricsts
song_data['number_of_genres'] = song_data['genre_ids'].apply(count_vals)
song_data['number_of_composers'] = song_data['composer'].apply(count_vals)
song_data['number_of_lyricists'] = song_data['lyricist'].apply(count_vals)

n_genres_max = song_data['number_of_genres'].max()
n_composers_max = song_data['number_of_composers'].max()
n_lyricists_max = song_data['number_of_lyricists'].max()

#print(n_genres_max, n_composers_max, n_lyricists_max)


max_comp_song = song_data.iloc[song_data['number_of_composers'].idxmax()]
max_lyr_song = song_data.iloc[song_data['number_of_lyricists'].idxmax()]
pd.set_option('display.max_colwidth', 200)
#print(max_comp_song[['artist_name', 'composer', 'lyricist', 'number_of_composers',
#                     'number_of_lyricists']], '\n')
#print(max_lyr_song[['artist_name', 'composer', 'lyricist', 'number_of_composers',
#                    'number_of_lyricists']])



x_genres = list(range(1,n_genres_max+1))
x_composers = list(range(1,n_composers_max+1))
x_lyricists = list(range(1,n_lyricists_max+1))

y_genres = [song_data[song_data['number_of_genres'] == x].shape[0] for x in x_genres]
y_composers = [song_data[song_data['number_of_composers'] == x].shape[0] for x in x_composers]
y_lyricists = [song_data[song_data['number_of_lyricists'] == x].shape[0] for x in x_lyricists]

# now, we get some zero values for the # of composers and # lyricists, lets get rid of them
empty_ids = [i for i, y in enumerate(y_composers) if y == 0]
x_composers_fixed = [x_composers[i] for i in range(0,n_composers_max) if i not in empty_ids]
y_composers_fixed = [y_composers[i-1] for i in x_composers_fixed]

empty_ids = [i for i, y in enumerate(y_lyricists) if y == 0]
x_lyricists_fixed = [x_lyricists[i] for i in range(0,n_lyricists_max) if i not in empty_ids]
y_lyricists_fixed = [y_lyricists[i-1] for i in x_lyricists_fixed]

y_repeat_chance_g = []
y_plays_g = []

for i in range(1,n_genres_max+1):
    genres_i = song_data[song_data['number_of_genres']==i]
    count = genres_i['plays'].sum()
    y_repeat_chance_g.append(genres_i['repeat_events'].sum() / count)
    y_plays_g.append(count)

y_repeat_chance_c = []
y_plays_c = []

for i in x_composers_fixed:
    composers_i = song_data[song_data['number_of_composers']==i]
    count = composers_i['plays'].sum()
    y_repeat_chance_c.append(composers_i['repeat_events'].sum() / count)
    y_plays_c.append(count)

y_repeat_chance_l = []
y_plays_l = []

for i in x_lyricists_fixed:
    lyricists_i = song_data[song_data['number_of_lyricists']==i]
    count = lyricists_i['plays'].sum()
    y_repeat_chance_l.append(lyricists_i['repeat_events'].sum() / count)
    y_plays_l.append(count)



fig = plt.figure(figsize=(15, 18))

ax331 = plt.subplot(3,3,1)
sns.barplot(x=x_genres,y=np.log10(y_genres))
ax331.set_ylabel('log10(# of songs)')
ax334 = plt.subplot(3,3,4)
sns.barplot(x=x_genres,y=np.log10(y_plays_g))
ax334.set_ylabel('log10(# of plays)')

ax337 = plt.subplot(3,3,7)
sns.barplot(x=x_genres,y=y_repeat_chance_g)
ax337.set_xlabel('# of genres')
ax337.set_ylabel('chance of repeated listen')


plt.subplot(3,3,2)
sns.barplot(x=x_composers_fixed,y=np.log10(y_composers_fixed))
plt.subplot(3,3,5)
sns.barplot(x=x_composers_fixed,y=np.log10(y_plays_c))
ax338 = plt.subplot(3,3,8)
sns.barplot(x=x_composers_fixed,y=y_repeat_chance_c)
ax338.set_xlabel('# of composers')


plt.subplot(3,3,3)
sns.barplot(x=x_lyricists_fixed,y=np.log10(y_lyricists_fixed))
plt.subplot(3,3,6)
sns.barplot(x=x_lyricists_fixed,y=np.log10(y_plays_l))
ax339 = plt.subplot(3,3,9)
sns.barplot(x=x_lyricists_fixed,y=y_repeat_chance_l)
ax339.set_xlabel('# of lyricists')
#plt.show() # commented out here because dont want the visual generating every time the script is run


## genres seem to have an influence on the number of replays a song will have


## lets look at language

languages = song_data['language'].unique()
print "number of unique langauges"
print(languages.shape[0])

language_count = []
language_plays = []
language_repeat_chance = []

for l in languages:
    if not np.isnan(l):
        songs_with_language = song_data[song_data['language']==l]
        count = songs_with_language['plays'].sum()
        language_repeat_chance.append(songs_with_language['repeat_events'].sum() / count)
        language_count.append(songs_with_language.shape[0])
        language_plays.append(count)
    else:
        songs_with_language = song_data[pd.isnull(song_data['language'])]
        count = songs_with_language['plays'].sum()
        language_repeat_chance.append(songs_with_language['repeat_events'].sum() / count)
        language_count.append(songs_with_language.shape[0])
        language_plays.append(count)

languages[10] = -100  # we'll replace the nan value with something different


fig = plt.figure(figsize=(15, 18))

ax1 = plt.subplot(3,1,1)
sns.barplot(x=languages,y=np.log10(language_count))
ax1.set_ylabel('log10(# of songs)')
ax2 = plt.subplot(3,1,2)
sns.barplot(x=languages,y=np.log10(language_plays))
ax2.set_ylabel('log10(# of plays)')
ax3 = plt.subplot(3,1,3)
sns.barplot(x=languages,y=language_repeat_chance)
ax3.set_ylabel('Chance of repeated listen')
ax3.set_xlabel('Song language')
#plt.show() # uncomment this to view plot showing language and % chance of repeatability



## looking at song length below

min_song_length_sec = float(song_data['song_length'].min()) / 1000.00  # the data is in msec
max_song_length_sec = float(song_data['song_length'].max()) / 1000.00
print "shortest and longest songs. note these are in seconds"
print(min_song_length_sec, max_song_length_sec)

#lets look into these songs

min_length_song = song_data.iloc[song_data['song_length'].idxmin()]
max_length_song = song_data.iloc[song_data['song_length'].idxmax()]
print(min_length_song[['artist_name', 'composer', 'lyricist', 'number_of_composers',
                       'number_of_lyricists', 'song_length', 'repeat_play_chance']], '\n')
print(max_length_song[['artist_name', 'composer', 'lyricist', 'number_of_composers',
                       'number_of_lyricists', 'song_length', 'repeat_play_chance']])


plt.figure(figsize=(15,8))
length_bins = np.logspace(np.log10(min_song_length_sec),np.log10(max_song_length_sec+1),100)
sns.distplot(song_data['song_length']/1000, bins=length_bins, kde=False,
             hist_kws={"alpha": 1})
plt.xlabel('song length, s')
plt.ylabel('# of songs')
plt.yscale('log')
plt.xscale('log')

## does a song's length influence the likelihood of being played again?

time_labels = list(range(length_bins.shape[0]-1))
song_data['time_cuts'] = pd.cut(song_data['song_length']/1000,
                                bins=length_bins, labels=time_labels)

y_repeat_chance_tc = []
y_plays_tc = []
y_rel_plays = []
for i in time_labels:
    timecut_i = song_data[song_data['time_cuts']==i]
    count = timecut_i['plays'].sum()
    y_plays_tc.append(count)
    if count != 0:
        y_repeat_chance_tc.append(timecut_i['repeat_events'].sum() / count)
        y_rel_plays.append(count / timecut_i.shape[0])
    else:
        y_repeat_chance_tc.append(0)
        y_rel_plays.append(0)

fig = plt.figure(figsize=(15, 16))

y_plays_tc = [yptc + 1 for yptc in y_plays_tc]  # otherwise we'll get errors when we take the log

ax211 = plt.subplot(2,1,1)
sns.barplot(x=length_bins[time_labels],y=np.log10(y_plays_tc))
ax211.set_ylabel('log10(# of plays)')

ax212 = plt.subplot(2,1,2)
sns.barplot(x=length_bins[time_labels],y=y_repeat_chance_tc)
ax212.set_ylabel('Chance of repeated listen')

#plt.show()

# its interesting here because shorter then average songs have a lower chance of getting repeated listens
# long songs (3hr +) all get repeated each time (sort of odd)
# does the number of songs an artist have effect the chance of repeated listens?



fig = plt.figure(figsize=(15, 8))

ax111 = plt.subplot(1,1,1)
sns.barplot(x=length_bins[time_labels],y=y_rel_plays)
ax111.set_ylabel('# of plays / # of tracks')

max_tracks = song_data['artist_name'].value_counts().max()
print(song_data['artist_name'].value_counts()[:4])

plt.figure(figsize=(15,8))
track_bins = np.logspace(0,np.log10(max_tracks+1),200)
# track_bins = np.linspace(1,max_tracks+1,100)
sns.distplot(song_data['artist_name'].value_counts(), bins=track_bins, kde=False,
             hist_kws={"alpha": 1})
plt.xlabel('# of tracks')
plt.ylabel('# of artists')
plt.yscale('log')
plt.xscale('log')

## the fewer the nummber of songs an artist has, the better chance the song gets repeated again

artist_groupby = song_data[['artist_name', 'plays']].groupby(['artist_name'])
artist_plays = artist_groupby['plays'].agg(['sum'])
artist_plays.reset_index(inplace=True)

min_plays = artist_plays['sum'].min()
max_plays = artist_plays['sum'].max()
print(min_plays, max_plays)


plt.figure(figsize=(15,8))
play_bins = np.logspace(np.log10(min_plays),np.log10(max_plays+1),100)
# track_bins = np.linspace(1,max_tracks+1,100)
sns.distplot(artist_plays['sum'], bins=play_bins, kde=False,
             hist_kws={"alpha": 1})
plt.xlabel('# of plays')
plt.ylabel('# of artists')
plt.yscale('log')
plt.xscale('log')
#
#So a lot of artists with just a couple of plays (the number of plays and number of tracks is an artist has is probably## #quite correlated).


#
#How does these variables correlate with the chance of repeated listens (as in - how does the number of songs an artist have affect the chance of his songs being played repeatedly; and how does the number of plays an artist has received affect the chance of his songs being played repeatedly)


artist_replgroupby = song_data[['artist_name', 'plays', 'repeat_events']].groupby(['artist_name'])
artist_replgroupby = artist_replgroupby['plays', 'repeat_events'].agg(['sum', 'count'])
artist_replgroupby.reset_index(inplace=True)
artist_replgroupby.columns = list(map(''.join, artist_replgroupby.columns.values))
artist_replgroupby.drop(['repeat_eventscount'], axis=1, inplace=True)
artist_replgroupby.columns = ['artist', 'plays', 'tracks', 'repeat_events']
artist_replgroupby['repeat_play_chance'] = artist_replgroupby['repeat_events'] / artist_replgroupby['plays']


plt.figure(figsize=(15,8))
chance_bins = np.linspace(0,1,100)
sns.distplot(artist_replgroupby['repeat_play_chance'], bins=chance_bins, kde=False,
             hist_kws={"alpha": 1})
plt.xlabel('Chance of repeated listens')
plt.ylabel('# of artists')
plt.yscale('log')
# plt.xscale('log')

## a lot of artists don't have a single repeated listen where as a lot of artists have a 100% chance of being listened to


artist_replgroupby['plays'].max()

play_bins = np.logspace(-0.01, np.log10(artist_replgroupby['plays'].max()), 100)
play_labels = list(range(play_bins.shape[0]-1))
artist_replgroupby['play_cuts'] = pd.cut(artist_replgroupby['plays'],
                                         bins=play_bins, labels=play_labels)

y_repeat_chance_p = []
y_plays_p = []
for i in play_labels:
    playcut_i = artist_replgroupby[artist_replgroupby['play_cuts']==i]
    count = artist_replgroupby['plays'].sum()
    y_plays_p.append(count)
    if count != 0:
        y_repeat_chance_p.append(playcut_i['repeat_events'].sum() / count)
    else:
        y_repeat_chance_p.append(0)

fig = plt.figure(figsize=(15, 16))

ax111 = plt.subplot(1,1,1)
sns.barplot(x=play_bins[play_labels],y=y_repeat_chance_p)
ax111.set_xlabel('log10(# of plays)')
ax111.set_ylabel('Chance of repeated listen')

#So it looks like artists with a lower number of plays also have a lower chance of repeated listens, whereas when the number of plays increases, the data behaves more erratically.


track_bins = np.logspace(-0.01, np.log10(artist_replgroupby['tracks'].max()), 50)
track_labels = list(range(track_bins.shape[0]-1))
artist_replgroupby['track_cuts'] = pd.cut(artist_replgroupby['tracks'],
                                          bins=track_bins, labels=track_labels)

y_repeat_chance_t = []
y_tracks_t = []
for i in track_labels:
    trackcut_i = artist_replgroupby[artist_replgroupby['track_cuts']==i]
    count = artist_replgroupby['tracks'].sum()
    y_tracks_t.append(count)
    if count != 0:
        y_repeat_chance_t.append(trackcut_i['repeat_events'].sum() / count)
    else:
        y_repeat_chance_t.append(0)

fig = plt.figure(figsize=(15, 16))

ax111 = plt.subplot(1,1,1)
sns.barplot(x=track_bins[track_labels],y=y_repeat_chance_t)
ax111.set_xlabel('log10(# of tracks)')
ax111.set_ylabel('Chance of repeated listen')

## in general, a artist with a low number of tracks leads to a lower replay chance



artist_langgroupby = song_data[['artist_name',  'language']].groupby(['artist_name'])
artist_langgroupby = artist_langgroupby.agg({"language": pd.Series.nunique})
artist_langgroupby.reset_index(inplace=True)
artist_langgroupby.columns = list(map(''.join, artist_langgroupby.columns.values))
artist_langgroupby.columns = ['artist', 'language']

artist_repl_lang = artist_replgroupby.merge(artist_langgroupby, on='artist')


plt.figure(figsize=(15,8))
chance_bins = np.linspace(1,artist_repl_lang['language'].max()+1,11)
sns.distplot(artist_repl_lang['language'], bins=chance_bins, kde=False,
             hist_kws={"alpha": 1})
plt.xlabel('# of languages an artist signs in')
plt.ylabel('# of artists')
plt.yscale('log')

# most artists sing in only one language, some sing up to 4 languages

y_repeat_chance_l = []
y_plays_l = []
y_tracks_l = []

max_l = int(artist_repl_lang['language'].max())
l_list = []

for i in range(1,max_l+1):
    arlang = artist_repl_lang[artist_repl_lang['language']==i]
    count = arlang['plays'].sum()
    if count != 0:
        y_tracks_l.append(arlang['tracks'].sum())
        y_plays_l.append(count)
        l_list.append(i)
        y_repeat_chance_l.append(arlang['repeat_events'].sum() / count)

fig = plt.figure(figsize=(15, 24))

ax311 = plt.subplot(3,1,1)
sns.barplot(x=l_list,y=np.log10(y_tracks_l))
ax311.set_xlabel('# of languages')
ax311.set_ylabel('log10(# of tracks)')

ax312 = plt.subplot(3,1,2)
sns.barplot(x=l_list,y=np.log10(y_plays_l))
ax312.set_xlabel('# of languages')
ax312.set_ylabel('log10(# of plays)')

ax313 = plt.subplot(3,1,3)
sns.barplot(x=l_list,y=y_repeat_chance_l)
ax313.set_xlabel('# of languages')
ax313.set_ylabel('Chance of repeated listen')

## not many artists who sing 4+ languages, but they tend to get a lot of plays
# artists who sing 5 langauges seem to have a lower replay chance


## now lets look at genres. some songs have 8 songs within the song
## need to split these up since they are | delimited

def split_genres(x, n):
    # n is the number of the genre
    if type(x) != str:
        if n == 1:
            if not np.isnan(x):
                return int(x)
            else:
                return x
    else:
        if x.count('|') >= n-1:
            return int(x.split('|')[n-1])


max_genres = song_data['number_of_genres'].max()

for i in range(1,max_genres+1):
    sp_g = lambda x: split_genres(x, i)
    song_data['genre_'+str(i)] = song_data['genre_ids'].apply(sp_g)

n_genres = set()

for i in range(1,max_genres+1):
    n_genres.update(song_data['genre_'+str(i)][song_data['genre_'+str(i)].notnull()].unique().tolist())


## there are 166 genres. there are also 7233 songs with no genre information (NaN genre)

print len(n_genres), song_data['genre_ids'].isnull().sum()



genres_plays = [0] * (len(n_genres) + 1)
genres_tracks = [0] * (len(n_genres) + 1)
genres_replays = [0] * (len(n_genres) + 1)

for i in range(1,max_genres+1):
    notnull_data = song_data[song_data['genre_'+str(i)].notnull()]
    for j, k in enumerate(n_genres):
        jk_sdata = notnull_data[notnull_data['genre_'+str(i)] == k]
        genres_plays[j] += jk_sdata['plays'].sum()
        genres_tracks[j] += jk_sdata['plays'].shape[0]
        genres_replays[j] += jk_sdata['repeat_events'].sum()

null_genre_data = song_data[song_data['genre_1'].isnull()]
genres_plays[len(n_genres)] = null_genre_data['plays'].sum()
genres_tracks[len(n_genres)] = null_genre_data['plays'].shape[0]
genres_replays[len(n_genres)] = null_genre_data['repeat_events'].sum()

genres_rel_plays = [x/y for x, y in zip(genres_plays, genres_tracks)]
genres_repl_chance = [x/y for x, y in zip(genres_replays, genres_plays)]

n_g_l = [x for x in n_genres]
n_g_l.append(-1)

fig = plt.figure(figsize=(15, 27))

ax411 = plt.subplot(3,1,1)
sns.barplot(x=n_g_l,y=np.log10(genres_tracks))
ax411.set_ylabel('log10(# of tracks)')


ax413 = plt.subplot(3,1,2)
sns.barplot(x=n_g_l,y=genres_rel_plays)
ax413.set_ylabel('# of plays/ # of tracks')

ax414 = plt.subplot(3,1,3)
sns.barplot(x=n_g_l,y=genres_repl_chance)
ax414.set_ylabel('Chance of repeat listen')

fig = plt.figure(figsize=(15, 24))

ax412 = plt.subplot(2,1,1)
sns.barplot(x=n_g_l,y=np.log10(genres_plays))
ax412.set_ylabel('log10(# of plays)')

plt.show()

