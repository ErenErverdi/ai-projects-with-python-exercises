import numpy as np
import pandas as pd

column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('users.data', sep='\t', names=column_names)

df_movie_title = pd.read_csv("movie_id_titles.csv")

df = pd.merge(df,df_movie_title,on='item_id')
df.head()

moviemat = df.pivot_table(index='user_id',columns='title',values='rating')
moviemat.head()

type(moviemat)

starwars_user_ratings = moviemat['Star Wars (1977)']
starwars_user_ratings

similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_to_starwars

type(similar_to_starwars)

corr_starwars = pd.DataFrame(similar_to_starwars, columns=['Correlation'])
corr_starwars.dropna(inplace=True)
corr_starwars

corr_starwars.sort_values('Correlation',ascending=False).head(50)

df.drop(['timestamp'],axis=1)

ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings.sort_values('rating',ascending=False).head(50)
ratings['rating_number'] = pd.DataFrame(df.groupby('title')['rating'].count())
ratings.sort_values('rating_number',ascending=False).head(50)
corr_starwars = corr_starwars.join(ratings['rating_number'])
corr_starwars
corr_starwars[corr_starwars['rating_number']>100].sort_values('Correlation',ascending=False).head(50)
