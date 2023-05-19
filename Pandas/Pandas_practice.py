#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 23:37:38 2023

@author: tungdang
"""

import pandas as pd 

#################### Reading data files ####################

wine_reviews = pd.read_csv("winemag-data-130k-v2.csv", index_col=0)
wine_reviews.head()
wine_reviews.tail()

wine_reviews["country"]
wine_reviews["country"][0]

# Randomly select fraction of rows 
wine_reviews.sample(frac=0.5)

# Randomly select n rows 
wine_reviews.sample(n=10)

# Select top n 
top_poins = wine_reviews.nlargest(10, "points")

# Select bottom n 
bottom_points = wine_reviews.nsmallest(10, "points")

points_price = wine_reviews.query("points > 91 and price > 42")

################# Index-based selection ####################

wine_reviews.iloc[0]

wine_reviews.iloc[:, 0]

wine_reviews.iloc[:3, 0]

wine_reviews.iloc[1:3, 0]

# Select last 5 lines 
wine_reviews.iloc[-5:]

# Access single value by index
wine_reviews.iat[1,2]


#################### Label-based selection ####################

wine_reviews.loc[0, "country"]

wine_reviews.loc[:, ['taster_name', 'taster_twitter_handle','points']]

# Access sigle value by label
wine_reviews.at[4, "country"]
wine_reviews.at[4, "description"]


#################### Manipulating the index ####################

wine_reviews.set_index("title")

wine_reviews.drop(columns = ["country", "description"])

#################### Conditional selection ####################

wine_reviews["country"] == "Italy"

wine_italy = wine_reviews.loc[wine_reviews["country"] == "Italy"]

wine_italy_90 = wine_reviews.loc[(wine_reviews["country"] == "Italy") & (wine_reviews["points"] >= 90)]

# Wines in Italy or points >= 90
wine_italy_90 = wine_reviews.loc[(wine_reviews["country"] == "Italy") | (wine_reviews["points"] >= 90)]

# Select wines only from Italy and France 
wine_italy_france = wine_reviews.loc[wine_reviews["country"].isin(["Italy", "France"])]

# Filter out wines lacking a price tag
wine_price = wine_reviews.loc[wine_reviews["price"].notnull()]


#################### Summary Function ####################

wine_reviews.shape

wine_reviews["points"].describe()

wine_reviews["taster_name"].describe()
wine_reviews["taster_name"].unique()

# List of unique values and how often they occur 
wine_reviews["taster_name"].value_counts()

wine_reviews_mean = wine_reviews["points"].mean()
wine_reviews["points"].map(lambda p: p - wine_reviews_mean)


# Want to transform a whole DataFrame by calling custom method on each row
def remean_points(row):
    row.points = row.points - wine_reviews_mean
    return row

wine_reviews.apply(remean_points, axis="columns")


#################### Groupwise analysis ####################

wine_reviews.groupby("points").points.count()

wine_reviews.groupby("points").price.mean()
wine_reviews.groupby("points").price.describe()


# select the name of the first wine reviewed from each winery 
wine_reviews.groupby("winery").apply(lambda df: df.title.iloc[0])

# pick up the best wine by country and provine
wine_country_province = wine_reviews.groupby(["country", "province"]).apply(lambda df: df.loc[df["points"].idxmax()])


# Generate a simple statistical summary of the dataset 
wine_reviews.groupby(["country"]).price.agg([len, min, max])



#################### Multiple indexs ####################

countries_index = wine_reviews.groupby(["country", "province"]).description.agg([len])


#################### Sorting ####################

countries_index = countries_index.reset_index()

countries_index.sort_values(by="len", ascending=False)

countries_index.sort_index()

countries_index.sort_values(by=["country", "len"])

wine_drop = wine_reviews.drop(columns = ["country", "description"])

#################### Missing values ####################

# Entry missing values are given the value NaN 
wine_reviews[pd.isnull(wine_reviews["country"])]

# Replacing missing values 
wine_reviews["region_2"].fillna("unknown")


#################### Renaming ####################

wine_score = wine_reviews.rename(columns = {"points": "score"})

wine_reviews.rename(index = {0: "firstEntry", 1: "secondEntry"})


#################### Spliting ####################

df1 = wine_reviews.iloc[:,:5]
df2 = wine_reviews.iloc[:,5:]

#################### Combining ####################

# Append rows of Data
pd.concat([df1, df2])

# Append columns of Data
pd.concat([df1, df2], axis=1)




































