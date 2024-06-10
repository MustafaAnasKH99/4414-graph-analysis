# 4414-graph-analysis

A team project exploring the network of collaborations between Spotify musicians and artists from across the globe! This project is done by a team of undergraduate students enrolled in EECS4414 in York University. The dataset was found from the following link: https://www.kaggle.com/datasets/jfreyberg/spotify-artist-feature-collaboration-network. We acknowledge that we do not own the dataset and are merely using it as an example database. 

## TO DO:
- Fix AverageGenres and AverageCountries
  - Push genres/countries to a set
  - register the len of the set
  - average len(artistGenres/set len)
  - store agerages in an array too
  - Average of all artist averages
- Find all pairs shortest paths to the full network [Mustafa : **In progress**]
- One or two extra community prediction algorithms and then compare to each other. [Jibril : **Completed**]
- Find most popular genre in each country and most popular country in each genre. [Jibril : **Completed**]
  - Ex. Like what genre is most popular in Canada and what country has the most rock artists/hits?
