# Plan Description
In this document we describe the main plan to desing the model for the competition

## Data description

We can describe our dataset as a graph, were the nodes can be:
- **User nodes**
- **Bussiness nodes**

This nodes can be connected by different edges:
- **User -> User**: Friend relationship, can be use to enrich the user prediction.
- **User -> Bussiness**: Review edge. Has different atributtes, including the `rating` label.

### User Description

The avalaible features for user nodes are:
- User Id
- Name
- Review count
- Yelping since (time on platform)
- Friends Id's (as an array)
- Useful votes
- Funny votes
- Cool votes
- Fans Number
- Average start ratings

### Bussiness Description

The avalaible features for bussiness nodes are:
- Bussiness Id
- Name
- Address
- City
- State
- Postal Code
- Latitude
- Longitude
- Average start ratings (rounded to first decimal)
- Is open (bool)
- Categories (as an array, probably one hot encoder applied)
- Atributtes. This is needs to be parsed somehow, probably as a different 

### Review Description

The avalaible features for the review edges are:
- Review id
- Starts (label to predict)
- Date
- Useful reactions
- Funny reactions
- Cool reactions

## User feature vector

The user embedding will be formed by:
- Review count
- Time on platform
- Useful votes
- Funny votes
- Cool votes
- Fans Number

Every user will have this feature vector assigned. A first study will determine how useful can be the features of the user's friends to enrich their own embedding.

If this study ends successfully, the user input will be the user feature vector plus the friends feature vector mean.

## Bussiness feature vector

The bussiness embedding will be defined by two feature vectors.
Firstly, we define the main features vector:
- City (Label Encoder target)
- State (Label Encoder target)
- Latitude
- Longitude
- Mean starts
- Review count
- Is open (boolean flag)
- Categories (Onehot or Label Encoder target, depending on the cardinality of the set)

The other part of the final embedding will be defined using the bussiness attributes. We need to explore this atributtes to ensure the best way to create this vector (DL embeddings, encoders)

## Review feature vector

The review feature vector will we defined by:
- Date timestamp
- Useful flag
- Funny flag
- Cool flag

This features can be used as correctors for the predicted rating, or as confidence value for the rating.

## TODO Analysis

- Check correlation between user fans and all of their review flags. It can be a good way to normalize.
- Check correlation between user time on platform and number of fans and number of reviews. This let us to normalize the users (users with older sign-up date probably has more fans and number of reviews)
- Explore the bussiness atributtes and decided the best strategy to build it vector.

## Model Design

TODO