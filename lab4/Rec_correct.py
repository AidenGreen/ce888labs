import pandas as pd
import numpy as np
from IPython.display import Image
np.set_printoptions(precision = 3)

Image(filename='books.png')

data = pd.read_csv("user_ratings.csv")
d = data.to_latex()
text_file = open("Output.txt", "w")
text_file.write(d)
text_file.close()

n_features = 2

user_ratings = data.values
latent_user_preferences = np.random.random((user_ratings.shape[0], n_features))
latent_item_features = np.random.random((user_ratings.shape[1],n_features))

latent_item_features

latent_user_preferences

def predict_rating(user_id,item_id):
    """ Predict a rating given a user_id and an item_id.
    """
    user_preference = latent_user_preferences[user_id]
    item_preference = latent_item_features[item_id]
    return user_preference.dot(item_preference)

def train(user_id, item_id, rating,alpha = 0.0001):
    
    #print item_id
    prediction_rating = predict_rating(user_id, item_id)
    err =  ( prediction_rating- rating )
    #print err
    user_pref_values = latent_user_preferences[user_id][:]
    latent_user_preferences[user_id] -= alpha * err * latent_item_features[item_id]
    latent_item_features[item_id] -= alpha * err * user_pref_values
    return err
    


def sgd(iterations = 20000):
    """ Iterate over all users and all items and train for 
        a certain number of iterations
    """
    for iteration in range(0,iterations):
        error = []
        for user_id in range(0,latent_user_preferences.shape[0]):
            for item_id in range(0,latent_item_features.shape[0]):
                rating = user_ratings[user_id][item_id]
                if(not np.isnan(rating)):
                    err = train(user_id,item_id,rating)
                    error.append(err)
        mse = (np.array(error) ** 2).mean()   
        if(iteration%10000 == 0 ):
            print(mse)
                    

sgd()

predictions = latent_user_preferences.dot(latent_item_features.T)

print(predictions)

values = [zip(user_ratings[i], predictions[i]) for i in range(0,predictions.shape[0])]
comparison_data = pd.DataFrame(values)
comparison_data.columns = data.columns
comparison_data.applymap(lambda x :x)
print(comparison_data)