# Import necessary libraries
import pandas as pd  # For handling data
from pathlib import Path
import numpy as np
from zipfile import ZipFile
import keras
from sklearn.cluster import KMeans

from huggingface_hub import from_pretrained_keras

NUM_GROUPS = 15 # Number of desired user groups

# 1. DATA INGESTION COMPONENT
class DataIngestion:
    def __init__(self, dataset_url):
        self.dataset_url = dataset_url

    def download_data(self):
        movielens_data_file_url = (
            self.dataset_url
        )
        movielens_zipped_file = keras.utils.get_file(
            "ml-latest-small.zip", movielens_data_file_url, extract=False
        )
        keras_datasets_path = Path(movielens_zipped_file).parents[0]
        movielens_dir = keras_datasets_path / "ml-latest-small"

        # Only extract the data the first time the script is run.
        if not movielens_dir.exists():
            with ZipFile(movielens_zipped_file, "r") as zip:
                # Extract files
                print("Extracting all the files now...")
                zip.extractall(path=keras_datasets_path)
                print("Done!")

        ratings_file = movielens_dir / "ratings.csv"
        ratings_df = pd.read_csv(ratings_file)

        movies_file = movielens_dir / "movies.csv"
        movie_df = pd.read_csv(movies_file)
        return ratings_df, movie_df

# 2. DATA PREPROCESSING COMPONENT
class DataPreprocessor:
    def __init__(self, data):
        self.data = data

    def encoded_df(self):
        # Encode movieId to movie
        movie_ids = self.data["movieId"].unique().tolist()
        movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
        movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}

        # Encode userId to user
        if "userId" in self.data.columns:
          user_ids = self.data["userId"].unique().tolist()
          user2user_encoded = {x: i for i, x in enumerate(user_ids)}
          userencoded2user = {i: x for i, x in enumerate(user_ids)}

        self.data["movie"] = self.data["movieId"].map(movie2movie_encoded)
        if "userId" in self.data.columns:
          self.data["user"] = self.data["userId"].map(user2user_encoded)

        if "rating" in self.data.columns:
          self.data["rating"] = self.data["rating"].values.astype(np.float32)
          self.data.drop(columns=["timestamp", "userId", "movieId"], inplace=True, errors="ignore")
          # Ensure columns exist before reordering
          columns_to_keep = [col for col in ["user", "movie", "rating"] if col in self.data.columns]
          self.data = self.data[columns_to_keep]
        else:
          self.data.drop(columns=["movieId"], inplace=True)
          columns_to_keep = [col for col in ["movie", "title", "genres"] if col in self.data.columns]
          self.data = self.data[columns_to_keep]

        return self.data

    def cluster(self, ratings_df, model, user_embeddings):

        # Define clustering model
        clustering_model = KMeans(n_clusters=NUM_GROUPS, random_state=42)

        # Fit the model and predict cluster labels
        labels_cluster = clustering_model.fit_predict(user_embeddings)

        # Create DataFrame that holds user profile vectors
        df_user = pd.DataFrame({
            'user': range(0, len(user_embeddings)),
            'profile_vec': [(embedding) for embedding in user_embeddings],
            'user_group': labels_cluster,
            'watched_movies': ratings_df.groupby('user').agg({
                                  'movie': list
                              }).reset_index().movie,
            'recommended_movies': [[] for _ in range(len(user_embeddings))],
            'satisfaction_score': None,
            'last_five_ratings': [[] for _ in range(len(user_embeddings))]
        })

        return df_user, clustering_model




# 3. MODEL COMPONENT
class PretrainedModel:
    def __init__(self, model_name):
        self.model = from_pretrained_keras(model_name)
        self.user_embeddings = self.model.user_embedding.get_weights()[0]
        self.fit = self.model.fit
        self.top_user_recommendations = None

    def get_group_ratings(self, group_id, ratings_df, df_user):
        # Step 1: Filter ratings over 3
        df_ratings_filtered = ratings_df[ratings_df['rating'] > 3]

        # Step 2: Filter users in the specific user group

        df_user_group = df_user[df_user['user_group'] == group_id]

        # Step 3: Merge the two dataframes on userId
        df_merged = df_user_group[['user']].merge(df_ratings_filtered, on='user', how='inner')

        # Step 4: Group by movie to get average rating for each movie in the group and count of ratings
        df_group_ratings = df_merged.groupby('movie').agg(
            avg_rating=('rating', 'mean'),
            rating_count=('rating', 'size')
        ).reset_index()

        # Step 5: Sort by average rating and then by rating count to rank the movies
        df_group_ratings_sorted = df_group_ratings.sort_values(by=['avg_rating', 'rating_count'], ascending=[False, False])

        return df_group_ratings_sorted

    def get_pairs_per_user(self, user, df_user, ratings_df):
        # Get the list of recommended movies for the current group
        group_movies_df = self.get_group_ratings(df_user.iloc[user].user_group, ratings_df, df_user)

        # Get the movie IDs for the current group
        movie_ids = group_movies_df['movie'].tolist()

        # Get the list of watched movies for the current user
        watched_movies = df_user.iloc[user].watched_movies

        # Exclude watched movies from the movie_ids list
        movie_ids = [movie_id for movie_id in movie_ids if movie_id not in watched_movies]

        # Get all users in the current group
        user_ids = [user]

        # Create pairs of each userId with each movieId in the list
        pairs = [[user_id, movie_id] for user_id in user_ids for movie_id in movie_ids]

        return pairs

    def recommendations(self, user, df_user, ratings_df):
        # Get the user-movie pairs for the current group
        user_movie_pairs = self.get_pairs_per_user(user, df_user, ratings_df)

        # Get the model predictions for the current group's user-movie pairs
        user_movie_pairs = [np.array(pair, dtype=np.int64) for pair in user_movie_pairs]
        predictions = self.model.predict(user_movie_pairs)

        # Convert predictions to a list of lists, where each sublist is [userId, movieId, predicted_rating]
        results = [[user_movie_pairs[i][0], user_movie_pairs[i][1], predictions[i][0]] for i in range(len(user_movie_pairs))]

        # Convert results to a DataFrame for easy manipulation
        df_results = pd.DataFrame(results, columns=['user', 'movie', 'predicted_rating'])

        # Group by userId and apply sorting within each user group to get the top 1 predicted_rating
        self.top_user_recommendations = (
            df_results.sort_values(by=['user', 'predicted_rating'], ascending=[True, False])
            .groupby('user')
            .head(10)  # Only keep the highest-predicted_rating movie for each user
        )

        df_user.at[user, 'recommended_movies'] = [(movie, predicted_rating) for movie, predicted_rating in zip(self.top_user_recommendations['movie'], self.top_user_recommendations['predicted_rating'])]

        return df_user



class MLPipeline:
    def __init__(self, dataset_url, model_name):
        self.data_ingestion = DataIngestion(dataset_url)
        self.preprocessor = None
        self.pretrained_model = PretrainedModel(model_name)
        self.feedback = Feedback()
        self.clustering_model = None
        self.ratings_df = None
        self.movie_df = None
        self.df_user = None
        self.learning_rate = 0.01

    def build_pipeline(self):
        # Step 1: Download and load data
        print("Downloading and loading data...")
        self.ratings_df, self.movie_df = self.data_ingestion.download_data()

        # Step 2: Preprocess data
        print("Preprocessing data...")
        self.preprocessor = DataPreprocessor(self.ratings_df)
        self.ratings_df = self.preprocessor.encoded_df()

        self.preprocessor = DataPreprocessor(self.movie_df)
        self.movie_df = self.preprocessor.encoded_df()
        print("Clustering users by profile vectors...")
        self.df_user, self.clustering_model = self.preprocessor.cluster(self.ratings_df, self.pretrained_model, self.pretrained_model.user_embeddings)
        return
    
    def retrain_model(self, feedback):
        self.ratings_df = self.ratings_df.sample(frac=1, random_state=42)
        x = self.ratings_df[["user", "movie"]].values
        y = self.ratings_df["rating"].apply(lambda x: feedback.normalize(x)).values

        train_indices = int(0.9 * self.ratings_df.shape[0])
        x_train, x_val, y_train, y_val = (
            x[:train_indices],
            x[train_indices:],
            y[:train_indices],
            y[train_indices:],
        )

        self.pretrained_model.fit(
            x=x_train,
            y=y_train,
            batch_size=64,
            epochs=5,
            verbose=1,
            validation_data=(x_val, y_val),
        )

        user_embeddings = self.pretrained_model.user_embedding.get_weights()[0]
        # Fit the model and predict cluster labels
        labels_cluster = self.clustering_model.fit_predict(user_embeddings)

        df_user = pd.DataFrame({
            'user': range(0, len(user_embeddings)),
            'profile_vec': [(embedding) for embedding in user_embeddings],
            'user_group': labels_cluster,
            'watched_movies': self.ratings_df.groupby('user').agg({
                                  'movie': list
                              }).reset_index().movie,
            'recommended_movies': [[] for _ in range(len(user_embeddings))],
            'satisfaction_score': None,
            'last_five_ratings': [[] for _ in range(len(user_embeddings))]
        })
        return 

    def recommend(self, user):
        # Step 3: Model prediction
        print("Making predictions using the model...")
        self.user_df = self.pretrained_model.recommendations(user, self.df_user, self.ratings_df)

        result = pd.merge(self.pretrained_model.top_user_recommendations, self.movie_df, on='movie', how='inner')
        return result[['title','genres','movie']]

    def user_interaction(self, user, movie, rating):
        retrain_model_treshold = False
        self.ratings_df, update_vec = self.feedback.user_feedback(user, movie, rating, self.df_user, self.ratings_df)
        if update_vec:
          self.df_user = self.feedback.update_user_vec(user, movie, self.df_user, self.pretrained_model, self.clustering_model, self.learning_rate)
        if self.df_user.satisfaction_score.mean() < 0.65:
          retrain_model_treshold = True
        if retrain_model_treshold:
          self.retrain_model(self.feedback)
        return 
    
    def print_df(self, name):
        dataframes = {
            'df_user': self.df_user,
            'ratings_df': self.ratings_df
        }
        if name in dataframes:
            print(f"DataFrame '{name}':")
            #display(dataframes[name])
        else:
            raise KeyError(f"'{name}' not found in the DataFrame dictionary.")

        return


# 5. FEEDBACK COMPONENT
class Feedback:
    def __init__(self):
        self.beta = 0.9
    
    def normalize(self, value):
        # Min and Max for the range
        min_val = 0
        max_val = 5

        # Normalization formula
        normalized_value = (value - min_val) / (max_val - min_val)
        return normalized_value
        
    def user_feedback(self, user, movie, rating, df_user, ratings_df):
        """
        Interacts with two dataframes: df_user, ratings_df

        input:
        - rating input is in scale 0 to 5

        """
        update_vec = False
        norm_rating = self.normalize(rating)

        # Add ratings to list

        predicted_rating = next((rating for movieintuple, rating in df_user.loc[user,'recommended_movies'] if movieintuple == movie), None)
        if (predicted_rating):
          df_user.loc[user, 'last_five_ratings'].append((norm_rating, predicted_rating))
          if (len(df_user.loc[user, 'last_five_ratings']) == 5):
            update_vec = True
        else:
          print("Error! Movie was not recommended!")
          return
        # Add movie to watched movies
        df_user.loc[user, 'watched_movies'].append(movie)

        # Add new rating to ratings_df
        new_rating = {'user': user, 'movie': movie, 'rating': rating}
        ratings_df = pd.concat([ratings_df, pd.DataFrame([new_rating])], ignore_index=True)

        # Update satisfaction score
        current_score = df_user.loc[user, 'satisfaction_score']
        if current_score is not None:
          new_score = self.beta * current_score + (1 - self.beta) * norm_rating
          new_score = max(0, min(new_score, 1)) # Clamp the score between 0 and 1
        else:
          new_score = self.beta * 1.0 + (1 - self.beta) * norm_rating
          new_score = max(0, min(new_score, 1))
        df_user.loc[user, 'satisfaction_score'] = new_score

        return ratings_df, update_vec

    def update_user_vec(user, movie, df_user, model, clustering_model, learning_rate):
        """
        Updates user vectors based on the last five ratings.

        """
        movie_embeddings = model.movie_embedding.get_weights()[0]
        gradient = np.zeros_like(movie_embeddings[movie])

        for actual, predicted in df_user.loc[user, "last_five_ratings"]:
          # Compute the error (prediction - actual)
          error = predicted - actual
          # Compute the gradient for the user embedding (derivative of (error)^2)
          gradient += 2 * error * movie_embeddings[movie]

        # Average the gradient for the last 5 ratings
        gradient /= 5

        # Apply gradient descent to update the user embedding
        new_user_vec = df_user.loc[user, "profile_vec"] - learning_rate * gradient

        # Update model weights
        user_embeddings = model.user_embedding.get_weights()
        user_embeddings[0][user] = new_user_vec
        model.user_embedding.set_weights(user_embeddings)

        df_user.at[user, "profile_vec"] = new_user_vec
        df_user.at[user, "last_five_ratings"] = [] # Empty the list

        # Recluster user
        label_cluster = clustering_model.predict(df_user.at[user, "profile_vec"].reshape(1,-1))
        df_user.at[user, "user_group"] = label_cluster

        return df_user


