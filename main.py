from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from recommendation_model import MLPipeline
from collections import OrderedDict
import json


app = Flask(__name__)
api = Api(app)


dataset_url = "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
model_name = "keras-io/collaborative-filtering-movielens"

pipeline = MLPipeline(dataset_url, model_name)
pipeline.build_pipeline()

@app.route('/recommend', methods=['GET'])
def recommend():
    """
    API endpoint to get recommendations.
    """
    try:
        # Get the 'userId' from the query string
        user_id = request.args.get('user_input')

        # Convert user_id to an integer (if needed)
        user_id = int(user_id)

        if user_id is None:
            return jsonify({"error": "Missing 'user_id' in the request."}), 400

        # Get recommendations
        recommendations = pipeline.recommend(user_id)

        # Convert DataFrame to JSON format
        recommendations_json = recommendations.to_dict(orient='records')
        
        response = OrderedDict([
            ("user_id", user_id),  # First key
            ("recommendations", recommendations_json)  # Second key
        ])

        # Use json.dumps to preserve order
        return app.response_class(
            response=json.dumps(response),
            status=200,
            mimetype='application/json'
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/user-interaction', methods=['GET'])
def user_interaction():
    """
    API endpoint to log user interaction via GET request.
    """
    try:
        # Get the 'user_input', 'movie_input', and 'rating_input' from query parameters
        user_input = request.args.get('user_input')
        movie_input = request.args.get('movie_input')
        rating_input = request.args.get('rating_input')

        # Check if any of the required parameters are missing
        if user_input is None or movie_input is None or rating_input is None:
            return jsonify({
                "error": "Missing one or more required query parameters: 'user_input', 'movie_input', 'rating_input'."
            }), 400

        # Convert inputs to appropriate data types (if necessary)
        user_input = int(user_input)
        movie_input = int(movie_input)
        rating_input = float(rating_input)

        # Call the pipeline method
        pipeline.user_interaction(user_input, movie_input, rating_input)


        # Return a success message
        return jsonify({
            "message": "User interaction recorded successfully."
        }), 200

    except Exception as e:
        # Handle exceptions and return an error response
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)