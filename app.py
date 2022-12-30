from flask import Flask, render_template, request
import tensorflow as tf


app = Flask(__name__)

reloaded_model = tf.keras.models.load_model('saved_model/20221228210641')
cols = ["Pclass", "Sex", "Age", "Fare"]


@app.route("/")
def home():
    return render_template("home.html")


@app.route('/predict', methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    last_features = []
    for i in features:
        try:
            last_features.append(float(i))
        except ValueError:
            last_features.append(str(i))
    combined_features = dict(zip(cols, last_features))
    tensor_features = {name: tf.convert_to_tensor([value]) for name, value in combined_features.items()}
    predictions = reloaded_model(tensor_features)
    prob = tf.nn.sigmoid(predictions[0][0])
    return render_template('home.html', pred=f'The chance of survival is {prob * 100:.1f}%')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)


