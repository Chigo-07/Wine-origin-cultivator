from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

with open("model.h5", "rb") as f:
    # Grab the model and scaler, and ignore anything else in the file
    data = pickle.load(f)
    model = data[0]
    scaler = data[1]
    

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        values = [float(x) for x in request.form.values()]
        values = scaler.transform([values])
        result = model.predict(values)[0]

    return render_template("index.html", result=result)

app.run(debug=True)
port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)
