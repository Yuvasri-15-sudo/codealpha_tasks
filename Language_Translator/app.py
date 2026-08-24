from flask import Flask, render_template, request
from deep_translator import GoogleTranslator

app = Flask(__name__)

history = []

@app.route('/', methods=['GET', 'POST'])
def home():
    global history
    translated_text = ""

    if request.method == "POST":
        text = request.form["text"]
        source = request.form["source"]
        target = request.form["target"]

        translated_text = GoogleTranslator(
            source=source,
            target=target
        ).translate(text)

        history.insert(0, {
            "original": text,
            "translated": translated_text
        })

        history = history[:5]

    return render_template("index.html", translated_text=translated_text,history=history)

if __name__ == "__main__":
    app.run(debug=True)