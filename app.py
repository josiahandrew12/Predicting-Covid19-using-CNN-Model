from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def hello():
    return render_template("index.html")

@app.route('/upload', methods = ['POST'])
def upload():
    # HTML -> .py
    if request.method == 'POST':
        name = request.form("file")
    
    # .py -> HTML
    return render_template("upload.html", chestImage = name)



if __name__ == '__main__':
    app.run(debug=True)