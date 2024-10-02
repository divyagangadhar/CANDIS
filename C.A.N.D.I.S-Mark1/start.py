from flask import Flask
from flask import request
from flask import render_template
import stringComparison

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template("form.html") # this should be the name of your html file

@app.route('/', methods=['POST'])
def my_form_post():
    text1 = request.form['text1']
    text2 = request.form['text2']
    mPercent = stringComparison.extremelySimpleChecker(text1,text2)
    if text2 && text1 != (__name__) :
        return "<h1>Opening new window</h1>"
    else :
        return "<h1>Please try again !</h1>"

if __name__ == '__main__':
    app.run()