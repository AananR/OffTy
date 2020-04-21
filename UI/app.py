from flask import Flask, render_template, request
import os
import main as ms

app = Flask(__name__)
STATIC_DIR = os.path.abspath('../static')


@app.route('/')


def index():
	return render_template('index.html')






@app.route('/search',methods = ['POST', 'GET'])
def search():
   if request.method == 'POST':
       query = request.form.get('query', None)

       result = ms.main(query)

       return render_template("search.html", result=result)


if __name__ == '__main__':
	app.run(debug=True)