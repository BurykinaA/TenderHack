from flask import Flask, render_template, request
from search import score, retrieve, build_index
from time import time

app = Flask(__name__, template_folder='.')
build_index()


@app.route('/Search.html', methods=['GET'])
def search():
    start_time = time()
    query = request.args.get('query')
    if query is None:
        query = ''
    documents = retrieve(query)
    results = [doc.format(query) for doc in documents]
    return render_template(
        'template/Search.html',
        time="%.2f" % (time() - start_time),
        query=query,
        results=results
    )


@app.route('/', methods=['GET'])
def index():
    query = request.args.get('query')
    return render_template('template/index.html')


@app.route('/Privacy.html', methods=['GET'])
def privacy():
    query = request.args.get('query')
    return render_template('template/Privacy.html')


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8080)
