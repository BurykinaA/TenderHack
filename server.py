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
    # scored = [(doc, score(query, doc)) for doc in documents]
    # scored = sorted(scored, key=lambda doc: -doc[1])
    # name, category, characteristic, proisvoditel
    # results = [doc.format(query)+['%.2f' % scr] for doc, scr in scored]
    results = [doc.format(query) for doc in documents]
    return render_template(
        'template/Search.html',
        query=query,
        results=results
    )


@app.route('/', methods=['GET'])
def index():
    return render_template('template/index.html')


@app.route('/Privacy.html', methods=['GET'])
def privacy():
    return render_template('template/Privacy.html')


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8080)
