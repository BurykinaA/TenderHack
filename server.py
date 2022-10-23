from flask import Flask, render_template, request, url_for
from search import score, retrieve, build_index, category_ans
from time import time


app = Flask(__name__, template_folder='')
build_index()


b_ = 0
ind_ = 0


@app.route('/Search.html', methods=['GET'])
def search():
    global b_, ind_
    start_time = time()
    query = request.args.get('query')
    startPrice = request.args.get('startPrice')
    endPrice = request.args.get('endPrice')
    if query is None:
        query = ''
    documents, category, b_, ind_ = retrieve(query, startPrice, endPrice)
    results = [doc.format(query) for doc in documents]
    return render_template(
        'template/Search.html',
        time="%.2f" % (time() - start_time),
        query=query,
        results=results,
        category=category.lower()
    )


@app.route('/Category.html', methods=['GET'])
def category():
    start_time = time()
    query = request.args.get('query')
    documents = category_ans(b_, ind_)
    results = [doc.format(query) for doc in documents]
    return render_template(
        'template/Category.html',
        time="%.2f" % (time() - start_time),
        query=query,
        results=results,
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
