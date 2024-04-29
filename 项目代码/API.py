# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import Info
import Nlp
import Data
import Forecast
import Analysis

app = Flask("my-app")

@app.route('/')
def start():
    answer = {}
    answer['flag']='success'
    return jsonify(answer)

@app.route('/info')
def info():
    answer = {}
    answer['flag'] = 'success'
    ans_text = Info.search_info("快讯")
    answer['ans'] = ans_text
    return jsonify(answer)

@app.route('/search_info')
def search_info():
    answer = {}
    answer['flag'] = 'success'
    word = request.args['word']
    ans_text = Info.search_info(word)
    answer['ans'] = ans_text
    return jsonify(answer)

@app.route('/nlp')
def word_nlp():
    answer = {}
    answer['flag'] = 'success'
    word = request.args['word']
    ans_text = Nlp.nlp(word)
    answer['ans'] = ans_text
    return jsonify(answer)

@app.route('/base_code')
def base_code():
    answer = {}
    answer['flag'] = 'success'
    ans_text = Data.base_code()[:500]
    answer['ans'] = ans_text
    return jsonify(answer)

@app.route('/search_code')
def search_stock():
    answer = {}
    answer['flag'] = 'success'
    word = request.args['word']
    ans_text = Data.search_code(word)[:500]
    answer['ans'] = ans_text
    return jsonify(answer)

@app.route('/all_data')
def all_data():
    answer = {}
    answer['flag'] = 'success'
    code = request.args['code']
    ans_text = Data.get_all_data(code)
    answer['ans'] = ans_text
    return jsonify(answer)

@app.route('/fore_base_code')
def fore_base_code():
    answer = {}
    answer['flag'] = 'success'
    ans_text = Forecast.base_code()[:500]
    answer['ans'] = ans_text
    return jsonify(answer)

@app.route('/fore_search_code')
def fore_search_stock():
    answer = {}
    answer['flag'] = 'success'
    word = request.args['word']
    ans_text = Forecast.search_code(word)[:500]
    answer['ans'] = ans_text
    return jsonify(answer)

@app.route('/fore_all_data')
def fore_all_data():
    answer = {}
    answer['flag'] = 'success'
    code = request.args['code']
    ans_text = Forecast.get_all_data(code)
    answer['ans'] = ans_text
    return jsonify(answer)

@app.route('/ana_base_code')
def ana_base_code():
    answer = {}
    answer['flag'] = 'success'
    ans_text = Analysis.get_all()
    answer['ans'] = ans_text
    return jsonify(answer)

@app.route('/ana_search_code')
def ana_search_stock():
    answer = {}
    answer['flag'] = 'success'
    word = request.args['word']
    ans_text = Analysis.search_code(word)
    answer['ans'] = ans_text
    return jsonify(answer)

@app.route('/ana_all_data')
def ana_all_data():
    answer = {}
    answer['flag'] = 'success'
    code = request.args['code']
    ans_text = Analysis.get_data(code)
    answer['ans'] = ans_text
    return jsonify(answer)

if __name__ == '__main__':
    app.config['JSON_AS_ASCII'] = False
    app.run(host='0.0.0.0', port=7021, debug=True)
    #app.run(ssl_context=('./Need/liuliang.plus_bundle.pem','./Need/liuliang.plus.key'),host='0.0.0.0', port=7021, debug=True)