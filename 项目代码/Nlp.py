import xmnlp

xmnlp.set_model('./xmnlp-onnx-models-v4/xmnlp-onnx-models')

def nlp(word):
    answer = []
    key = xmnlp.keyword(word,k=5,stopword=True)
    ans = ""
    for words in key:
        ans= ans + words[0] + "；"
    answer.append(ans)
    sen = xmnlp.keyphrase(word,k=1)
    answer.append(sen[0])
    tag = xmnlp.sentiment(word)
    if(tag[0]>=tag[1]):
        answer.append("消极情绪")
    else:
        answer.append("积极情绪")
    answer.append(str(tag[0]))
    answer.append(str(tag[1]))
    return answer