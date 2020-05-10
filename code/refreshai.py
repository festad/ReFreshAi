import json
import os
from math import log, sqrt

import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def preprocessing_to_array(text):
    # text = text.decode("utf8")
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    stop = stopwords.words("english")
    min_chars = 3
    tokens = [token for token in tokens if token not in stop]
    tokens = [word for word in tokens if len(word) >= min_chars]
    tokens = [word.lower() for word in tokens]
    lmtzr = WordNetLemmatizer()
    tokens = [lmtzr.lemmatize(word) for word in tokens]
    return tokens


def reut_file_names(reut_dir):
    return [f for f in os.listdir(reut_dir)
            if os.path.isfile(os.path.join(reut_dir, f))
            and f.startswith("reut")]


def generate_all_json(reut_dir, json_dir):
    files_names = reut_file_names(reut_dir)
    doc_tpcs_wds = {}
    doc_tpcs = {}
    doc_wds = {}
    wd_docs = {}
    wd_tpcs = {}
    tpc_wds = {}
    wds = {}
    tpcs = {}
    for name in files_names:
        print(name)
        with open(reut_dir + "/" + name, "r") as fi:
            text = fi.read()
            soup = BeautifulSoup(text, "lxml")
            reuters = soup.find_all("reuters")
            for reut in reuters:
                if reut["lewissplit"] == "TRAIN" \
                        and reut["topics"] == "YES" \
                        and reut.content is not None \
                        and len(reut.topics.find_all("d")) >= 1:

                    tops = reut.topics.find_all("d")
                    pre_strtops = [s.get_text() for s in tops]
                    lower_classes = ["money-fx", "dlr", "yen", "dmk"]
                    strtops = []
                    for t in pre_strtops:
                        if t in lower_classes and "money" not in strtops:
                            strtops.append("money")
                        else:
                            strtops.append(t)

                    for t in strtops:
                        if t in tpcs:
                            tpcs[t] += 1
                        else:
                            tpcs[t] = 1

                    nid = reut["newid"]

                    doc_tpcs_wds[nid] = ([], [])
                    doc_tpcs[nid] = []

                    doc_wds[nid] = {}

                    for t in strtops:
                        doc_tpcs_wds[nid][0].append(t)
                        doc_tpcs[nid].append(t)

                    arr = preprocessing_to_array(reut.content.get_text())
                    for w in arr:
                        doc_tpcs_wds[nid][1].append(w)

                        if w in doc_wds[nid]:
                            doc_wds[nid][w] += 1
                        else:
                            doc_wds[nid][w] = 1

                        if w in wd_docs:
                            if nid in wd_docs[w]:
                                wd_docs[w][nid] += 1
                            else:
                                wd_docs[w][nid] = 1
                        else:
                            wd_docs[w] = {}
                            wd_docs[w][nid] = 1

                        if w in wd_tpcs:
                            for t in strtops:
                                if t in wd_tpcs[w]:
                                    wd_tpcs[w][t] += 1
                                else:
                                    wd_tpcs[w][t] = 1
                        else:
                            wd_tpcs[w] = {}
                            for t in strtops:
                                wd_tpcs[w][t] = 1

                        for t in strtops:
                            if t in tpc_wds:
                                if w in tpc_wds[t]:
                                    tpc_wds[t][w] += 1
                                else:
                                    tpc_wds[t][w] = 1
                            else:
                                tpc_wds[t] = {}
                                tpc_wds[t][w] = 1

                        if w in wds:
                            wds[w] += 1
                        else:
                            wds[w] = 1

    tpc_f = {}
    tf = {}
    tpc_sum = sum(tpcs.values())
    print("generating tpc_f and tf")
    for t in tpcs:
        tpc_f[t] = tpcs[t] / tpc_sum

        n_wds = sum(tpc_wds[t].values())
        tf[t] = {}
        for w in tpc_wds[t]:
            tf[t][w] = tpc_wds[t][w] / n_wds
    print("generating idf")
    idf = {}
    n_tpcs = len(tpcs)
    for w in wd_tpcs:
        idf[w] = log(n_tpcs / len(wd_tpcs[w]))
    print("calculating tf_idf")
    tf_idf = {}
    for t in tf:
        tf_idf[t] = {}
        for w in tf[t]:
            tf_idf[t][w] = tf[t][w] * idf[w]
    print("generating doc_f ")
    d_tf = {}
    for d in doc_wds:
        d_wds = sum(doc_wds[d].values())
        d_tf[d] = {}
        for w in doc_wds[d]:
            d_tf[d][w] = doc_wds[d][w] / d_wds
    print("generating idf for doc")
    d_idf = {}
    n_docs = len(doc_wds)
    for w in wd_docs:
        d_idf[w] = log(n_docs / len(wd_docs[w]))
    print("calculating tf_idf for docs")
    d_tf_idf = {}
    for d in d_tf:
        d_tf_idf[d] = {}
        for w in d_tf[d]:
            d_tf_idf[d][w] = d_tf[d][w] * d_idf[w]
    print("end of calculations")

    with open(json_dir + "/" + "doc_tpcs_wds.json", "w+") as fi:
        fi.write(json.dumps(doc_tpcs_wds, indent=4, sort_keys=True))

    with open(json_dir + "/" + "doc_tpcs.json", "w+") as fi:
        fi.write(json.dumps(doc_tpcs, indent=4, sort_keys=True))

    with open(json_dir + "/" + "wd_tpcs.json", "w+") as fi:
        fi.write(json.dumps(wd_tpcs, indent=4, sort_keys=True))

    with open(json_dir + "/" + "doc_wds.json", "w+") as fi:
        fi.write(json.dumps(doc_wds, indent=4, sort_keys=True))

    with open(json_dir + "/" + "wd_docs.json", "w+") as fi:
        fi.write(json.dumps(wd_docs, indent=4, sort_keys=True))

    with open(json_dir + "/" + "tpc_wds.json", "w+") as fi:
        fi.write(json.dumps(tpc_wds, indent=4, sort_keys=True))

    with open(json_dir + "/" + "wds.json", "w+") as fi:
        fi.write(json.dumps(wds, indent=4, sort_keys=True))

    with open(json_dir + "/" + "tpcs.json", "w+") as fi:
        fi.write(json.dumps(tpcs, indent=4, sort_keys=True))

    with open(json_dir + "/" + "tpc_f.json", "w+") as fi:
        fi.write(json.dumps(tpc_f, indent=4, sort_keys=True))

    with open(json_dir + "/" + "tf.json", "w+") as fi:
        fi.write(json.dumps(tf, indent=4, sort_keys=True))

    with open(json_dir + "/" + "idf.json", "w+") as fi:
        fi.write(json.dumps(idf, indent=4, sort_keys=True))

    with open(json_dir + "/" + "tf_idf.json", "w+") as fi:
        fi.write(json.dumps(tf_idf, indent=4, sort_keys=True))

    with open(json_dir + "/" + "d_tf.json", "w+") as fi:
        fi.write(json.dumps(d_tf, indent=4, sort_keys=True))

    with open(json_dir + "/" + "d_idf.json", "w+") as fi:
        fi.write(json.dumps(d_idf, indent=4, sort_keys=True))

    with open(json_dir + "/" + "d_tf_idf.json", "w+") as fi:
        fi.write(json.dumps(d_tf_idf, indent=4, sort_keys=True))


def naive_bayes(json_direct, pcd_w):
    with open(json_direct + "/" + "wd_tpcs.json") as f:
        json_string = f.read()
    wd_tpcs = json.loads(json_string)
    with open(json_direct + "/" + "tpc_wds.json") as f:
        json_string = f.read()
    tpc_wds = json.loads(json_string)
    with open(json_direct + "/" + "wds.json") as f:
        json_string = f.read()
    wds = json.loads(json_string)
    with open(json_direct + "/" + "tpc_f.json") as f:
        json_string = f.read()
    tpc_f = json.loads(json_string)
    smoother = len(wds)
    ptpc = {}
    for t in tpc_f:
        ptpc[t] = tpc_f[t]
    for t in tpc_f:
        for w in pcd_w:
            tmp_f = 0
            if w in wd_tpcs:
                tmp_f = wd_tpcs[w][t] if t in wd_tpcs[w] else 0
            tmp_p = (1 + tmp_f) / (sum(tpc_wds[t].values()) + smoother)
            ptpc[t] *= tmp_p
    sorted_t = {k: v for k, v in sorted(ptpc.items(), key=lambda item: item[1], reverse=True)}
    for k in sorted_t:
        return k


def cosine_distance(json_direct, preprocessed):
    with open(json_direct + "/" + "idf.json") as f:
        json_string = f.read()
    idf = json.loads(json_string)
    tstv = {}
    l_pr = len(preprocessed)
    for w in idf:
        if w in preprocessed:
            counter = 0
            for k in preprocessed:
                if w == k:
                    counter += 1
            w_f = counter / l_pr
            w_tfidf = w_f * idf[w]
            tstv[w] = w_tfidf
        else:
            tstv[w] = 0
    with open(json_direct + "/" + "tf_idf.json") as f:
        json_string = f.read()
    tf_idf = json.loads(json_string)
    ptpc = {}
    for t in tf_idf:
        tmp = {}
        for w in tf_idf[t]:
            tmp[w] = tstv[w] * tf_idf[t][w]
        n = sum(tmp.values())
        if n != 0:
            m_tstv = sqrt(sum([n * n for n in tstv.values()]))
            m_tf_idf = sqrt(sum([n * n for n in tf_idf[t].values()]))
            d = m_tf_idf * m_tstv
            ptpc[t] = n / d
        else:
            ptpc[t] = 0
    sorted_t = {k: v for k, v in sorted(ptpc.items(), key=lambda item: item[1], reverse=True)}
    # print(sorted_topic_probability)
    for k in sorted_t:
        return k


def kn_cosine_distance(json_direct, preprocessed):
    with open(json_direct + "/" + "d_idf.json") as f:
        json_string = f.read()
    d_idf = json.loads(json_string)
    tstv = {}
    l_pr = len(preprocessed)
    for w in d_idf:
        if w in preprocessed:
            counter = 0
            for k in preprocessed:
                if w == k:
                    counter += 1
            w_f = counter / l_pr
            w_tfidf = w_f * d_idf[w]
            tstv[w] = w_tfidf
        else:
            tstv[w] = 0
    with open(json_direct + "/" + "d_tf_idf.json") as f:
        json_string = f.read()
    d_tf_idf = json.loads(json_string)
    pd = {}
    for do in d_tf_idf:
        tmp = {}
        for w in d_tf_idf[do]:
            tmp[w] = tstv[w] * d_tf_idf[do][w]
        n = sum(tmp.values())
        if n != 0:
            m_tstv = sqrt(sum([n * n for n in tstv.values()]))
            m_tf_idf = sqrt(sum([n * n for n in d_tf_idf[do].values()]))
            d = m_tf_idf * m_tstv
            pd[do] = n / d
        else:
            pd[do] = 0
    sorted_d = {k: v for k, v in sorted(pd.items(), key=lambda item: item[1], reverse=True)}
    print(sorted_d)
    for doc in sorted_d:
        return doc


def test(reut_dir, json_dir, algorithm):
    files_names = reut_file_names(reut_dir)
    n_success = 0
    n_fail = 0
    with open(json_dir + "/" + "doc_tpcs.json") as f:
        json_string = f.read()
    doc_tpcs = json.loads(json_string)
    for name in files_names:
        print(name)
        with open(reut_dir + "/" + name, "r") as fi:
            text = fi.read()
            soup = BeautifulSoup(text, "lxml")
            reuters = soup.find_all("reuters")
            for reut in reuters:
                if reut["lewissplit"] == "TEST" \
                        and reut["topics"] == "YES" \
                        and reut.content is not None \
                        and len(reut.topics.find_all("d")) >= 1:
                    tops = reut.topics.find_all("d")
                    pre_strtops = [s.get_text() for s in tops]
                    lower_classes = ["money-fx", "dlr", "yen", "dmk"]
                    strtops = []
                    for t in pre_strtops:
                        if t in lower_classes and "money" not in strtops:
                            strtops.append("money")
                        else:
                            strtops.append(t)
                    if algorithm == "nb":
                        prediction = naive_bayes(json_dir, preprocessing_to_array(reut.content.get_text()))
                    elif algorithm == "cd":
                        prediction = cosine_distance(json_dir, preprocessing_to_array(reut.content.get_text()))
                    elif algorithm == "kn":
                        doc = kn_cosine_distance(json_dir, preprocessing_to_array(reut.content.get_text()))
                        predictions = doc_tpcs[doc]
                    else:
                        raise Exception
                    if algorithm == "nb" or algorithm == "cd":
                        if prediction in strtops:
                            n_success += 1
                            print("success")
                        else:
                            n_fail += 1
                            print("fail")
                    elif algorithm == "kn":
                        success = False
                        for p in predictions:
                            if p in strtops:
                                success = True
                        if success:
                            n_success += 1
                            print("success")
                        else:
                            n_fail += 1
                            print("fail")

    print("success {}, fail {}".format(n_success, n_fail))
    with open(json_dir + "/" + "test_result.json", "w+") as fi:
        fi.write("success {}, fail {}".format(n_success, n_fail))


def main():
    # reut_dir = "/home/denizu/PycharmProjects/ReFreshAi/reut"
    json_dir = "/home/denizu/PycharmProjects/ReFreshAi/json"
    # generate_all_json(reut_dir, json_dir)
    text = "zuccherifici"
    print(kn_cosine_distance(json_dir, preprocessing_to_array(text)))
    # test(reut_dir, json_dir, "kn")


if __name__ == '__main__':
    main()
