# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 16:10:34 2018

@author: Red
"""

import numpy as np
import matplotlib.pyplot as plt

def stop_words_remove(vocab_list):
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    for word in vocab_list:
        if word in stop_words:
            vocab_list.remove(word)
    
    return vocab_list

# 统计单词表词频
def vocab_freq_get(vocab_list, msg_array):
    words_list = []
    for i in msg_array:
        words_list += i
    
    freq_list = []
    for i in vocab_list:
        freq_list.append(words_list.count(i))
    
    return freq_list

# 移除前 num 个高频词
def vocab_list_create_remove_freq(msg_array, num=10):
    vocab_list = vocab_list_create(msg_array)
    freq_list = vocab_freq_get(vocab_list, msg_array)
    for i in range(num):
        index = freq_list.index(max(freq_list))
        #print(vocab_list[index])
        freq_list.pop(index)
        vocab_list.pop(index)
    return vocab_list

def vocab_list_create(msgs, remove_stop_word=False):
    vocab_set = set()
    for i in msgs:
        vocab_set |= set(i)
    
    vocab_list = sorted(list(vocab_set))
    if remove_stop_word == False:
        return vocab_list
    
    return stop_words_remove(vocab_list)

 # 如果处理的消息很长，那么应该遍历词汇表，否则应该遍历信息
def message2vec(vocab_list, msg):
    vec = np.zeros(len(vocab_list))
    for word in msg:
        if word in vocab_list:
            vec[vocab_list.index(word)] = 1
    return vec

# every row is a message vec 
def messages2vecs(vocab_list, msgs):
    msgs_len = len(msgs)
    shape = (msgs_len,len(vocab_list))
    matrix = np.zeros(shape)
    
    for i in range(msgs_len):
        for word in msgs[i]:
            if word in vocab_list:
                matrix[i,vocab_list.index(word)] = 1
    return matrix

def bag_message2vec(vocab_list, msg):
    vec = np.zeros(len(vocab_list))
    for word in msg:
        if word in vocab_list:
            vec[vocab_list.index(word)] += 1
    return vec

def bag_messages2vecs(vocab_list, msgs):
    msgs_len = len(msgs)
    shape = (msgs_len,len(vocab_list))
    matrix = np.zeros(shape)
    
    for i in range(msgs_len):
        for word in msgs[i]:
            if word in vocab_list:
                matrix[i,vocab_list.index(word)] += 1
    return matrix

sentences = ['I want to go to BeiJing', 'Watch the dog watch the dog']
def sentence2lists(sentences):
    msg_list = []
    for i in sentences:
        msg_list.append(i.lower().split())
    
    return msg_list

def test_sentence2lists():
    msg_list = sentence2lists(sentences)
    vocab_list = vocab_list_create(msg_list)
    msg_vecs = bag_messages2vecs(vocab_list, msg_list)
    print(vocab_list)
    print(msg_vecs)

def word_probability_vecs(msg_vecs, class_list):
    index_vec = np.array(class_list)
    prob_vecs = []
    for cls in set(class_list):
        cls_index = index_vec == cls
        cls_vecs = msg_vecs[cls_index,:]
        prob_vec = (np.sum(cls_vecs, axis=0) + 1) / (np.sum(cls_index) + 2)
        prob_vecs.append(prob_vec)
            
    return prob_vecs

def class_probability(class_list):
    cls_vec = np.array(class_list)
    total_msgs = len(class_list)
    
    cls_prob_vecs = []
    for cls in set(class_list):
        cls_prob = len(cls_vec[cls_vec==cls]) / total_msgs
        cls_prob_vecs.append(cls_prob)

    return cls_prob_vecs

# msg_vector is a vector
def naive_bayes_classifier(msg_vec, prob_vecs, cls_prob_vecs):
    ps = []
    
    for prob, cls_prob in zip(prob_vecs, cls_prob_vecs):
        p = np.sum(np.log(prob) * msg_vec) + np.log(cls_prob)
        ps.append(p)
    
    return ps.index(max(ps))

#cls = naive_bayes_classifier(message2vec(vocab_list, messages[1]), prob_vecs, cls_prob_vecs)

class NB():
    def __init__(self):
       pass

    def word_probability_vecs(self, msg_vecs, class_list):
        index_vec = np.array(class_list)
        prob_vecs = []
        for cls in set(class_list):
            cls_index = index_vec == cls
            cls_vecs = msg_vecs[cls_index,:]
            prob_vec = (np.sum(cls_vecs, axis=0) + 1) / (np.sum(cls_index) + 2)
            prob_vecs.append(np.log(prob_vec))

        return prob_vecs
    
    def class_probability(self, class_list):
        cls_vec = np.array(class_list)
        total_msgs = len(class_list)
        
        cls_prob_vecs = []
        for cls in set(class_list):
            cls_prob = len(cls_vec[cls_vec==cls]) / total_msgs
            cls_prob_vecs.append(np.log(cls_prob))
    
        return cls_prob_vecs
    
    def fix(self, train_msgs, train_class):
        # 生成分类集合
        self.class_set = set(train_class)
        self.class_num = len(self.class_set)
        self.class_array = np.array(list(self.class_set))
        
        # 生成单词表
        self.vocab_list = vocab_list_create(train_msgs)
        self.vocab_list = vocab_list_create_remove_freq_class(train_msgs, 
                                                              np.array(train_class), num=50)
        
        # 训练集留言转换为特征向量
        self.msg_vecs = messages2vecs(self.vocab_list, train_msgs)
        
        # 计算各分类上单词的条件概率 P(wk|ci)
        self.prob_vecs = self.word_probability_vecs(self.msg_vecs, train_class)
        
        # 计算各分类的先验概率 P(ci)
        self.cls_prob_vecs = self.class_probability(train_class)
        
    def predict(self, msgs):
        msgs_len = len(msgs)
        
        # 将信息列表转换为 2D array，每行对一特征向量
        predict_vecs = messages2vecs(self.vocab_list, msgs)
        
        # 生成 msgs_len * class_num 的数组，每一行对应在不同分类上的预测概率
        predict_array = np.zeros((msgs_len, self.class_num))

        for i in range(self.class_num):
            prob_vec = self.prob_vecs[i][:,np.newaxis] # transfrom to n*1
            predic_prob = predict_vecs.dot(prob_vec) + self.cls_prob_vecs[i] # msgs_len*1
            predict_array[:, i] = predic_prob[:,0]
        
        # 计算每一行上的概率最大索引
        index = np.argmax(predict_array, axis=1)

        # 通过索引获取分类信息
        return self.class_array[index]

    def predict_accurate(self, predicted_cls, label_cls):
        label_vec = np.array(label_cls)
        correct_num = np.sum(label_vec == predicted_cls)
        ratio = correct_num / len(predicted_cls)
        
        print("Predict accurate percent {}%".format(ratio * 100))
        return ratio

    def max_prob_words_get(self, prob_vec, num=10):
        d = {key:val for key,val in zip(range(len(prob_vec)), prob_vec)}
        d = sorted(d.items(), key=lambda x:x[1], reverse=True)
        
        max_prob_words = []
        for i in range(num):
            index = d[i][0]
            max_prob_words.append(self.vocab_list[index])
        
        return max_prob_words
    
    def show_max_prb_words(self, num=10):
        for i in range(self.class_num):
            words_list = self.max_prob_words_get(self.prob_vecs[i], num=num)
            print("class {} max probility words {}".format(self.class_array[i], words_list))
        
class BagNB(NB):
    def __init__(self):
       pass

    # P(wk|ci)
    def word_probability_vecs(self, msg_vecs, class_list, V):
        index_vec = np.array(class_list)
        prob_vecs = []
        for cls in set(class_list):
            cls_index = (index_vec == cls)
            cls_vecs = msg_vecs[cls_index,:]
            cls_total_words = np.sum(msg_vecs[cls_index,:])
            prob_vec = (np.sum(cls_vecs, axis=0) + 1) / (cls_total_words + V)
            prob_vecs.append(np.log(prob_vec))

        return prob_vecs
    
    # P(ci)
    def class_probability(self, msg_vecs, class_list):
        index_vec = np.array(class_list)
        total_words = np.sum(msg_vecs)
        
        cls_prob_vecs = []
        for cls in set(class_list):
            cls_index = index_vec == cls
            cls_total_words = np.sum(msg_vecs[cls_index,:])
            cls_prob = cls_total_words / total_words
            cls_prob_vecs.append(np.log(cls_prob))
    
        return cls_prob_vecs
    
    def fix(self, train_msgs, train_class):
        # 生成分类集合
        self.class_set = set(train_class)
        self.class_num = len(self.class_set)
        self.class_array = np.array(list(self.class_set))
        
        # 生成单词表
        self.vocab_list = vocab_list_create(train_msgs)
        
        # 训练集留言转换为特征向量
        self.msg_vecs = bag_messages2vecs(self.vocab_list, train_msgs)
        
        # 计算各分类上单词的条件概率 P(wk|ci)
        self.prob_vecs = self.word_probability_vecs(self.msg_vecs, train_class, 
                                                    len(self.vocab_list))
        
        # 计算各分类的先验概率 P(ci)
        self.cls_prob_vecs = self.class_probability(self.msg_vecs, train_class)
        
    def predict(self, msgs):
        msgs_len = len(msgs)
        
        # 将信息列表转换为 2D array，每行对一特征向量
        predict_vecs = messages2vecs(self.vocab_list, msgs)
        
        # 生成 msgs_len * class_num 的数组，每一行对应在不同分类上的预测概率
        predict_array = np.zeros((msgs_len, self.class_num))

        for i in range(self.class_num):
            prob_vec = self.prob_vecs[i][:,np.newaxis] # transfrom to n*1 
            predic_prob = predict_vecs.dot(prob_vec) + self.cls_prob_vecs[i] # msgs_len*1
            predict_array[:, i] = predic_prob[:,0]
        
        # 计算每一行上的概率最大索引
        index = np.argmax(predict_array, axis=1)
        
        # 通过索引获取分类信息
        return self.class_array[index]

    def predict_accurate(self, predicted_cls, label_cls):
        label_vec = np.array(label_cls)
        correct_num = np.sum(label_vec == predicted_cls)
        ratio = correct_num / len(predicted_cls)
        
        print("Predict accurate percent {}%".format(ratio * 100))
        return ratio

def test_bayes():
    messages =[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
           ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
           ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
           ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
           ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
           ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_list = [0, 1, 0, 1, 0, 1]    #1 is abusive, 0 not
    
    nb = BagNB()
    nb.fix(messages, class_list)

    test_messages = [['you', 'are', 'stupid'],
                     ['I', 'am', 'very', 'well']]
    test_labels = [1, 0]
    
    predicted_cls = nb.predict(test_messages)
    nb.predict_accurate(predicted_cls, test_labels)

'''
    corpus = [
        'This is the first &&*document.',
        'This is the second second document.',
        'And the third one. !!',
        'Is this the first document? <#$>',
        ]
'''
def msg_list2vects(msg_list):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()

    bag = vectorizer.fit_transform(msg_list)
    #vocab_sorted = sorted(vectorizer.vocabulary_.items(), key=lambda x:x[1], reverse=False)
    #print(vocab_sorted)
    return bag.toarray()

def msg2list(msg, ngram_range=(1,2)):
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(ngram_range=ngram_range)
    analyze = vectorizer.build_analyzer()

    return analyze(msg)

def shuffle(X, y, seed=None):
    idx = np.arange(X.shape[0])
    
    np.random.seed(seed)
    np.random.shuffle(idx)
    
    return X[idx], y[idx]

def load_email_msgs():
    words_list = []
    email_array,class_array = load_emails()
    for i in email_array:
        words_list.append(msg2list(i))
        
    words_array = np.array(words_list)
    return words_array,class_array

def load_emails():
    import os
    ham_mail_dir = r'db/email/ham/'
    spam_mail_dir = r'db/email/spam/'
    
    email_list = []
    file_list = os.listdir(ham_mail_dir)
    class_list = [0] * len(file_list)
    for i in file_list:
        with open(ham_mail_dir + i, "r", encoding='ISO-8859-1') as f:
            msg = f.read(-1)
            email_list.append(msg)

    file_list = os.listdir(spam_mail_dir)
    class_list += [1] * len(file_list)
    for i in file_list:
        with open(spam_mail_dir + i, "r", encoding='ISO-8859-1') as f:
            msg = f.read(-1)
            email_list.append(msg)
    
    email_array = np.array(email_list)
    class_array = np.array(class_list)

    return email_array,class_array

def test_email_nb_classifier(msg_array, class_array):
    msg_array, class_array = shuffle(msg_array, class_array)
    
    # split into train set and test set
    train_num = 40
    train_array = msg_array[0:train_num]
    train_class_list = list(class_array[0:train_num])
    test_array = msg_array[train_num:]
    test_class_list = list(class_array[train_num:])

    nb = NB()
    nb.fix(train_array, train_class_list)
    
    #nb.show_max_prb_words()
    predicted_cls = nb.predict(test_array)
    return nb.predict_accurate(predicted_cls, test_class_list)

def test_sklearn(email_array, class_array):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import BernoulliNB,MultinomialNB
    
    email_array, class_array = shuffle(email_array, class_array)
    
    # split into train set and test set
    train_num = 40
    train_array = email_array[0:train_num]
    train_class = class_array[0:train_num]
    test_array = email_array[train_num:]
    test_class = class_array[train_num:]

    '''
    # 去除高频词
    msg_array, class_array = load_email_msgs()
    vocab_list = vocab_list_create(msg_array)
    stop_words =  high_freq_stop_words_get(vocab_list, msg_array, class_array, num=50)
    '''
    
    # add stop_words='english' here
    vectorizer = CountVectorizer(stop_words=None)
    bag = vectorizer.fit_transform(train_array)
    #print(vectorizer.get_feature_names())
    
    with_tfidf = 0
    if with_tfidf:
        from sklearn.feature_extraction.text import TfidfTransformer
        tfidf_transformer = TfidfTransformer()
        train_vecs = tfidf_transformer.fit_transform(bag).toarray()

        # 生成测试集特征向量
        test_vecs = tfidf_transformer.transform(vectorizer.transform(test_array)).toarray()
    else:
        train_vecs = bag.toarray()
        # 生成测试集特征向量
        test_vecs = vectorizer.transform(test_array).toarray()

    clf = MultinomialNB()
    #clf = BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
    clf.fit(train_vecs, train_class)

    predicted_cls = clf.predict(test_vecs)
    #print(predicted_cls, test_class)
    correct_num = np.sum(test_class == predicted_cls)
    
    return correct_num / len(predicted_cls)

def scikit_average_test(test_times=100):
    score = 0.0
    
    email_array, class_array = load_emails()
    for i in range(test_times):
        score += test_sklearn(email_array, class_array)

    print("Predict average accurate percent {:.2f}%".format(score / test_times * 100))

def average_test(test_times=100):
    score = 0.0
    
    msg_array, class_array = load_email_msgs()
    for i in range(test_times):
        score += test_email_nb_classifier(msg_array, class_array)

    print("Predict average accurate percent {:.2f}%".format(score / test_times * 100))
    
def max_freq_index_get(freq_list, num=10):
    d = {key:val for key,val in zip(range(len(freq_list)), freq_list)}
    d = sorted(d.items(), key=lambda x:x[1], reverse=True)
    
    max_freq_index = []
    for i in range(num):
        max_freq_index.append(d[i][0])
    
    return max_freq_index

# 返回高频停止词：不同分类中前 num 高频词中的交集
def high_freq_stop_words_get(vocab_list, msg_array, class_array, num=50):
    freq_list_c0 = vocab_freq_get(vocab_list, msg_array[class_array==0])
    freq_list_c1 = vocab_freq_get(vocab_list, msg_array[class_array==1])
    
    high_freq_c0_index = max_freq_index_get(freq_list_c0, num=num)
    high_freq_c1_index = max_freq_index_get(freq_list_c1, num=num)
    
    high_freq_words = []
    both_freq_index_set = set(high_freq_c0_index).intersection(set(high_freq_c1_index))
    for i in both_freq_index_set:
        high_freq_words.append(vocab_list[i])
    
    return high_freq_words

# 移除不同分类中前 num 高频词中的交集词汇
def vocab_list_create_remove_freq_class(msg_array, class_array, num=50):
    vocab_list = vocab_list_create(msg_array)
    
    high_freq_words = high_freq_stop_words_get(vocab_list, msg_array, 
                                               class_array, num=num)
    for word in high_freq_words[:num]:
        vocab_list.remove(word)
    
    return vocab_list

def high_freq_stop_words_list(num=50):
    msg_array, class_array = load_email_msgs()
    vocab_list = vocab_list_create(msg_array)
    
    return high_freq_stop_words_get(vocab_list, msg_array, class_array, num=num)

if __name__ == "__main__":
    scikit_average_test(100)
    #average_test(100)

