# coding:utf-8

import logging
from pprint import pprint

# first extract the 20 news_group dataset to /scikit_learn_data
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import HashingVectorizer

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# all categories
newsgroup_train = fetch_20newsgroups(subset='train')

# part categories
print '+++++++++++++++++++++++++++ loading data ...'
categories = ['comp.graphics',
              'comp.os.ms-windows.misc',
              'comp.sys.ibm.pc.hardware',
              'comp.sys.mac.hardware',
              'comp.windows.x']
newsgroup_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroup_test = fetch_20newsgroups(subset='test', categories=categories)

# print category names
pprint(list(newsgroup_train.target_names))


print '+++++++++++++++++++++++++++ extract features ...'
# newsgroup_train.data is the original documents, but we need to extract the
# feature vectors inorder to model the text data
vectorizer = HashingVectorizer(stop_words='english', non_negative=True,
                               n_features=10000)
fea_train = vectorizer.fit_transform(newsgroup_train.data)
fea_test = vectorizer.fit_transform(newsgroup_test.data)


# return feature vector 'fea_train' [n_samples,n_features]
print 'Size of fea_train:' + repr(fea_train.shape)
print 'Size of fea_train:' + repr(fea_test.shape)
# 11314 documents, 130107 vectors for all categories
print 'The average feature sparsity is {0:.3f}%'.format(
    fea_train.nnz / float(fea_train.shape[0] * fea_train.shape[1]) * 100)
