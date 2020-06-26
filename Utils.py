"""*****************************************************************************************
MIT License
Copyright (c) 2019 Ibrahim Jubran, Murad Tukan, Alaa Maalouf, Dan Feldman
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*****************************************************************************************"""

import numpy as np
import PointSet
from scipy.linalg import null_space
import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation
import pandas as pd
import time
import os
import glob

from sklearn.datasets.samples_generator import make_blobs



K = 2
ANIMATE = False
NUM_SAMPLES = 8
M = 2
DIMF = K * M
TAU = 1.0 / 24.0
GAMMA = 1.0 / (2.0 * K)
R_EARTH = 6371.0  # KM
MED_SUBSAMPLE = True
NUM_INIT = 100
REPS = 10
PARALLELIZE = False
FROM_INPUT = True
Z = 2.0
READ_RANDOM_ROWS_DATA = True
OPTIMIZE_ON_ALL_DATA = False
FILE_NAME = 'RandomData'
CHECK_TILL_CONVERGED = 3
KMEANS_USE_SUB_SAMPLE=True

CONSIDER_HISTORY = False

Unifrom_History = []
Our_History = []


def resetHistory(consider=False):
    global CONSIDER_HISTORY, Unifrom_History, Our_History
    CONSIDER_HISTORY = consider
    Unifrom_History = []
    Our_History = []

NUM_THREADS = 4

color_matching = {
    'Our coreset': 'red',
    'Uniform subsampling': 'blue',
    'All data': 'black'
}


def changeK(k=2.0):
    global K, GAMMA, DIMF
    K = k
    GAMMA = 1.0 / (2.0 * K)
    DIMF = K * M


robust_median_sample_size = (lambda x, n=1e10: int(1.0 / (TAU ** 4.0 * GAMMA ** 2.0) * (x + np.log(n))))

fraction_lambda = (lambda n, gamma: np.ceil(n * gamma))

list_of_images = []


def computeEps(approx_val, opt_val):
    return approx_val / opt_val - 1.0


def generateMPointsAlongVector(vector, m, translation, random_start, random_stop):
    # random_start = np.random.randint(20, 100)
    # random_stop = np.random.randint(random_start, 10.0*random_start)
    coeffs = np.linspace(random_start, random_stop, num=m, dtype=np.float)
    return np.multiply(np.expand_dims(coeffs, 1), vector) + translation


def getPointOnUnitBall(n=int(5), d=2, r=10.0):
    theta = np.expand_dims(np.arange(start=0, step=2.0*np.pi / n, stop=2.0*np.pi + 1.0/n), 1).T
    return np.vstack((r*np.cos(theta), r*np.sin(theta)))

def getPointOnUnitBallOLD(n=int(5), d=2):
    theta = np.expand_dims(np.arange(start=0, step=1.0 / n, stop=2.0*np.pi + 1.0/n), 1).T
    return np.vstack((np.cos(theta), np.sin(theta)))

def generateMPointsAlongVector2(vector, m, r=60.0):
    unit_v = vector / np.linalg.norm(vector)
    coeffs = np.array([r * i for i in range(m)])
    return np.multiply(np.expand_dims(coeffs, 1), unit_v) + vector


def createFlowerDataset(n=9910, d=2, m = M, r = 1.0):
    A = np.abs(np.random.randn(n, d))
    # A = np.array(map(lambda row: row / np.linalg.norm(row), A))
    A = getPointOnUnitBall(n, d).T
    N = np.ma.size(A, 0)

    # translate = -50.0 * np.random.rand(d,)
    # random_start = np.random.randint(1, 2)
    # random_stop = np.random.randint(random_start, 3 * random_start)
    setP = []
    P1 = getPointOnUnitBall(n, d, 1).T
    setP += np.apply_along_axis(lambda x: PointSet.PointSet(
        generateMPointsAlongVector2(x, m, 2)), axis=1, arr=P1).tolist()

    # print(len(setP))
    # setP = np.apply_along_axis(lambda x: PointSet.PointSet(
    #     generateMPointsAlongVector(x, m, np.zeros((d,)),random_start, 2*random_stop)), axis=1, arr=A)
    # setQ = np.apply_along_axis(lambda x: PointSet.PointSet(
    #     generateMPointsAlongVector(x - 100*np.ones((d,)), m, np.zeros((d,)), random_start, random_stop)), axis=1,
    #                            arr=A[M + 1: N-1, :])

    # outliers = np.vstack((np.array([1e6, 1e6]), np.array([1100000, 2e6])))

    n_out = 90

    P2 = getPointOnUnitBall(n_out, d, 0.1).T
    setP += np.apply_along_axis(lambda x: PointSet.PointSet(
        generateMPointsAlongVector2(x, m, 0.2) + r * np.array([32.0, 32.0])), axis=1, arr=P2).tolist()

    return setP




def createFlowerDatasetOLD(n=3154, d=2, m=M):
    A = np.abs(np.random.randn(n, d))
    A = np.array(map(lambda row: row / np.linalg.norm(row), A))
    A = getPointOnUnitBall(n, d).T
    N = np.ma.size(A, 0)
    M = N // 2

    # translate = -50.0 * np.random.rand(d,)
    random_start = np.random.randint(1, 2)
    random_stop = np.random.randint(random_start, 3 * random_start)
    setP = np.apply_along_axis(lambda x: PointSet.PointSet(
        generateMPointsAlongVector(x, m, np.zeros((d,)), 0.5 * random_start, random_stop)), axis=1, arr=A[0:M, :])
    # setQ = np.apply_along_axis(lambda x: PointSet.PointSet(
    #     generateMPointsAlongVector(x - 100*np.ones((d,)), m, np.zeros((d,)), random_start, random_stop)), axis=1,
    #                            arr=A[M + 1: N-1, :])

    # outliers = np.vstack((np.array([1e6, 1e6]), np.array([1100000, 2e6])))

    n_out = 90
    for i in range(n_out):
        setP = np.hstack((setP, PointSet.PointSet(np.vstack((np.array([1e6 + 1.0/n_out * i, 1e6]),
                                                             np.array([1100000 + 10.0/n_out * i, 2e6]))))))

    # setP = np.hstack((setP, PointSet.PointSet(np.vstack((np.array([1e6, 1e6]), np.array([1100000, 2e6]))))))
    # setP = np.hstack((setP, PointSet.PointSet(np.vstack((np.array([1000025, 1e6]), np.array([1100080, 2e6]))))))

    # setP = np.hstack((setP, []))

    print('Number of total points is {}'.format(len(setP)))
    return setP


def plotPoints(set_P):
    fig = plt.figure()
    ax = fig.add_subplot(111)#, projection='3d')
    for P in set_P:
        center = np.average(P.P, axis=0)
        R = np.max(np.linalg.norm(P.P - center, axis=0))
        radii = np.array([R, min(R/15.0, 10)])

        v = np.expand_dims(P.P[0, :], 1)
        v = v / np.linalg.norm(v)
        rotation = np.hstack((v, null_space(v.T)))

        # plot points
        ax.scatter(P.P[:, 0], P.P[:, 1], color='r', marker='o', s=30)
        ax.can_zoom()
        # plot ellipsoid
        plotEllipsoid(radii, rotation, center, ax)
        # plt.show()

    plt.show()


def createRandomData(n=50000,d=2,m=1):
    P = [PointSet.PointSet(np.random.rand(m,d)) for i in range(n)]
    return P


def generateSamples(n):
    min_val = 20  # int(math.log(n))
    max_val = 200  # int(n)
    step_size = 20
    samples = range(min_val, max_val , step_size)
    samples = [10,20,30,40,50, 60,70,80,90,100]
    # samples = np.geomspace(min_val, max_val, NUM_SAMPLES)
    return samples


def plotEllipsoid(radii, V, center, ax):
    N = 20
    # _, D, V = sp.linalg.svd(ellipsoid, full_matrices=True)
    a = radii[0]
    b = radii[1]
    theta = np.expand_dims(np.arange(start=0, step=1.0 / N, stop=2.0*np.pi + 1.0/N), 1).T

    state = np.vstack((a * np.cos(theta), b * np.sin(theta)))
    X = np.dot(V, state) + np.expand_dims(center, 1)

    ax.plot(X[0, :], X[1, :], color='blue')
    ax.grid(True)


def makeAnimation(fig, list_imgs):
    ani = animation.ArtistAnimation(fig, list_imgs, interval=50, blit=True,
                                    repeat_delay=1000)

    ani.save('test.mp4')


def convertLongLatTwo3D(lon, lat):
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    x = R_EARTH * np.cos(lat) * np.cos(lon)
    y = R_EARTH * np.cos(lat) * np.sin(lon)
    z = R_EARTH * np.sin(lat)
    return np.array([x,y,z])


def union2(dict1, dict2):
    return dict(list(dict1.items()) + list(dict2.items()))


def convertHWRowToMSet(row, header_columns):
    home_data = row[[x for x in header_columns if 'home' in x]].to_numpy()
    work_data = row[[x for x in header_columns if 'work' in x]].to_numpy()
    data = np.vstack((home_data, work_data))
    return PointSet.PointSet(data)


def createNMSetForHWData(s = 5000):
    global FILE_NAME
    start_time = time.time()
        
    filename = "../dataset2/HWData.csv"
        
    n = sum(1 for line in open(filename)) - 2  # number of records in file (excludes header)
   
    #skip = sorted(np.random.choice(range(1, n + 1), n - s, False))  # the 0-indexed header will not be included in the skip list
    #input_data = pd.read_csv(filename, skiprows=skip, low_memory=False, header=0)
    input_data = pd.read_csv(filename, low_memory=False, header=0)

    # input_data = pd.read_csv('CaliHWData.csv', low_memory=False, header=0, nrows=50000) #nrows=100000, float_precision='round_trip')
    #input_data = input_data.drop(input_data.index[input_data['home_x'] == 'home_x'], axis=0)
    raw_data = input_data.values
    n_m_set = np.array([PointSet.PointSet(np.vstack((x, y))) for x, y in zip(raw_data[:, 1:3].astype(np.float),
                                                                             raw_data[:, 4:6].astype(np.float))])

    print('Preprocessed data in {:.4f} seconds'.format(time.time() - start_time));  
    FILE_NAME = 'HWDataCali'
    return n_m_set


def alaaData():
    n = 5000
    d = 5

    centers = [(-500, -500, -500, -500, -500), (500, 500, 500, 500, 500), (-1000, -1000, -1000, -1000, -1000),
               (1000, 1000, 1000, 1000, 1000)]
    cluster_std = [100, 100, 100, 100]

    X, y = make_blobs(n_samples=n, cluster_std=cluster_std, centers=centers, n_features=d, random_state=1)
    # X2, y2 = make_blobs(n_samples=n, cluster_std=cluster_std, centers=centers, n_features=d, random_state=1)

    P = [PointSet.PointSet(X[i].reshape(1, -1)) for i in range(n)]

    return P



# n_m_set = np.array([])
# header_columns = list(input_data.columns)
#
# start_time = time.time()
# for _, line in input_data.iterrows():
#     home_data = line[[x for x in header_columns if 'home' in x]].to_numpy()
#     work_data = line[[x for x in header_columns if 'work' in x]].to_numpy()
#     data = np.vstack((home_data, work_data))
#     n_m_set= np.hstack((n_m_set, PointSet.PointSet(data)))
#
# print('First Approach Finished in {:.4f} seconds'.format(time.time() - start_time))
#
# start_time = time.time()
#
# P = input_data.apply(lambda row: convertHWRowToMSet(row, header_columns), axis=1, raw= True)
#
# print('Second Approach Finished in {:.4f} seconds'.format(time.time() - start_time))
#


def computeRelativeSizes(sample_size=10000):
    sizes = []; paths = "../dataset2/*_2010.csv"
    for fname in glob.glob(paths):
        state = fname.split('_')[0]
        print(state)
        lookup_data = pd.read_csv(state+'_xwalk.csv', low_memory=False, index_col=False, usecols=['tabblk2010', 'blklatdd', 'blklondd'], header=0)
        raw_data = pd.read_csv(fname, low_memory=False, index_col=False, usecols=['w_geocode', 'h_geocode'], header=0)
        N,_ = raw_data.shape
        sizes.append(N)
    #print('Finished sizes')
    sizes = np.array(sizes, dtype=np.float)
    relateive_sizes = sizes / np.sum(sizes)
    print('sizes = {}, relateive_sizes={}'.format(sizes, relateive_sizes))
    return sizes, np.ceil(relateive_sizes * sample_size)
    
    
    #return sizes, sizes
    
    
    
    
    

def readGeoCode():
    paths = "../dataset2/*_2010.csv"
    header = True
    Ns, sizes = computeRelativeSizes()
    print('sizes = {}, samples = {}'.format(Ns, sizes))
    sizes[0] = 300
    i = 0
    for fname in glob.glob(paths):
        state = fname.split('_')[0]     
        print('State = {}'.format(state))
        lookup_data = pd.read_csv(state+'_xwalk.csv', low_memory=False, index_col=False, usecols=['tabblk2010', 'blklatdd', 'blklondd'], header=0)
        skip = sorted(np.random.choice(range(1 + int(Ns[i])), int(Ns[i] - sizes[i]), False))
        if 0 in skip: skip.remove(0)
        #print (fname)
        raw_data = pd.read_csv(fname, low_memory=False, skiprows=skip, index_col=False, usecols=['w_geocode', 'h_geocode'], header=0)
        #N,_ = raw_data.shape
        converted_data = pd.DataFrame()

        rows = []

        j = 0
        i += 1
        for _, line in raw_data.iterrows():
            geocodes = line['w_geocode'], line['h_geocode']
            row_in_df = {}
            for idx, item in enumerate(geocodes):
                #print('idx: = {}, item={}'.format(idx, item))
                #print('{}'.format(lookup_data.loc[lookup_data['tabblk2010'] == float(item)][['blklatdd', 'blklondd']]))
                geo_data = np.array(lookup_data.loc[lookup_data['tabblk2010'] == float(item)][['blklatdd', 'blklondd']],
                                    dtype=np.float) 
                #print (geo_data)
                point = convertLongLatTwo3D(geo_data[0][0], geo_data[0][1])
                if idx == 0:
                    row_in_df = union2(row_in_df, dict(zip(['work_' + x for x in ['x','y','z']], point)))
                else:
                    row_in_df = union2(row_in_df, dict(zip(['home_' + x for x in ['x', 'y', 'z']], point)))

            rows.append(row_in_df)
            j = j + 1
            #if j % 1000 == 0:
            #   database = pd.DataFrame.from_dict(rows, orient='columns')
            #   database.to_csv('HWData.csv', sep=',', header=header, mode='a')
            #   print('Progress on State {}: {:.4f}%'.format(state, j / N))
            #   rows = []
            #   header = False

        database = pd.DataFrame.from_dict(rows, orient='columns')
        database.to_csv('HWData.csv', sep=',', header=header, mode='a')
        header = False



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                                    Handlning Documents
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
from nltk.corpus import stopwords, reuters
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import functools, operator

from sklearn.decomposition import TruncatedSVD
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re

DESIRED_DIMENSION_FOR_DOCS = 20
cachedStopWords = stopwords.words("english")


def tokenize(text):
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(text))
    words = [word for word in words if word not in cachedStopWords]
    tokens = (list(map(lambda token: PorterStemmer().stem(token),
                       words)))
    p = re.compile('[a-zA-Z]+')
    filtered_tokens = list(filter(lambda token: p.match(token) and len(token) >= min_length, tokens))
    return filtered_tokens


def preprocessDocuments():
    stop_words = stopwords.words("english")

    # List of document ids
    documents = reuters.fileids()

    train_docs_id = list(filter(lambda doc: doc.startswith("train"),
                                documents))
    test_docs_id = list(filter(lambda doc: doc.startswith("test"),
                               documents))

    all_docs_id = train_docs_id + test_docs_id

    docs = [reuters.raw(doc_id) for doc_id in all_docs_id]

    docs_by_paragraph = [x.split('\n      ') for x in docs]
    num_paragraph_per_doc = [len(x) for x in docs_by_paragraph]

    all_combined_paragraphs = functools.reduce(operator.iconcat, docs_by_paragraph, [])

    # Tokenisation
    vectorizer = TfidfVectorizer(stop_words=stop_words,
                                 tokenizer=tokenize)

    # Learn and transform train documents
    vectorized_paragraphs = vectorizer.fit_transform(all_combined_paragraphs)

    clf = TruncatedSVD(n_components=DESIRED_DIMENSION_FOR_DOCS, n_iter=7, random_state=42)

    vectorized_paragraphs_lower_dim = clf.fit_transform(vectorized_paragraphs.toarray())
    set_P = []

    np.savez('reutersData', data=vectorized_paragraphs_lower_dim, num_paragraphs=num_paragraph_per_doc)


def readReutersDocuments():
    global FILE_NAME
    start_time = time.time()
    doc_data = np.load('reutersData.npz')
    data = doc_data['data']
    num_paragraphs = doc_data['num_paragraphs']

    j=0
    set_P = []
    for num_paragraphs_per_doc in num_paragraphs:
        idxs = list(range(j, j+num_paragraphs_per_doc, 1))
        if len(idxs) <= 4:
            set_P += [PointSet.PointSet(data[idxs, :])]
        j += num_paragraphs_per_doc

    print('Number of documents: {}'.format(len(set_P)))
    print('Converting preprocessed documents into nm-sets took {:.3f} seconds'.format(time.time() - start_time))
    FILE_NAME = 'ReutersDocuments'
    return set_P



"""
Utility Function
"""


def createDirectory(dir_name):
    try:
        # Create target Directory
        os.mkdir(dir_name)
        print("Directory ", dir_name, " Created ")
    except FileExistsError:
        print("Directory ", dir_name, " already exists")


if __name__ == '__main__':
    # set_P = createFlowerDataset()
    # plotPoints(set_P)
    # readGeoCode()
    # createNMSetForHWData()
    #preprocessDocuments()
    readReutersDocuments()