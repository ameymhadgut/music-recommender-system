import pandas as pd
import numpy as np
import ast
from scipy.spatial import distance
import time
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt
from annoy import AnnoyIndex

def build_annoy(item_vectors, num_trees):
    f = len(item_vectors[0])
    t = AnnoyIndex(f, 'dot')  # Length of item vector that will be indexed
    for i in range(len(item_vectors)):
        t.add_item(i, item_vectors[i])
    t.build(num_trees) # 10 trees
    return t

def getNumpyArray(values):
    return np.array(list([ast.literal_eval(curr[0]) for curr in values]))

def search_bf(given_vec, search_space, k):
    startTime = time.time()
    result = np.argsort(-search_space.dot(given_vec))[:k]
    return result, time.time() - startTime

#Linear Search - https://en.wikipedia.org/wiki/Nearest_neighbor_search#Linear_search
def recommendForUserSubset_bf(user_vectors, item_vectors, num_of_rec):
    result_all = list()
    exe_time_all = list()
    for user in tqdm(user_vectors):
        res, exe_time = search_bf(user, item_vectors, 500)
        result_all.append(res)
        exe_time_all.append(exe_time)
    return result_all, exe_time_all

def search_ay(t, given_vec, num_of_rec):
    startTime = time.time()
    result = t.get_nns_by_vector(given_vec, num_of_rec)
    return result, time.time() - startTime

def recommendForUserSubset_ay(t, user_vectors, item_vectors, num_of_rec):
    result_all = list()
    exe_time_all = list()
    for user in tqdm(user_vectors):
        res, exe_time = search_ay(t, user, 500)
        result_all.append(res)
        exe_time_all.append(exe_time)
    return result_all, exe_time_all

def recall(ground_truth, predicited):
    ground_truth = set(ground_truth)
    predicited = set(predicited)
    return len(predicited.intersection(ground_truth))/len(predicited)

def get_stats(result_all_bf, result_all_ay, exe_time_all_bf, exe_time_all_ay):
    N = len(result_all_bf)
    total_recall = 0
    avg_exe_time_ay = sum(exe_time_all_ay)/N
    avg_exe_time_bf = sum(exe_time_all_bf)/N
    for i in range(0, N):
        total_recall += recall(result_all_bf[i], result_all_ay[i])
    avg_recall_per_query = total_recall/N
    avg_exe_time_per_query = [avg_exe_time_bf, avg_exe_time_ay]
    return avg_recall_per_query, avg_exe_time_per_query

def benchmark(t, user_vectors, item_vectors, num_of_search): #, 1000, 10000, 100000]):
    recall_list = list()
    queries_per_sec_list_bf = list()
    queries_per_sec_list_ay = list()
    for num in num_of_search:
        users = user_vectors[:num]
        result_all_bf, exe_time_all_bf = recommendForUserSubset_bf(users, item_vectors, 500)
        result_all_ay, exe_time_all_ay = recommendForUserSubset_ay(t, users, item_vectors, 500)
        avg_recall_per_query, avg_exe_time_per_query = get_stats(result_all_bf, result_all_ay, exe_time_all_bf, exe_time_all_ay)
        recall_list.append(avg_recall_per_query)
        queries_per_sec_list_bf.append(avg_exe_time_per_query[0])
        queries_per_sec_list_ay.append(avg_exe_time_per_query[1])
    return recall_list, queries_per_sec_list_bf, queries_per_sec_list_ay

def plot_results(recall_list, queries_per_sec_list_bf, queries_per_sec_list_ay, num_of_search):
    # Time Complexity
    plt.plot(num_of_search, queries_per_sec_list_bf, label="Using Brute Force (linear search)")
    plt.plot(num_of_search, queries_per_sec_list_ay, label="Using Annoy Time")
    plt.xlabel('Number of queries')
    plt.ylabel('Time')
    plt.legend(loc="upper left")
    plt.suptitle('Time Complexity Comparison', fontsize=10)
    plt.show()
    
    queries_per_sec_list_bf = [1/i for i in queries_per_sec_list_bf]
    queries_per_sec_list_ay = [1/i for i in queries_per_sec_list_ay]
    # Recall - queries per second
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.semilogy([1] * len(num_of_search), queries_per_sec_list_bf, label="Using Brute Force")
    ax.semilogy(recall_list, queries_per_sec_list_ay, label="Using Annoy")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Queries per second (1/s)")
    ax.legend(loc="upper left")
    plt.suptitle('Recall-Queries per second (1/s) tradeoff - up and to the right is better', fontsize=10)