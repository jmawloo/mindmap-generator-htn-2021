# Works on python 3.8.1, not tested on higher versions.

import tensorflow_hub as hub
import tensorflow as tf
# !NOTE: You may receive import errors on first run.
# Ensure the following 2 lexicons are also imported if you receive errors:
# stopwords
# (I forgot the other one sorry, but it should come up during build which command to run).

import numpy as np
# import os
# import pandas as pd
from collections import defaultdict, OrderedDict, deque
# import re

from rake_nltk import Rake
import spacy

# With help from https://stackoverflow.com/a/62027959
# also run python -m spacy download en_core_web_sm
noun_filter = spacy.load('en_core_web_sm')
whitelist = {"NOUN", "VERB", "ADJ", "ADV", "ADP", "PROPN"}
# "ADP", "PROPN"

# import re
# import seaborn as sns

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)

""" SETUP: """
# model = tf.keras.models.load_model('./model')
rake_inst = Rake()

# Environment variables / arguments:
numinputs = 8
# cutoff_corr_val = 0.5   # Anything below this will not have edges created for it.

# Preprocessor variables.
# Longest and shortest character counts allowed before line splits are applied.
mincharlen = 0
longest_para_chr_cnt = 500
shortest_para_chr_cnt = 100
topic_len_chr_cnt = 30 # NOTE: Don't use this? Just combine with other string?

max_keyword_count = 8


def preprocess(text):
    # Takes in text and parse the contents into reasonable sentences. 
    # One heuristic: Treat short phrases as topics. 
    global mincharlen, longest_para_chr_cnt, shortest_para_chr_cnt
    global topic_len_chr_cnt

    ignorechars = {'\n', '\t'}
    splitonchars = {'.', '?', '!'}

    count = 0
    lines, temp = [], []
    for c in text:
        temp.append(' ' if c in ignorechars else c)
        count += 1
        
        # Split string or between  shortest_para_chr_cnt and longest_para_chr_cnt
        # NOTE: also split on newlines if total count is greater than topic lenghth
        if (shortest_para_chr_cnt <= count <= longest_para_chr_cnt) and c in splitonchars or c == '\n' and count > topic_len_chr_cnt:
            count = 0
            lines.append(''.join(temp))
            temp = []

    # lines = list(filter(lambda l: len(l) > mincharlen, [l.strip("\t\n ") for l in file.readlines()]))
    return lines

def get_corr(text):
    text_embedded = model(text)
    corr = np.inner(text_embedded, text_embedded)
    return corr

def get_keywords(text, count=-1, maxwords=-1):
    rake_inst.extract_keywords_from_text(text) # void

    extracted_keys = [(e[1], 1) for e in rake_inst.get_ranked_phrases_with_scores()]
    # scores, extracted_keys = [e[0] for e in extracted_keys_scores[e[1] for e in extracted_keys_scores]
    # Return keywords with max word count of maxwords 
    #-1 by default includes all keywords.
    res = list(filter(lambda x: len(x[0].split()) <= maxwords if maxwords >= 0 else True, extracted_keys))[:count]

    # Prune keywords to not contain non-standard characters in words.
    allowed_special_chars = {' ', '-'}
    res = [(''.join(filter(lambda y: y in allowed_special_chars or y.isalnum(), s[0])), s[1]) for s in res]

    # Remove any keyword that's not a noun, adjective, verb:
    res = [(' '.join(map(lambda u: u.text, filter(lambda t: t.pos_ in whitelist, noun_filter(s[0])))), s[1]) for s in res]
    res = list(filter(lambda x: x[0] not in {" ", ""}, res))
    # print(res)

    return OrderedDict(res)

def get_maxspantree_edges(matrix):
    """Following part filters out edges to include the minimum to make a spanning tree:
        TODO: Modularize (function)
    """
    # 1) Assemble all edges 
    # Traverse correlation multi-array, only considering upper half of diagonal:
    edges = []
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            edges += [(i, j, matrix[i][j])]

    # Sort edges by weight largest first:
    edges.sort(key=lambda x: x[2], reverse=False) # I was wrong. Minimum spanning tree is the way to go. 
    # print(edges)
    
    #2) Run kruskal's algorithm here for MINIMUM spanning tree
    # TODO: Use disjoint sets for an O(mlogm) algo.
    # Naive solution uses a suboptimal O(mlogm + n^2) algo
    
    # cost = 0
    tree_set = [i for i in range(n)]
    tree_edges = [] # Our answer.

    #https://cp-algorithms.com/graph/mst_kruskal.html
    # Greedy implementation

    for e in edges:
        u, v, weight = e
        if tree_set[u] == tree_set[v]: # Skip connected nodes.
            continue
        tree_edges.append(e)

        old_id, new_id = tree_set[u], tree_set[v]

        for i in range(n):
            if tree_set[i] == old_id:
                tree_set[i] = new_id

    return tree_edges


class TreeStruct:
    def __init__(self, id, text):
        self.id = id
        self.text = text                   # Original text
        # self.keywords = self.parse_keywords(-1, -1) 
        self.children = []                 # children nodes. Empty if leaf.

    def parse_keywords(self):
        #dfs to get common keywords from children nodes.
        def dfs(root):
            root.keywords = get_keywords(root.text, -1, -1) # extracted keywords from text.
            
            for node in root.children:
                dfs(node)
                # Todo: function that effectively combines keywords array while maintaining scoring.
                for text, num in node.keywords.items():
                    if text not in root.keywords:
                        root.keywords[text] = num
                    else:
                        root.keywords[text] += num

            # Prune keywords array to only contain most common entries:
            count = 0
            temp = []
            for e in sorted(root.keywords.items(), key=lambda x: x[1], reverse=True):
                if count > max_keyword_count: 
                    break

                count += 1
                temp.append([e[0], e[1]])

            root.keywords = OrderedDict(temp)

        dfs(self)

    # DEBUG: used to output data structure. NOTE: Doesn't contain old text.
    def sprint_self_and_children(self):
        out = ""
        # Reject already used keywords:
        # NOTE: Assumes we don't exhaust the keywords in subsequent sets:
        used = set()
        keyword_cnt = 1

        # out += str(["".format(e.id, ','.join(a[0] for a in e.keywords[:keyword_cnt])) for e in q]) + '\n'
        prefixcnt = 0
        def dfs(root):
            nonlocal out, used, keyword_cnt, prefixcnt

            keys = []
            for k in root.keywords:
                if len(keys) < keyword_cnt and k not in used:
                    used.add(k)
                    keys.append(k)
            if len(keys) != 0:
                out += '\t' * prefixcnt + '- ' + ','.join(keys) + '\n'

            for node in root.children:
                if len(keys) == 0:
                    dfs(node)
                else:
                    prefixcnt += 1
                    dfs(node)
                    prefixcnt -= 1
            
        dfs(self)
        return out

    # DEBUG: used to output data structure. NOTE: Doesn't contain old text.
    def sprint_self_and_children_debug(self):
        out = ""
        q = deque([self])
        size = 1
        out += str(["id: {}, keywords: [{}]".format(e.id, ','.join(e.keywords)) for e in q]) + '\n'

        while q:
            u = q.pop()
            size -= 1

            if u.children:
                q.extendleft(u.children)
            
            if size == 0 and len(q) != 0:
                size = len(q)
                out += str(["id: {}, keywords: [{}]".format(e.id, ','.join(e.keywords)) for e in q]) + '\n'

        return out

# Returns tree root
def buildTree(edges, lines):
    # Assemble tree data structure. Start with the weakest node edge as root.
    # NOTE: accept tree_edges and lines as input.
    # 1) Assemble adjacency list:
    adjList = [[] for i in range(len(edges) + 1)]
    for u, v, _ in edges:
        adjList[u].append(v)
        adjList[v].append(u)

    # 2) bfs and create TreeNode struct:
    uroot = edges[-1][0] if len(edges) != 0 else 0    # Handle case with single node. 
    visited = set()
    root = TreeStruct(uroot, lines[uroot]) # Return this.
    q = deque([root])   # houses data structure.

    # TODO: struct instead of plain numbers b/c we need to add children.
    while q:
        u = q.pop()
        visited.add(u.id)

        for v_id in adjList[u.id]:
            # Check if neighbors were already created:
            if v_id in visited:
                continue
            
            v_node = TreeStruct(v_id, lines[v_id])
            #TODO: Replace this with more convenient serializable function as you see fit. Creating data structure for now...
            q.appendleft(v_node)
            u.children.append(v_node)

    return root
"""
API endpoint: 

Input: text
Output: Formatted markdown / JSON.

"""

# with open("./sampletext1.txt", 'r') as f:
def parseText(inputtext):
    
    """
    Assumptions: 
    Line-by-line text extraction - somehow parse subheadings
    Alphanumeric characters with punctuation only (ascii without escape sequences or newlines / tabs/any weird characters)
    
    If sentences need to be combined, assume they're combined with nearest neighbors (Controversial - may be more optimal to group with
    other sentences elsewhere but would be too complicated of an algorithm)

    NOTE: model supports grouping sentences into paragraphs too!

    """

    lines = preprocess(inputtext)
    corr_res = get_corr(lines)

    """
    # DEBUG/INFO:
    print("corr_res has dimensions {} by {}".format(len(corr_res), len(corr_res[0])))

    # DEBUG/INFO:
    # NOTE: values close to 0 represent no correlation, whereas values close to -1 represent inversely correlated.
    # Max and min correlation that's not perfect (DEBUG):
    maxes = [max(filter(lambda a: round(a, 5) != 1, e)) for e in corr_res]
    mins = [min(e) for e in corr_res]
    print(max(maxes))
    print(min(mins))

    # DEBUG/INFO: Frequency count of correlations:
    low, high = round(min(mins) - 0.1, 1), round(max(maxes) + 0.1, 1)
    rangeval = np.arange(low, high, 0.1)
    bucket = defaultdict(int)

    for arr in corr_res:
        for e in arr:
            idx = str(round(e, 1))
            bucket[idx] += 1

    print(bucket.items())
    """

    # Get minimum spanning tree:
    tree_edges = get_maxspantree_edges(corr_res)
    
    # build tree:
    root = buildTree(tree_edges, lines)    
    tree_edges.clear()

    # print(tree_edges)
    # print(len(tree_edges))
    

    # Build keywords
    root.parse_keywords()

    # Prettify tree and return plaintext/markdown.
    return root.sprint_self_and_children()
    # NOTE: One thing to ask mentors: Worth the time to invest in data cleansing more or just select a few data points (paragraphs) and roll with it?

# MAIN
if __name__ == "main":
    for i in range(1, numinputs + 1):
        fileinname, fileoutname = "./sampletext{}.txt".format(i), "./sampleoutput{}.txt".format(i)

        f = open(fileinname, 'r', encoding="utf8")
        text = f.read()
        f.close()

        out = parseText(text)

        with open(fileoutname, 'w', encoding="utf8") as g:
            g.write(out)
