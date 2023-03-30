import numpy as np

def keyphrase_ids_to_document_weights(keyphrases_ids, document_ids, weights):
    def KMPSearch(pat, txt):
        M = len(pat)
        N = len(txt)
        lps = [0] * M
        j = 0
        computeLPSArray(pat, M, lps)
        indxs = list()
        i = 0
        while i < N:
            if pat[j] == txt[i]:
                i += 1
                j += 1

            if j == M:
                indxs.append(i - j)
                j = lps[j - 1]
            elif i < N and pat[j] != txt[i]:
                if j != 0:
                    j = lps[j - 1]
                else:
                    i += 1
        return indxs

    def computeLPSArray(pat, M, lps):
        len = 0
        i = 1
        while i < M:
            if pat[i] == pat[len]:
                len += 1
                lps[i] = len
                i += 1
            else:
                if len != 0:
                    len = lps[len - 1]
                else:
                    lps[i] = 0
                    i += 1

    search = keyphrases_ids
    search = [i[1:-1] for i in search]
    ind_range = [(KMPSearch(i, document_ids), len(i)) for i in search]
    token_weighting = np.zeros(len(document_ids), dtype=float)
    for i, positions in enumerate(ind_range):
        for j in positions[0]:
            slots = range(j, (j + positions[1] - 1))
            token_weighting[slots] = token_weighting[slots] + weights[i]

    return token_weighting, ind_range










