"""def longest_from_string(s, file_path):
    if s == "":
        return 0
    max_len_word = 0
    longest_word = ""
    with open(file_path, 'r') as file_source:
        content = file_source.readlines()
        content = [x.strip() for x in content]
        for word in content:
            if word in s and int(len(word)) > max_len_word:
                max_len_word = int(len(word))
                longest_word = word
    return max_len_word
"""

def palindrome_sub_sequences(s):
    array_palindrome = []
    for i in range(0, int(len(s))):
        for j in range(i, int(len(s))):
            left_index = i
            right_index = j
            while left_index <= right_index and s[left_index] == s[right_index] and s[left_index:right_index+1] not in array_palindrome:
                left_index += 1
                right_index -= 1
            if s[left_index:right_index+1] in array_palindrome and s[i:j+1] not in array_palindrome:
                array_palindrome.append(s[i:j + 1])
            elif left_index > right_index and s[left_index - 1] == s[right_index + 1] and s[i:j+1] not in array_palindrome:
                array_palindrome.append(s[i:j+1])
    return max(array_palindrome, key=len)

print(palindrome_sub_sequences("a"), 1)

def max_pal(s):
    dict_of_index = {}
    for i in range(len(s)):
        for j in range(i+1):
            print(s[i-j:len(s)-j])
            if s[i-j:len(s)-j] == s[i-j:len(s)-j][::-1]:
                return s[i-j:len(s)-j]
    return ""

print(max_pal(""))

def lcs(array_1, array_2, array_3):
    m = len(array_1)
    n = len(array_2)
    t = len(array_3)
    L = [[[0 for k in range(0, t+1)] for j in range(0, n+1)] for i in range(0, m+1)]
    for k in range(0, t + 1):
        for i in range(0, n + 1):
            for j in range(0, m + 1):
                if i == 0 or j == 0 or k == 0:
                    L[j][i][k] = 0
                elif array_1[j - 1] == array_2[i - 1] == array_3[k - 1] :
                    L[j][i][k] = L[j - 1][i - 1][k - 1] + 1
                else:
                    L[j][i][k] = max(L[j - 1][i][k], L[j][i - 1][k], L[j][i][k - 1])
    index = L[m][n][t]
    lcs = [""] * (index)
    i = m
    j = n
    k = t
    while i > 0 and j > 0 and k > 0:
        if array_1[i - 1] == array_2[j - 1] and array_3[k - 1] == array_2[j - 1]:
            lcs[index - 1] = array_1[i - 1]
            i -= 1
            j -= 1
            k -= 1
            index -= 1
        elif L[i - 1][j][k] >= L[i][j - 1][k] and L[i - 1][j][k] >= L[i][j][k - 1]:
            i -= 1
        elif L[i][j - 1][k] >= L[i - 1][j][k] and L[i][j - 1][k] >= L[i][j][k - 1]:
            j -= 1
        else:
            k -= 1
    return len(lcs)


def max_lcs(array_1, array_2, array_3):
    m = len(array_1)
    n = len(array_2)
    L = [[0 for j in range(0, n+1)] for i in range(0, m+1)]
    for i in range(0, n + 1):
         for j in range(0, m + 1):
            if j == 0:
                L[j][i] = array_1[j]
            if i == 0:
                L[j][i] = array_2[i]
            elif array_1[j - 1] == array_2[i - 1]:
                L[j][i] = L[j - 1][i - 1]*array_2[i - 1]
            else:
                L[j][i] = max(L[j - 1][i], L[j][i - 1])
    index = L[m][n]
    lcs = [""] * (index)
    i = m
    j = n
    while i > 0 and j > 0:
        if array_1[i - 1] == array_2[j - 1]:
            lcs[index - 1] = array_1[i - 1]
            i -= 1
            j -= 1
            index -= 1
        elif L[i - 1][j] >= L[i][j - 1]:
            i -= 1
        elif L[i][j - 1] >= L[i - 1][j]:
            j -= 1
    return len(lcs)



def longest_monotonic(array):
    n = len(array)
    non_decreasing_array = [1] * n
    non_increasing_array = [1] * n
    for i in range(1, n):
        for j in range(0, i):
            if array[i] >= array[j] and non_decreasing_array[j] + 1 > non_decreasing_array[i]:
                non_decreasing_array[i] = non_decreasing_array[j] + 1
            if array[i] <= array[j] and non_increasing_array[j] + 1 > non_increasing_array[i]:
                non_increasing_array[i] = non_increasing_array[j] + 1
    return max(non_decreasing_array)

"""

print(palindrome_sub_sequences("CDBCEECDEECDDCAECDAEDACEDAECAEAAEAAEEEBEDBEDCBCDBCDAAEDBDDADADADDADCAABCEDDACBEEDBEACEEEADDCAEABBABA"))
print(lcs([2, 2, 3, 2, 3, 1, 1, 2, 1, 1, 3, 3, 1, 0, 0, 3, 1, 3, 2, 3, 1, 2, 2, 2, 1, 2], [2, 2, 1, 3, 2, 1, 0, 2, 1, 0, 1, 2, 1, 3, 1, 3, 3, 2, 2, 0, 3, 2, 0, 0], [3, 2, 1, 3, 3, 2, 1, 0, 3, 2, 2, 1, 1, 3, 0, 2, 2, 0, 3, 2, 3, 2, 0, 3, 2]))
print(longest_monotonic([1, 0, 0, 3, 0, 2, 3, 3, 3, 0, 3, 1, 2, 3, 2, 2, 3, 0, 3, 1, 3, 0, 2, 3, 2, 2, 2, 3, 1, 0, 3, 0, 1, 3, 2, 1, 3, 2, 0, 1, 0, 3, 2, 2, 2, 3, 1, 2, 0, 0, 3, 3, 2, 3, 0, 1, 3, 0, 3, 3, 0, 2, 1, 1, 1, 0, 0, 3, 0]))
"""

A1={2, 2, 3, 2, 3, 1, 1, 2, 1, 1, 3, 3, 1, 0, 0, 3, 1, 3, 2, 3, 1, 2, 2, 2, 1, 2}

A2={2, 2, 1, 3, 2, 1, 0, 2, 1, 0, 1, 2, 1, 3, 1, 3, 3, 2, 2, 0, 3, 2, 0, 0}

A3={3, 2, 1, 3, 3, 2, 1, 0, 3, 2, 2, 1, 1, 3, 0, 2, 2, 0, 3, 2, 3, 2, 0, 3, 2}


def Longest_Palindrome(s):
    s2 = "|".join(s.split())
    array_Palindrome_radii = [0]*len(s2)
    print(s2)
Longest_Palindrome("wrty")


def convert(s, numRows=3):
    if numRows == 1:
        return s
    t = ""
    for i in range(numRows):
        if i == 0 or i == numRows - 1:
            t += s[i::(numRows - 1) * 2]
        else:
            if i + 0.1 > numRows / 2:
                j = i
                a = 0
                while j < len(s):
                    t += s[j]
                    if a == 0:
                        k = (numRows - i - 1) * 2
                        a = 1
                    else:
                        k = (i) * 2
                        a = 0
                    j += k
            else:
                if i < numRows // 2:
                    j = i
                    a = 0
                    while j < len(s):
                        t += s[j]
                        if a == 0:
                            k = (numRows - 1 - i) * 2
                            a = 1
                        else:
                            k = i * 2
                            a = 0
                        j += k
                elif i == numRows // 2:
                    t += s[i::numRows - 1]
    return t

print(convert("PAYPALISHIRING", 5))