#!pip install pyvis
import copy
import queue
import random
from IPython.core.display import display, HTML
import numpy as np
from pyvis.network import Network

#שלום לכולם, בתוכנית זו כתבתי את רוב האלגוריתמים שלמדנו בהרצאות ובתרגולים. כל האלגוריתמים תכתבו ע"פ הפדואוקודים.
#אוכל לנסות להמחיש לכם את האלגוריתמים pyvis - בעזרת הספרייה
# בואו נתחיל:
#  true כ directed  כך נכריז על רשת/גרף, על מנת שהגרף יהיה מכוון, נשים את הארגומנט
net = Network(directed=True)
# ניתן להוסיף קודקוד בודד כך
#net.add_node(0)
# וניתן באמצעות מערך של קודקודים
nodes = ["1", "2", "3", "4", "5", "6", "7"]
net.add_nodes(nodes)

# נוסיף קשתות כך:
net.add_edge(1, 2, arrows='to', value=30, title="30")
net.add_edge(2, 3, arrows='to', value=5, title="5")
net.add_edge(3, 4, arrows='to', value=3, title="3")
net.add_edge(4, 5, arrows='to', value=10, title="10")
net.add_edge(5, 6, arrows='to', value=5, title="5")
net.add_edge(6, 1, arrows='to', value=10, title="10")
net.add_edge(1, 7, arrows='to', value=10, title="10")

# אם נרצה להוסיף קשת ללא חץ בגרף לא מכוון נוסיף זאת כך:
#net.add_edge(1, 5)

#   נדפיס את הגרף. שימו לב לחלון הקופץ show, באמצעות המתודה
#net.show('mygraph.html')
# collab את הפונקציה הזו יש להפעיל אם נרצה להדפיס בהרצה ב
display(HTML('mygraph.html'))

#בהתחלה מימשתי כמה פונק' עזר ממליץ לכם לעבור ישירות לbfs

#תוכנית שיוצרת גרף רנדומלי לבקשת המשתמש
def random_network(vertex, edges, directed):
    if directed:
        network = Network(directed=True)
        for i in range(0, vertex):
            network.add_node(i)
        for i in range(0, edges):
            v_from = random.randrange(0, vertex, 1)
            v_to = random.randrange(0, vertex, 1)
            val = random.randrange(1, 10, 1)
            network.add_edge(v_from, v_to, arrows='to', value=val, title=val)
    else:
        network = Network(directed=False)
        for i in range(0, vertex):
            network.add_node(i)
        for i in range(0, edges):
            v_from = random.randrange(0, vertex, 1)
            v_to = random.randrange(0, vertex, 1)
            val = random.randrange(0, 10, 1)
            network.add_edge(v_from, v_to, value=val, title=val)
    return network


def nodes_to_numbers(net):
    d ={}
    i = 0
    for v in net.get_nodes():
        d[i] = v
        i += 1
    return d


def get_edges_net(net):
    array = []
    for i in net.get_edges():
        array.append((i["from"], i["to"]))
    return array


def make_network(net, edges, p):
    network = Network(directed=True)
    for i in net.get_nodes():
        network.add_node(i)
    for e in edges:
        if edges[e] > 0:
            if e in p:
                network.add_edge(e[0], e[1], arrows='to', value=edges[e], title=edges[e], color='#00ff1e')
            else:
                network.add_edge(e[0], e[1], arrows='to', value=edges[e], title=edges[e])
    return network


# complexity - O(E+V)
def bfs(net, s):
    queue = []
    visited = [False] * (net.num_nodes() + 1)
    path = []
    d = {}
    counter = 0
    queue.append(s)
    path.append(s)
    visited[s] = True
    while queue:
        v = queue.pop(0)
        for i in net.neighbors(v):
            if visited[i] is False:
                queue.append(i)
                path.append(i)
                visited[i] = True
    return path


# complexity - O(E+V)
def DFS(net, scc=False, array=[]):
    #אתחול
    color = {}
    path = []
    time_dict = {}
    time = 0
    for v in net.get_nodes():
        color[v] = "white"
    # scc ישמש רק להחזרת זמני הסיום
    if scc is False:
        for v in net.get_nodes():
            if color[v] == "white":
                time, color, path, time_dict = DFS_Visit(net, v, time, color, path, time_dict)
        return time_dict
    else:
        scc_connects = {}
        index = 0
        for v in array:

            if color[int(v[0])] == "white":
                time, color, path, time_dict = DFS_Visit(net, int(v[0]), time, color, path, time_dict)
                scc_connects[index] = path
                path = []
                index += 1
        return scc_connects



def DFS_Visit(net, s, time, color, path, time_dict):
    color[s] = "Gray"
    time += 1
    path.append(s)

    for i in net.neighbors(s):
        if color[i] == "white" and (s, i) in get_edges_net(net):
            time, color, path, time_dict = DFS_Visit(net, i, time, color, path, time_dict)
    color[s] = "Black"
    time += 1
    time_dict[s] = time
    return time, color, path, time_dict


# תוכנית היוצרת גרף transpose
# complexity - O(E)
def transpose(net):
    transpose_net = Network(directed=True)

    for i in get_edges_net(net):
        if i[0] not in transpose_net.get_nodes():
            transpose_net.add_node(i[0], color='#00ff1e')
        if i[1] not in transpose_net.get_nodes():
            transpose_net.add_node(i[1],color='#00ff1e')
        transpose_net.add_edge(i[1], i[0], arrows="to")
    return transpose_net


# מציאת רכיבים קשירים חזק
# complexity - O(E)
def strong_connected_component(net):
    # יצירת גרף transpose
    tra = transpose(net)
    # אם תרצו להדפיס את הגרף תורידו מההערה
    #tra.show('mygraphtranspose.html')
    # הרצת DFS על הגרף הרגיל
    dict_f = DFS(net)
    array_f = []
    # הכנסת זמני הסיום ומיונם
    for key in dict_f.keys():
        array_f.append((key, dict_f[key]))
    array_f.sort(key=lambda place: place[1], reverse=True)
    # הרצת DFS על הגרף המהופך
    return DFS(tra, scc=True, array=array_f)


print(strong_connected_component(net))


# תוכנית הבודקת אם קיים מסלול בין שני קודקודים בגרף
# complexity - O(V)
def exist_path(v_1, v_2, net, color):
    if len(net.neighbors(v_2)) == 0:
        return False
    color[v_2] = "black"
    flag = False
    for i in net.neighbors(v_2):
        if i == v_1:
            return True
        if color[i] == "white":
            flag = exist_path(v_1, i, net, color)
        if flag:
            return True
    return False


#תוכנית הבודקת אם צלע מסויימת סוגרת מעגל בגרף
# complexity - O(V+E)
def close_circle(edge, net):
    copy = Network(directed=False)
    copy.add_node(edge[0])
    copy.add_node(edge[1])
    for e in get_edges_net(net):
        if e[0] not in copy.get_nodes():
            copy.add_node(e[0])
        if e[1] not in copy.get_nodes():
            copy.add_node(e[1])
        if not e == edge:
            copy.add_edge(e[0], e[1])
    color = {}
    for v in net.get_nodes():
        color[v] = "white"
    return exist_path(edge[0], edge[1], copy, color)


# קרוסקל- מציאת עץ פורש מינימלי
# complexity - O(ElogV)
def kruskal(net):
    array_of_edges = []
    for i in net.get_edges():
        array_of_edges.append((i["from"], i["to"], i["value"]))
    # מיון הקשתות
    array_of_edges.sort(key=lambda place: place[2])
    minimal_span_graph = Network(directed=False)
    sum = 0
    for edge in array_of_edges:
        # הכנסת הקודקודים
        if edge[0] not in minimal_span_graph.get_nodes():
            minimal_span_graph.add_node(edge[0])
        if edge[1] not in minimal_span_graph.get_nodes():
            minimal_span_graph.add_node(edge[1])
        # הכנסת הצלע המינימלית אם היא לא סוגרת מעגל
        if not close_circle(edge, minimal_span_graph):
            minimal_span_graph.add_edge(edge[0], edge[1], value=edge[2], title=edge[2])
            sum += edge[2]

    # אופציה להדפסה:
    #minimal_span_graph.show('mygraphkruskal.html')
    display(HTML('mygraphkruskal.html'))
    return sum


# החזרת קשתות שכנות
def neighbors_edges(net, net_1 ,v, array):
    for i in net.get_edges():
        if i["to"] in net.neighbors(v) and not i["to"] in net_1.get_nodes():

            array.append((v, i["to"], i["value"]))
        if i["from"] in net.neighbors(v) and not i["from"] in net_1.get_nodes():
            array.append((v, i["from"], i["value"]))
    return array


# פרים- עץ פורש מינימלי באמצעות בחירת הצלע המינימלית שבחתך שלא סוגרת מעגל
# complexity - O(E+ VlogV) (אצלי זה V*V +E)
def prim(net, r):
    minimal_span_graph = Network(directed=False)
    minimal_span_graph.add_node(r, color='#162347')
    array_of_neighbors_edges = neighbors_edges(net, minimal_span_graph, r, [])
    v = r
    sum = 0
    while minimal_span_graph.num_nodes() < net.num_nodes():
        min_edge = min(array_of_neighbors_edges, key=lambda place: place[2])
        array_of_neighbors_edges.remove(min_edge)
        if not close_circle(min_edge, minimal_span_graph):
            if min_edge[0] not in minimal_span_graph.get_nodes():
                minimal_span_graph.add_node(min_edge[0], color='#162347')
            minimal_span_graph.add_node(min_edge[1], color='#162347')
            minimal_span_graph.add_edge(min_edge[0], min_edge[1], value=min_edge[2], title=min_edge[2])
            sum += min_edge[2]
            v = min_edge[1]
            array_of_neighbors_edges = neighbors_edges(net, minimal_span_graph, v, array_of_neighbors_edges)

    #minimal_span_graph.show('prim.html')
    display(HTML('prim.html'))
    return sum


def minDistance(net, d, vistSet):
    min = 10001
    min_index = net.get_nodes()[0]
    i = 1
    while vistSet[min_index]:
        min = d[net.get_nodes()[i]]
        min_index = net.get_nodes()[i]
        i += 1
    for v in net.get_nodes():
        if d[v] < min and vistSet[v] is False:
            min = d[v]
            min_index = v
    return min_index


# אלגוריתם למציאת מרחק קצר ביותר בין שני קודקודיםף אצלי זה מחזיר מערך של המחרקים מהקודקוד הראשון
def dijkstra(net, s):
    graph = Network(directed=True)
    graph.add_node(s, color='#00ff1e')
    edges = {}
    for i in net.get_edges():
        edges[(i["from"], i["to"])] = i["value"]
    d = {}
    d[s] = 0
    pi = {}
    queue = []
    vistSet = {}
    vistSet[s] = True
    for v in net.get_nodes():
        if not v == s:
            d[v] = 10000
            pi[v] = None
            queue.append(v)
            graph.add_node(v, color='#00ff1e')
            vistSet[v] = False
    # סוף אתחול
    # עבור כל השגנים של S
    for v in net.neighbors(s):
    #  בדיקה האם ניתן לקצר דרך s
        if d[v] > d[s] + edges[(s, v)]:
            graph.add_edge(s, v, value=edges[(s, v)], title=edges[(s, v)])
            d[v] = d[s] + edges[(s, v)]
            pi[v] = s
    # בדיקה של שאר הקודקודים
    while not len(queue) == 0:
        u = minDistance(net, d, vistSet)
        vistSet[u] = True
        if u in queue:
            queue.remove(u)

        for v in net.neighbors(u):

            if d[v] > d[u] + edges[(u, v)]:
                graph.add_edge(u, v, value=edges[(u, v)], title=edges[(u, v)])
                d[v] = d[u] + edges[(u, v)]
                pi[v] = u
    #graph.show('dijkstra.html')
    # display(HTML('dijkstra.html'))
    return d

# עוד מימוש של דייקסטרה שעשיתי בשביל פונק אחרת
def dijkstra_2(net, s, w):
    graph = Network(directed=True)
    graph.add_node(s, color='#00ff1e')
    for v in net.get_nodes():
        graph.add_node(v)
    for e in w.keys():
        graph.add_edge(e[0], e[1], value=w[e], title=w[e])
    d = {}
    d[s] = 0
    queue = []
    vistSet = {}
    vistSet[s] = True
    for v in graph.get_nodes():
        if not v == s:
            d[v] = 10000
            queue.append(v)
            vistSet[v] = False
    for v in net.neighbors(s):
        if d[v] > d[s] + w[(s, v)]:
            graph.add_edge(s, v, value=w[(s, v)], title=w[(s, v)])
            d[v] = d[s] + w[(s, v)]
    while not len(queue) == 0:
        u = minDistance(graph, d, vistSet)
        vistSet[u] = True
        if u in queue:
            queue.remove(u)
        for v in graph.neighbors(u):
            if d[v] > d[u] + w[(u, v)]:
                d[v] = d[u] + w[(u, v)]
    array = []
    for key in d.keys():
        array.append(d[key])
    return array


# בלמן פורד- מודד את כל הדרכים הקצרות ביותר מהמקור לשאר הקודקודים, מתריע על מעגל שלילי
# complexity - O(E*V)
def bellman_ford(net, s):
    graph = Network(directed=True)
    edges = {}
    for i in net.get_edges():
        edges[(i["from"], i["to"])] = i["value"]
    d = {}
    pi = {}
    for v in net.get_nodes():
        d[v] = 10000
        pi[v] = None
        graph.add_node(v, color='#00ff1e')
    d[s] = 0
    # סוף אתחול

    # מעבר על כל הקודקודים ועל כל הקשתות לייעול הדרך
    for v in net.get_nodes():
        if not v == s:
            for edge in edges.keys():
                if d[edge[1]] > d[edge[0]] + edges[edge]:
                    graph.add_edge(edge[0], edge[1], value=edges[(edge[0], edge[1])], title=edges[(edge[0], edge[1])])
                    d[edge[1]] = d[edge[0]] + edges[edge]
                    pi[edge[1]] = edge[0]
    # בדיקה אם יש עוד ייעול לאחר כל האינטרציות תיתן אינדיקציה למעגל שלילי
    count = 0
    for edge in edges.keys():
        if d[edge[1]] > d[edge[0]] + edges[edge]:
            count+=1
    return count


# אלגוריתם למציאת מרחקים קצרים ביותר בין כל זוג קודקודים
# complexity - O(V^3)
def floyd_warshall(net):
    d = [[10000 for k in range(0, net.num_nodes())] for j in range(0, net.num_nodes())]
    D = np.array(d, dtype=np.float64)
    edges = {}
    for i in net.get_edges():
        edges[(i["from"], i["to"])] = i["value"]
    # אתחול הערכים ההתחלתיים לקשת או ל0 אם מדובר באותו קודקוד
    for v in net.get_nodes():
        D[v][v] = 0
        for u in net.get_nodes():
            if (v, u) in edges.keys():
                x = edges[(v, u)]
                D[v][u] = x
    # סוף אתחול
    for k in net.get_nodes():
        for i in net.get_nodes():
            for j in net.get_nodes():
                # השורה הכי חשובה באלגוריתם, השמת המרחק המינימלי בין הערך הקודם לערך בעזרת שימוש במסלול אחר
                # שימו לב יש מספר שאלות שמשחקות על השורה הזאת
                D[i][j] = min(D[i][j], D[i][k] + D[k][j])
    change = []
    C = copy.deepcopy(D)
    for k in net.get_nodes():
        for i in net.get_nodes():
            for j in net.get_nodes():
                C[i][j] = min(C[i][j], C[i][k] + C[k][j])

    for i in net.get_nodes():
        for j in net.get_nodes():
            if C[i][j] < D[i][j]:
                if j not in change:
                    change.append(j)
    print("---len = ", len(change))
    print(change)
    return D


# החזרת מערך של אפסים ואחדות אם יש דרך או אין
def transitive_closure(net):
    D = [[0 for k in range(0, net.num_nodes())] for j in range(0, net.num_nodes())]
    edges = {}
    for i in net.get_edges():
        edges[(i["from"], i["to"])] = i["value"]
    for v in net.get_nodes():
        D[v][v] = 1
        for u in net.get_nodes():
            if (v, u) in edges.keys():
                D[v][u] = 1
    for k in net.get_nodes():
        for i in net.get_nodes():
            for j in net.get_nodes():
                D[i][j] = D[i][j] or (D[i][k] and D[k][j])
    return D


# ג'והנסון משתמש בבלמן פורד לאינדקציה אם יש מעגל שליל. אם לא מה טוב, אם כן אז מריץ דיקסטרה באמצעות משקול מחדש ע"י הוספת קודקוד
# חיצוני המשמש לשימור המרחק הקצר ביותר
def johnson(net, s):
    if bellman_ford(net, s):
        return True
    graph = Network(directed=True)
    graph.add_node(net.num_nodes()+1, color='#00ff1e')
    edges = {}
    for i in net.get_edges():
        edges[(i["from"], i["to"])] = i["value"]
    h = {}
    for v in net.get_nodes():
        h[v] = 10000
        graph.add_node(v, color='#00ff1e')
        graph.add_edge(net.num_nodes()+1, v, value=0, title=0)
        #edges[net.num_nodes()+1, v] = 0
    h[net.num_nodes()+1] = 0
    for v in graph.get_nodes():
        if not v == net.num_nodes()+1:
            for edge in edges.keys():
                if h[edge[1]] > h[edge[0]] + edges[edge]:
                    graph.add_edge(edge[0], edge[1], value=edges[(edge[0], edge[1])], title=edges[(edge[0], edge[1])])
                    h[edge[1]] = h[edge[0]] + edges[edge]
    w = {}
    D = []
    n_dict = nodes_to_numbers(net)
    for edge in edges.keys():
        w[(edge[0], edge[1])] = edges[edge] + h[edge[0]] - h[edge[1]]
    for i in n_dict.keys():
        D.append(dijkstra_2(net, n_dict[i], w))
        for j in n_dict.keys():
            D[i][j] -= h[n_dict[i]] - h[n_dict[j]]
    print("johnson:\n", np.array(D))


# מציאת דרך שיורית באמצעות dfs
def augmenting_path_flow(net, s, t, f):
    visited = [False] * (net.num_nodes()+1)
    queue = []
    pi = {}

    queue.append(s)
    visited[s] = True

    while queue:
        u = queue.pop(0)
        for v in net.neighbors(u):
            if visited[v] is False and f[(u, v)] > 0:
                queue.append(v)
                visited[v] = True
                pi[v] = u
    path = []
    if visited[t]:
        v = t
        path.append(t)
        c_min = f[(pi[v], v)]
        while pi[v] is not s:
            path.append(pi[v])
            v = pi[v]
            if f[(pi[v], v)] < c_min:
                c_min = f[(pi[v], v)]
        path.append(s)
        path.reverse()
        path_edge = []
        for i in range(0, len(path) - 1):
            path_edge.append((path[i], path[i + 1]))
        return c_min, path_edge
    return 0, None


# פורד פולקסרון מציאת זרימה מקסימלית בגרף
# complexity - O(V*E^2)


def ford_fulkerson(net, s, t):
    flow = 0
    f = {}
    for i in net.get_edges():
        f[(i["from"], i["to"])] = i["value"]
    c, p = augmenting_path_flow(net, s, t, f)
    while c:
        flow += c
        for edge in p:
            val_1 = f[(edge[0], edge[1])]
            val_0 = 0
            if (edge[1], edge[0]) in f:
                val_0 = f[(edge[1], edge[0])]
            f[(edge[0], edge[1])] = val_1 - c
            f[(edge[1], edge[0])] = val_0 + c
        network = make_network(net, f, p)
        #הדפסת שלב שלב
        #network.show("1.html")
        c, p = augmenting_path_flow(network, s, t, f)

    return flow


#driver



ex_net = Network(directed=True)
nodes = ["0", "1", "2", "3", "4", "5", "6", "7"]
ex_net.add_nodes(nodes)
ex_net.add_edge(0, 5, arrows='to', value=5, title="5")
ex_net.add_edge(0, 3, arrows='to', value=7, title="7")
ex_net.add_edge(0, 4, arrows='to', value=1, title="1")
ex_net.add_edge(1, 3, arrows='to', value=1, title="1")
ex_net.add_edge(1, 5, arrows='to', value=3, title="3")
ex_net.add_edge(1, 4, arrows='to', value=6, title="6")
ex_net.add_edge(2, 3, arrows='to', value=4, title="4")
ex_net.add_edge(2, 5, arrows='to', value=7, title="7")
ex_net.add_edge(2, 4, arrows='to', value=7, title="7")
ex_net.add_edge(3, 7, arrows='to', value=16, title="16")
ex_net.add_edge(4, 7, arrows='to', value=15, title="15")
ex_net.add_edge(5, 7, arrows='to', value=5, title="5")
ex_net.add_edge(6, 0, arrows='to', value=9, title="9")
ex_net.add_edge(6, 1, arrows='to', value=25, title="25")
ex_net.add_edge(6, 2, arrows='to', value=5, title="5")





def adj_mat_to_net(adj, len):
    net = Network(directed=True)
    for i in range(0, len):
        net.add_node(i)
    for i in range(0, len):
        for j in range(0, len):
            if not adj[i][j] == 0:
                net.add_edge(i, j, arrows='to', value=adj[i][j], title=adj[i][j])
    return net


def mul_vector(v, u):
    r = 0
    for i in range(len(v)):
        r += v[i]*u[i]
    return r


def mult(A, B):
    result = [[0 for k in range(len(A))] for j in range(len(B[0]))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            result[i][j] = mul_vector(A[i], [row[j] for row in B])
    return result


from collections import defaultdict


# This class represents a directed graph
# using adjacency matrix representation
class Graph:

    def __init__(self, graph):
        self.graph = graph  # residual graph
        self.org_graph = [i[:] for i in graph]
        self.ROW = len(graph)
        self.COL = len(graph[0])

    '''Returns true if there is a path from
    source 's' to sink 't' in
    residual graph. Also fills
    parent[] to store the path '''

    def BFS(self, s, t, parent):

        # Mark all the vertices as not visited
        visited = [False] * (self.ROW)

        # Create a queue for BFS
        queue = []

        # Mark the source node as visited and enqueue it
        queue.append(s)
        visited[s] = True

        # Standard BFS Loop
        while queue:

            # Dequeue a vertex from queue and print it
            u = queue.pop(0)

            # Get all adjacent vertices of
            # the dequeued vertex u
            # If a adjacent has not been
            # visited, then mark it
            # visited and enqueue it
            for ind, val in enumerate(self.graph[u]):
                if visited[ind] == False and val > 0:
                    queue.append(ind)
                    visited[ind] = True
                    parent[ind] = u

        # If we reached sink in BFS starting
        # from source, then return
        # true, else false
        return True if visited[t] else False

    # Function for Depth first search
    # Traversal of the graph
    def dfs(self, graph, s, visited):
        visited[s] = True
        for i in range(len(graph)):
            if graph[s][i] > 0 and not visited[i]:
                self.dfs(graph, i, visited)

    # Returns the min-cut of the given graph
    def minCut(self, source, sink):

        # This array is filled by BFS and to store path
        parent = [-1] * (self.ROW)

        max_flow = 0  # There is no flow initially

        # Augment the flow while there is path from source to sink
        while self.BFS(source, sink, parent):

            # Find minimum residual capacity of the edges along the
            # path filled by BFS. Or we can say find the maximum flow
            # through the path found.
            path_flow = float("Inf")
            s = sink
            while (s != source):
                path_flow = min(path_flow, self.graph[parent[s]][s])
                s = parent[s]

            # Add path flow to overall flow
            max_flow += path_flow
            print("===",max_flow)
            # update residual capacities of the edges and reverse edges
            # along the path
            v = sink
            while (v != source):
                u = parent[v]
                self.graph[u][v] -= path_flow
                self.graph[v][u] += path_flow
                v = parent[v]

        visited = len(self.graph) * [False]
        self.dfs(self.graph, s, visited)

        # print the edges which initially had weights
        # but now have 0 weight
        for i in range(self.ROW):
            for j in range(self.COL):
                if self.graph[i][j] == 0 and \
                        self.org_graph[i][j] > 0 and visited[i]:
                    print(self.org_graph[i][j])

                    print(str(i) + " - " + str(j))



# Create a graph given in the above diagram
graph = [[0, 23, 10, 4, 17, 20, 28, 36, 37, 0], [0, 0, 29, 32, 39, 0, 36, 19, 8, 0], [0, 0, 0, 31, 17, 0, 9, 0, 39, 20],
         [0, 0, 0, 0, 19, 29, 27, 31, 14, 0], [0, 0, 0, 0, 0, 0, 0, 5, 1, 11], [0, 34, 12, 0, 16, 0, 32, 4, 7, 26],
         [0, 0, 0, 0, 9, 0, 0, 14, 0, 0], [0, 0, 4, 0, 0, 0, 0, 0, 0, 29], [0, 0, 0, 0, 0, 0, 13, 16, 0, 0],
         [23, 34, 0, 1, 0, 0, 29, 0, 10, 0]]

g = Graph(graph)

source = 0
sink = 9

g.minCut(source, sink)




def ex_1():
    #ex_1
    st = " [[ 0 11 16 35 0 14 32 14 0 0] [ 0 0 0 8 0 15 5 0 39 21] [ 0 32 0 4 0 10 27 23 1 36] [ 0 0 0 0 12 14 30 8 31 3] [15 27 32 0 0 0 14 26 18 0] [ 0 0 0 0 20 0 28 17 36 8] [ 0 0 0 0 0 0 0 31 4 0] [ 0 36 0 0 0 0 0 0 0 32] [31 0 0 0 0 0 0 24 0 12] [34 0 0 0 4 0 3 0 0 0]]"
    st = st.replace(" ", ",")
    st = st.replace("[,", "[")
    print(st)
    adj = [[0,23,0,0,17,0,28,36,37,0],[0,0,29,32,39,0,36,19,8,0],[0,0,0,31,17,0,9,0,39,0],[0,0,0,0,19,29,27,31,14,0],[0,0,0,0,0,0,0,5,1,0],[0,34,12,0,16,0,32,4,7,0],[0,0,0,0,9,0,0,14,0,0],[0,0,4,0,0,0,0,0,0,0],[0,0,0,0,0,0,13,16,0,0],[23,34,0,1,0,0,29,0,10,0]]
    net = adj_mat_to_net(adj, 10)
    print("---",ford_fulkerson(net, 1, 10))
    for i in range(len(adj)):
        for j in range(len(adj[0])):
            adj[i][j] = 52
            max_flow = ford_fulkerson(adj_mat_to_net(adj, 10), 1, 10)
            if max_flow > 0:
                print(max_flow)
            adj[i][j] = 0








    # ex_2
    st =" [[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [ 0 0 0 2 0 0 0 1 0 0 0 0 0 3 0] [ 3 0 0 0 0 0 2 5 -5 0 0 3 0 0 0] [ 0 -4 0 0 0 0 0 0 0 0 -5 0 0 0 0] [ 0 4 0 0 0 0 0 0 0 1 0 0 0 0 4] [ 0 0 0 1 0 0 0 0 0 0 0 5 0 0 2] [ 2 0 5 4 0 0 0 0 0 0 0 0 0 0 0] [ 0 0 0 0 0 0 0 0 1 0 0 0 0 2 4] [ 0 0 0 4 0 4 3 0 0 0 0 0 0 0 0] [ 0 0 0 0 0 0 0 0 0 0 3 0 0 0 0] [ 0 0 0 5 0 1 0 0 0 0 0 0 2 2 0] [ 0 0 5 4 0 0 0 5 0 0 0 0 3 0 0] [ 0 0 0 0 0 0 0 0 0 0 5 0 0 0 0] [ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [ 0 0 0 0 5 0 0 0 0 0 0 0 0 0 0]]"
    st = st.replace(" ", ",")
    st = st.replace("[,", "[")
    print("2---", st)
    adj =[[0,4,0,4,0,0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,-3,0,5,0,0],[0,0,0,0,0,0,4,0,0,0,0,0,0,4,2],[0,3,4,0,0,0,0,5,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,0,2,0,0,2,0,3],[5,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,3,5,0,0,0,0],[3,0,0,0,-1,0,3,0,-4,0,3,0,0,0,0],[-2,0,4,-4,0,3,0,3,0,0,0,3,4,0,0],[0,0,0,0,0,0,0,0,0,0,4,0,5,0,0],[0,0,0,0,0,3,5,1,0,0,0,3,0,0,0],[0,0,0,0,5,-4,0,0,0,0,0,0,2,0,0],[-4,-4,5,0,0,0,2,1,0,0,0,5,0,0,4],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[3,0,0,0,0,0,0,2,0,0,0,0,0,0,0]]
    net = adj_mat_to_net(adj, len(adj[0]))
    net.show('1.html')
    c = floyd_warshall(net)
    print(bellman_ford(net, 0))


    # ex_3

    st = "{{0., 807.109, 2793.39, 2394.44, 4149.56, 2099.16, 2412.32, 1157.21, 2408.42}, {807.109, 0., 2068.34, 1695.67, 3364.35, 1362.91, 1778.76, 650.342, 1621.68}, {2793.39, 2068.34, 0., 415.114, 1465.8, 1804.06, 574.377, 1669.38, 1484.05}, {2394.44, 1695.67, 415.114, 0., 1878.2, 1671.56, 293.5, 1258.88, 1436.62}, {4149.56, 3364.35, 1465.8, 1878.2, 0., 2546., 2026.21, 3091.98, 2115.53}, {2099.16, 1362.91, 1804.06, 1671.56, 2546., 0., 1931.9, 1654.67, 430.614}, {2412.32, 1778.76, 574.377, 293.5, 2026.21, 1931.9, 0., 1255.41, 1720.7}, {1157.21, 650.342, 1669.38, 1258.88, 3091.98, 1654.67, 1255.41, 0., 1754.69}, {2408.42, 1621.68, 1484.05, 1436.62, 2115.53, 430.614, 1720.7, 1754.69, 0.}}"
    st = st.replace("{", "[")
    st = st.replace("}", "]")
    print("A",st)
    a = [[0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]]
    A = adj_mat_to_net(a, len(a[0]))
    #A.show('A.html')

    st = "{{0, 0, 0, 1, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0, 0}, {0, 0, 1, 0, 0, 0, 0}, {0, 0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 0, 1,0}, {0, 0, 0, 0, 0, 0, 1}}"
    st = st.replace("{", "[")
    st = st.replace("}", "]")
    #print("B",st)
    b = [[0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1,0], [0, 0, 0, 0, 0, 0, 1]]
    B = adj_mat_to_net(b, len(b[0]))
    #B.show('B.html')









    # ex_4
    st = "{{0., 4787.63, 3199.1, 2885.06, 2228.24, 1668.47, 1473.44, 2189.97, 1665.18}, {4787.63, 0., 2742.77, 2048.73, 3984.82, 4048.28, 3317.18, 3830.08, 4046.87}, {3199.1, 2742.77, 0., 1208.35, 1400.62, 1728.43, 2094.6, 1292.7, 1729.41}, {2885.06, 2048.73, 1208.35, 0., 2025.83, 2008.23, 1449.3, 1859.6, 2006.56}, {2228.24, 3984.82, 1400.62, 2025.83, 0., 563.68, 1813.92, 169.886, 567.283}, {1668.47, 4048.28, 1728.43, 2008.23, 563.68, 0., 1396.23, 527.339, 4.14614}, {1473.44, 3317.18, 2094.6, 1449.3, 1813.92, 1396.23, 0., 1674.15, 1392.33}, {2189.97, 3830.08, 1292.7, 1859.6, 169.886, 527.339, 1674.15, 0., 530.159}, {1665.18, 4046.87, 1729.41, 2006.56, 567.283, 4.14614, 1392.33, 530.159, 0.}}"
    st = st.replace("{", "[")
    st = st.replace("}", "]")
    print("4:  ",st)
    adj = [[0., 804.58, 149.599, 2274.01, 816.666, 1970.93, 786.202, 1244.73, 786.58], [804.58, 0., 622.563, 2808.08, 1612.17, 1203.14, 1140.19, 1743.54, 1143.37], [149.599, 622.563, 0., 2360.44, 964.44, 1797.07, 795.745, 1305.77, 796.958], [2274.01, 2808.08, 2360.44, 0., 1756.27, 3485.65, 1668.47, 3084.31, 1665.18], [816.666, 1612.17, 964.44, 1756.27, 0., 2729.34, 986.035, 1320.72, 982.913], [1970.93, 1203.14, 1797.07, 3485.65, 2729.34, 0., 1961.63, 2910.08, 1965.76], [786.202, 1140.19, 795.745, 1668.47, 986.035, 1961.63, 0., 1977.99, 4.14614], [1244.73, 1743.54, 1305.77, 3084.31, 1320.72, 2910.08, 1977.99, 0., 1977.16], [786.58, 1143.37, 796.958, 1665.18, 982.913, 1965.76, 4.14614, 1977.16, 0.]]
    net = adj_mat_to_net(adj, len(adj[0]))
    net.show('7.html')
    print("4:", prim(net, 1)*2)




ex_1()
