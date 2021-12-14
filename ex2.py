class Tree:
    def __init__(self, id_, left=None, right=None, parent=None):
        self.id = id_
        self.left = left
        self.right = right
        self.balance_factor = 0
        self.size = 0
        self.parent = parent

    def print_tree(self):
        print(self.data)


def get_height(root):
    if root is None:
        return 0

    my_height = 1 + max(get_height(root.left), get_height(root.right))
    return my_height


def insert_balance_factor(tree):
    if tree is None:
        return
    tree.balance_factor = get_height(tree.left) - get_height(tree.right)
    insert_balance_factor(tree.left)
    insert_balance_factor(tree.right)


def max_balance_factor(tree):
    if tree is None:
        return 0
    return max(tree.balance_factor, max_balance_factor(tree.left), max_balance_factor(tree.right))


def is_in_tree(tree, id_0):
    if tree is None:
        return False
    if tree.id == int(id_0):
        return True
    return is_in_tree(tree.left, id_0) or is_in_tree(tree.right, id_0)


def find_in_tree(tree, id_0):
    if tree is None:
        return
    if tree.id == id_0:
        return tree
    if is_in_tree(tree.left, id_0):
        return find_in_tree(tree.left, id_0)
    if is_in_tree(tree.right, id_0):
        return find_in_tree(tree.right, id_0)


def from_edge_to_tree(list_of_edge):
    tree = Tree(id_=list_of_edge[0][0], left=Tree(id_=list_of_edge[0][1]))
    list_of_edge = list_of_edge[1:len(list_of_edge)]
    for v1, v2 in list_of_edge:
        node = find_in_tree(tree, v1)
        if node is None:
            return tree
        if node.left is None:
            node.left = Tree(id_=v2)
        else:
            node.right = Tree(id_=v2)

    return tree


import functools as fn


def printBTree(node, nodeInfo=None, inverted=False, isTop=True):

   # node value string and sub nodes
   stringValue, leftNode, rightNode = str(node.id), node.left, node.right

   stringValueWidth  = len(stringValue)

   # recurse to sub nodes to obtain line blocks on left and right
   leftTextBlock     = [] if not leftNode else printBTree(leftNode,nodeInfo,inverted,False)

   rightTextBlock    = [] if not rightNode else printBTree(rightNode,nodeInfo,inverted,False)

   # count common and maximum number of sub node lines
   commonLines       = min(len(leftTextBlock),len(rightTextBlock))
   subLevelLines     = max(len(rightTextBlock),len(leftTextBlock))

   # extend lines on shallower side to get same number of lines on both sides
   leftSubLines      = leftTextBlock  + [""] *  (subLevelLines - len(leftTextBlock))
   rightSubLines     = rightTextBlock + [""] *  (subLevelLines - len(rightTextBlock))

   # compute location of value or link bar for all left and right sub nodes
   #   * left node's value ends at line's width
   #   * right node's value starts after initial spaces
   leftLineWidths    = [ len(line) for line in leftSubLines  ]
   rightLineIndents  = [ len(line)-len(line.lstrip(" ")) for line in rightSubLines ]

   # top line value locations, will be used to determine position of current node & link bars
   firstLeftWidth    = (leftLineWidths   + [0])[0]
   firstRightIndent  = (rightLineIndents + [0])[0]

   # width of sub node link under node value (i.e. with slashes if any)
   # aims to center link bars under the value if value is wide enough
   #
   # ValueLine:    v     vv    vvvvvv   vvvvv
   # LinkLine:    / \   /  \    /  \     / \
   #
   linkSpacing       = min(stringValueWidth, 2 - stringValueWidth % 2)
   leftLinkBar       = 1 if leftNode  else 0
   rightLinkBar      = 1 if rightNode else 0
   minLinkWidth      = leftLinkBar + linkSpacing + rightLinkBar
   valueOffset       = (stringValueWidth - linkSpacing) // 2

   # find optimal position for right side top node
   #   * must allow room for link bars above and between left and right top nodes
   #   * must not overlap lower level nodes on any given line (allow gap of minSpacing)
   #   * can be offset to the left if lower subNodes of right node
   #     have no overlap with subNodes of left node
   minSpacing        = 2
   rightNodePosition = fn.reduce(lambda r,i: max(r,i[0] + minSpacing + firstRightIndent - i[1]), \
                                 zip(leftLineWidths,rightLineIndents[0:commonLines]), \
                                 firstLeftWidth + minLinkWidth)

   # extend basic link bars (slashes) with underlines to reach left and right
   # top nodes.
   #
   #        vvvvv
   #       __/ \__
   #      L       R
   #
   linkExtraWidth    = max(0, rightNodePosition - firstLeftWidth - minLinkWidth )
   rightLinkExtra    = linkExtraWidth // 2
   leftLinkExtra     = linkExtraWidth - rightLinkExtra

   # build value line taking into account left indent and link bar extension (on left side)
   valueIndent       = max(0, firstLeftWidth + leftLinkExtra + leftLinkBar - valueOffset)
   valueLine         = " " * max(0,valueIndent) + stringValue
   slash             = "\\" if inverted else  "/"
   backslash         = "/" if inverted else  "\\"
   uLine             = "¯" if inverted else  "_"

   # build left side of link line
   leftLink          = "" if not leftNode else ( " " * firstLeftWidth + uLine * leftLinkExtra + slash)

   # build right side of link line (includes blank spaces under top node value)
   rightLinkOffset   = linkSpacing + valueOffset * (1 - leftLinkBar)
   rightLink         = "" if not rightNode else ( " " * rightLinkOffset + backslash + uLine * rightLinkExtra )

   # full link line (will be empty if there are no sub nodes)
   linkLine          = leftLink + rightLink

   # will need to offset left side lines if right side sub nodes extend beyond left margin
   # can happen if left subtree is shorter (in height) than right side subtree
   leftIndentWidth   = max(0,firstRightIndent - rightNodePosition)
   leftIndent        = " " * leftIndentWidth
   indentedLeftLines = [ (leftIndent if line else "") + line for line in leftSubLines ]

   # compute distance between left and right sublines based on their value position
   # can be negative if leading spaces need to be removed from right side
   mergeOffsets      = [ len(line) for line in indentedLeftLines ]
   mergeOffsets      = [ leftIndentWidth + rightNodePosition - firstRightIndent - w for w in mergeOffsets ]
   mergeOffsets      = [ p if rightSubLines[i] else 0 for i,p in enumerate(mergeOffsets) ]

   # combine left and right lines using computed offsets
   #   * indented left sub lines
   #   * spaces between left and right lines
   #   * right sub line with extra leading blanks removed.
   mergedSubLines    = zip(range(len(mergeOffsets)), mergeOffsets, indentedLeftLines)
   mergedSubLines    = [ (i,p,line + (" " * max(0,p)) )       for i,p,line in mergedSubLines ]
   mergedSubLines    = [ line + rightSubLines[i][max(0,-p):]  for i,p,line in mergedSubLines ]

   # Assemble final result combining
   #  * node value string
   #  * link line (if any)
   #  * merged lines from left and right sub trees (if any)
   treeLines = [leftIndent + valueLine] + ( [] if not linkLine else [leftIndent + linkLine] ) + mergedSubLines

   # invert final result if requested
   treeLines = reversed(treeLines) if inverted and isTop else treeLines

   # return intermediate tree lines or print final result
   if isTop : print("\n".join(treeLines))
   else     : return treeLines


def ex1(list_of_edges):
    t = from_edge_to_tree(list_of_edges)
    printBTree(t)
    insert_balance_factor(t)
    print("max balance factor is :", max_balance_factor(t))



def sucssesor(x):
    y = x
    if x.right is not None:
        x = x.right
        while x.left is not None:
            x = x.left
    else:
        if x.parent.left is not None and x.parent.left.id == x.id:
            return x.parent.id
        if x.parent is not None:
            x = x.parent
            while x is not None and x.parent is not None and x.id > x.parent.id:
                if x.parent.parent is None and x.id > x.parent.id:
                    return False
                if x.parent.parent is None and x.id < x.parent.id:
                    return x.parent.id
                x = x.parent
            if x is not None and x.parent is not None and x.parent.id > x.id:
                return x.parent.id
    return x.id

def inorder_number(root, tree):
    if tree is None:
        return
    if tree.left is not None:
        inorder_number(root, tree.left)
    print(tree.id, sucssesor(tree))
    inorder_number(root, tree.right)


def postorder_number(tree):
    if tree is None:
        return
    if tree.left is not None:
        postorder_number(tree.left)
    postorder_number(tree.right)
    print(tree.id, end="")


def preorder_number(tree):
    if tree is None:
        return
    print(tree.id, end="")
    if tree.left is not None:
        preorder_number(tree.left)
    preorder_number(tree.right)


def insert(tree, v):
    if tree is None:
        return
    if v > tree.id and tree.right is None:
        tree.right = Tree(v, parent=tree)
    if v < tree.id and tree.left is None:
        tree.left = Tree(v, parent=tree)
    if v > tree.id:
        insert(tree.right, v)
    else:
        insert(tree.left, v)


def bst_from_list(list_of_vertex):
    tree = Tree(list_of_vertex[0])
    list_of_vertex = list_of_vertex[1:len(list_of_vertex )]
    for v in list_of_vertex:
        insert(tree, v)
    return tree


def left_rotate(tree):
    if tree.right is None:
        return
    if tree.right.left is not None:
        root = tree.right
        t_r_l = root.left
        root.left = tree
        tree.right = t_r_l
        printBTree(root)
        return root
    root = tree.right
    root.left = tree
    tree.right = None
    printBTree(root)
    return root


def count_left_rotate(tree):
    count = 0
    while tree.right is not None:
        tree = left_rotate(tree)
        count += 1
    return count


def insert_size(tree):
    if tree is None:
        return 0
    size = 1
    if tree.left is not None:
        size += insert_size(tree.left)
    if tree.right is not None:
        size += insert_size(tree.right)
    tree.size = size
    return size


def rank(root, node):
    if root.id == node.id:
        if root.left is not None:
            return root.left.size + 1
        return 1
    size_left = 0
    if node.left is not None:
        size_left = node.left.size
    size_right = 0
    if node.right is not None:
        size_right = node.right.size
    #אם הוא בן שמאלי של אבא שלו
    if node.parent.left is not None and node.parent.left.id == node.id:
        return rank(root, node.parent) - 1 - size_right
    # אם הוא בן ימני של אבא שלו
    if node.parent.right is not None and node.parent.right.id == node.id:
        return size_left + rank(root, node.parent) + 1


list_1 = [5000, 8238, 9541, 9032, 2102, 1407, 6773, 6068, 7766, 6893, 868, 9116, 5534, 7308, 3220, 7698, 6427, 2390, 5890, 8598, 1896, 5207, 4050, 820, 5656, 7366, 974, 7175, 5560, 9152, 9182, 3939]
t = bst_from_list(list_1)
insert_size(t)
#print(rank(t, t.right.left))
printBTree(t)
inorder_number(t, t)
print("\n")
#print("num of left rotate is:", count_left_rotate(t))


def find_suc(root, node, from_r):
    #אם יש בן ימני ולא עליתי מימין
    if node.right is not None and not from_r:
        son = node.right
        while son.left is not None:
            son = son.left
        return son.id
    else:
        #אם אני בן שמאלי של אבי
        if node.parent.left is not None and node.parent.left.id == node.id:
            return node.parent.id
        else:
            # בו ימני של אבי
            if node.parent.id == root.id:#הגעתי לשורש
                return False
            return find_suc(root, node.parent, True)



printBTree(t)
print(sucssesor(t.left.right.right))