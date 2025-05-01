# -*- coding: utf-8 -*-
"""
@author: Tom from Blue Sky
https://blog.csdn.net/u012421852/article/details/79801466
Aim: After obtaining the decision tree dictionary, use Python to draw the corresponding decision tree figure.
Input is a decision tree in dictionary form, e.g.:
dtree = {'house?': {'hourse_no': {'working?': {'work_no': 'refuse', 'work_yes': 'agree'}}, 'hourse_yes': 'agree'}}
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams['font.sans-serif']=['SimHei']  # Used to properly display Chinese labels

# Define the decision node shape; boxstyle indicates the type of text box, fc controls fill color
decisionNode = dict(boxstyle="round4", color='r', fc='0.9')
# Define the leaf node shape
leafNode = dict(boxstyle="circle", color='m')
# Define arrow shape from parent to child nodes
arrow_args = dict(arrowstyle="<-", color='g')

def plot_node(node_txt, center_point, parent_point, node_style):
    '''
    Draw the arrow from parent node to child node and fill in the text
    :param node_txt: text content
    :param center_point: coordinates of the current node
    :param parent_point: coordinates of the parent node
    '''
    createPlot.ax1.annotate(node_txt, 
                            xy=parent_point,
                            xycoords='axes fraction',
                            xytext=center_point,
                            textcoords='axes fraction',
                            va="center",
                            ha="center",
                            bbox=node_style,
                            arrowprops=arrow_args)

def get_leafs_num(tree_dict):
    '''
    Get the number of leaf nodes
    :param tree_dict: dictionary representing the tree
    :return: total number of leaf nodes in tree_dict
    '''
    leafs_num = 0
    # Get the root node
    root = list(tree_dict.keys())[0]
    # Get the subtree of the root
    child_tree_dict = tree_dict[root]
    for key in child_tree_dict.keys():
        # Check if the child is a subtree (dictionary)
        if type(child_tree_dict[key]).__name__ == 'dict':
            # Add leaf count from the subtree
            leafs_num += get_leafs_num(child_tree_dict[key])
        else:
            # It's a leaf node
            leafs_num += 1
    return leafs_num

def get_tree_max_depth(tree_dict):
    '''
    Get the maximum depth of the tree
    :param tree_dict: dictionary representing the tree
    :return: maximum depth
    '''
    max_depth = 0
    root = list(tree_dict.keys())[0]
    child_tree_dict = tree_dict[root]
    for key in child_tree_dict.keys():
        this_path_depth = 0
        if type(child_tree_dict[key]).__name__ == 'dict':
            this_path_depth = 1 + get_tree_max_depth(child_tree_dict[key])
        else:
            this_path_depth = 1
        if this_path_depth > max_depth:
            max_depth = this_path_depth
    return max_depth

def plot_mid_text(center_point, parent_point, txt_str):
    '''
    Compute the middle point between parent and child, and add text there
    :param center_point: coordinates of the child node
    :param parent_point: coordinates of the parent node
    '''
    x_mid = (parent_point[0] - center_point[0]) / 2.0 + center_point[0]
    y_mid = (parent_point[1] - center_point[1]) / 2.0 + center_point[1]
    createPlot.ax1.text(x_mid, y_mid, txt_str)
    return

def plotTree(tree_dict, parent_point, node_txt):
    '''
    Draw the tree recursively
    :param tree_dict: tree structure
    :param parent_point: coordinates of the parent node
    :param node_txt: label text
    '''
    leafs_num = get_leafs_num(tree_dict)
    root = list(tree_dict.keys())[0]
    center_point = (plotTree.xOff + (1.0 + float(leafs_num)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plot_mid_text(center_point, parent_point, node_txt)
    plot_node(root, center_point, parent_point, decisionNode)
    child_tree_dict = tree_dict[root]
    # Move down to the next level of the tree
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in child_tree_dict.keys():
        if type(child_tree_dict[key]).__name__ == 'dict':
            plotTree(child_tree_dict[key], center_point, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plot_node(child_tree_dict[key], (plotTree.xOff, plotTree.yOff), center_point, leafNode)
            plot_mid_text((plotTree.xOff, plotTree.yOff), center_point, str(key))
    # After drawing all child nodes, move back up one level
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD
    return

def createPlot(tree_dict):
    '''
    Create the plot for the decision tree
    :param tree_dict: decision tree dictionary
    '''
    # Set background color
    fig = plt.figure(1, facecolor='white')
    # Clear figure
    fig.clf()
    # Set up plot without ticks
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(get_leafs_num(tree_dict))
    plotTree.totalD = float(get_tree_max_depth(tree_dict))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(tree_dict, (0.5, 1.0), '')
    plt.show()

def main(tree_dict):
    createPlot(tree_dict)

if __name__ == '__main__':
    tree_dict = {'house?': {'hourse_no': {'working?': {'work_no': 'refuse', 'work_yes': 'agree'}}, 'hourse_yes': 'agree'}}
    createPlot(tree_dict)
