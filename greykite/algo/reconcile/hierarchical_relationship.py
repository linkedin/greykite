# BSD 2-CLAUSE LICENSE

# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:

# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# #ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# original author: Albert Chen
"""Represents hierarchical relationships between nodes.
Each node corresponds to a time series.
"""
import numpy as np

from greykite.common.python_utils import flatten_list


class HierarchicalRelationship:
    """Represents hierarchical relationships between nodes (time series).

    Nodes are indexed by their position in the tree, in breadth-first search (BFS) order.
    Matrix attributes such as ``bottom_up_transform`` are applied from the
    left against tree values, represented as a `numpy.array` 2D array with
    the values of each node as a row.

    Attributes
    ----------
    levels : `list` [`list` [`int]] or None
        Specifies the number of children of each parent (internal) node in the tree.
        The number of inner lists is the height of the tree. The ith inner list provides the number
        of children of each node at depth i.
        For example::

            # root node with 3 children
            levels = [[3]]

            # root node with 3 children, who have 2, 3, 3 children respectively
            levels = [[3], [2, 3, 3]]
            # These children are ordered from "left" to "right", so that the one with
            # 2 children is the first in the 2nd level.
            # This will be used as our running example.
            #           0                # level 0
            #   1       2        3       # level 1
            #  4 5    6 7 8    9 10 11   # level 2

        All leaf nodes must have the same depth. Thus, the first sublist must have one
        integer, the length of a sublist must equal the sum of the previous sublist,
        and all integers in ``levels`` must be positive.

    num_children_per_parent : `list` [`int`]
        Flattened version of ``levels``.
        The number of children for each parent (internal) node.
        [3, 2, 3, 3] in our example.
    num_internal_nodes : `int`
        The number of internal (parent) nodes (i.e. with children).
        4 in our example.
    num_leaf_nodes : `int`
        The number of leaf nodes (i.e. without children).
        8 in our example.
    num_nodes : `int`
        The total number of nodes.
        12 in our example.
    nodes_per_level : `list` [`int`]
        The number of nodes at each level of the tree.
        [1, 3, 8] in our example.
    starting_index_per_level : `list` [`int`]
        The index of the first node in each level.
        [0, 1, 4] in our example.
    starting_child_index_per_parent : `list` [`int`]
        For each parent node, the index of its first child.
        [1, 4, 6, 9]  in our example.
    sum_matrix : `numpy.array`, shape (``self.num_nodes``, ``self.num_leaf_nodes``)
        Sum matrix used to compute values of all nodes from the leaf nodes. When
        applied to a matrix with the values for leaf nodes, returns values for every
        node by bubbling up leaf node values to the internal nodes. A node's value is
        equal to the sum of its corresponding leaf nodes' values.

        ``Y_{all} = sum_matrix @ Y_{leaf}``
        In our example::

            # 4   5   6   7   8   9   10  11  (leaf nodes)
            [[1., 1., 1., 1., 1., 1., 1., 1.], # 0
             [1., 1., 0., 0., 0., 0., 0., 0.], # 1
             [0., 0., 1., 1., 1., 0., 0., 0.], # 2
             [0., 0., 0., 0., 0., 1., 1., 1.], # 3
             [1., 0., 0., 0., 0., 0., 0., 0.], # 4
             [0., 1., 0., 0., 0., 0., 0., 0.], # 5
             [0., 0., 1., 0., 0., 0., 0., 0.], # 6
             [0., 0., 0., 1., 0., 0., 0., 0.], # 7
             [0., 0., 0., 0., 1., 0., 0., 0.], # 8
             [0., 0., 0., 0., 0., 1., 0., 0.], # 9
             [0., 0., 0., 0., 0., 0., 1., 0.], # 10
             [0., 0., 0., 0., 0., 0., 0., 1.]] # 11 (all nodes)

    leaf_projection_matrix : `numpy.array`, shape (``self.num_leaf_nodes``, ``self.num_nodes``)
        Projection matrix to get leaf nodes. When applied to a matrix with the values for
        all nodes, the projection matrix selects only the rows corresponding to leaf nodes.

        ``Y_{leaf} = leaf_projection_matrix @ Y_{actual}``
        In our example::

            # 0   1   2   3   4   5   6   7   8   9   10  11  (all nodes)
            [[0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],  # 4
             [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],  # 5
             [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],  # 6
             [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],  # 7
             [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],  # 8
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],  # 9
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],  # 10
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]  # 11 (leaf nodes)

    bottom_up_transform: `numpy.array`, shape (``self.num_nodes,`` ``self.num_nodes``)
        Bottom-up transformation matrix. When applied to a matrix with the values for
        all nodes, returns values for every node by bubbling up leaf node values to the
        internal nodes. The original values of internal nodes are ignored.

        ``Y_{bu} = bottom_up_transform @ Y_{actual}``
        Note that ``bottom_up_transform = sum_matrix @ leaf_projection_matrix``.
        In our example::

            # 0   1   2   3   4   5   6   7   8   9   10  11  (all nodes)
            [[0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.], # 0
             [0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0.], # 1
             [0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0.], # 2
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1.], # 3
             [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.], # 4
             [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.], # 5
             [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.], # 6
             [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.], # 7
             [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.], # 8
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.], # 9
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.], # 10
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]] # 11 (all nodes)

    constraint_matrix : `numpy.array`, shape (``self.num_internal_nodes``, ``self.num_nodes``)
        Constraint matrix representing hierarchical additive constraints, where a parent's value
        is equal the sum of its leaf nodes' values.
        ``constraint_matrix @ Y_{all} = 0`` if ``Y_{all}`` satisfies the constraints.
        In our example::

            #  0    1    2    3    4    5    6    7    8    9    10   11  (all nodes)
            [[-1.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],  # 0
             [ 0., -1.,  0.,  0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],  # 1
             [ 0.,  0., -1.,  0.,  0.,  0.,  1.,  1.,  1.,  0.,  0.,  0.],  # 2
             [ 0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.]]  # 3 (internal nodes)

    Methods
    -------
    get_level_of_node : callable
        Returns a node's level in the tree
    get_child_nodes : callable
        Returns the indices of a node's children in the tree
    __set_sum_matrix : callable
        Constructs the summing matrix to compute values of all nodes from the leaf nodes.
    __set_leaf_projection_matrix : callable
        Constructs leaf projection matrix to retain only values of leaf nodes.
    __set_constraint_matrix : callable
        Constructs constraint matrix that requires each parent's
        value to be the sum of its leaf node's values.
    """

    def __init__(self, levels):
        """Computes attributes given the tree structure in ``levels``.

        Parameters
        ----------
        levels : `list` [`list` [`int]] or None
            See above.
        """
        if len(levels) == 0:
            raise ValueError("`levels` must contain at least one list")

        self.levels = levels

        # Flatten the list to get the number of children for each parent
        self.num_children_per_parent = flatten_list(levels)

        # Does not allow a node to have 0 children.
        # This requirement could be relaxed in the future.
        if any([i <= 0 for i in self.num_children_per_parent]):
            raise ValueError("Every parent node must have at least one child, "
                             "so that leaf nodes are at the same depth.")

        self.num_internal_nodes = len(self.num_children_per_parent)
        # The leaf nodes are children of the last level of parents
        self.num_leaf_nodes = sum(self.levels[-1])
        self.num_nodes = self.num_internal_nodes + self.num_leaf_nodes

        # There is one root node. Sum the number of children at each level to get the number of nodes at the next level
        self.nodes_per_level = [1] + [sum(arr) for arr in levels]
        # Alternatively, directly count the number of nodes per level
        alt_nodes_per_level = [len(arr) for arr in levels] + [self.num_leaf_nodes]
        if self.nodes_per_level != alt_nodes_per_level:
            raise ValueError(
                f"The number of children does not match the expected length. Found {self.nodes_per_level}\n"
                f"Expected {alt_nodes_per_level}. The length of a sublist must equal the sum"
                f"of the numbers in the preceding sublist.")

        # Indexes nodes starting from 0 for the root node, and continuing in BFS order.
        # The first index of each level is the number of nodes in all previous levels.
        self.starting_index_per_level = [0] + list(np.cumsum(self.nodes_per_level[:-1]))  # 0-indexed, length # levels

        # For each parent node, the index of its first child relative to the first node in
        # the next level (e.g. 0th in level, 2nd in level). length self.num_internal_nodes
        within_level_child_index_per_parent = flatten_list([list(np.cumsum([0] + arr[:-1])) for arr in levels])
        # For each parent node, the index of its first child, length self.num_internal_nodes
        self.starting_child_index_per_parent = [
            # the index of the next level plus the relative offset to its first child
            self.starting_index_per_level[self.get_level_of_node(parent)+1] + within_level_child_index_per_parent[parent]
            for parent in range(self.num_internal_nodes)]

        # Sum matrix to compute all nodes from leaf nodes. Y_{all} = self.sum_matrix @ Y_{leaf}
        # Shape (self.num_nodes, self.num_leaf_nodes)
        self.sum_matrix = self.__set_sum_matrix()

        # Projection matrix to get leaf nodes. Y_{leaf} = self.leaf_projection_matrix @ Y_{all}
        # Shape (self.num_leaf_nodes, self.num_nodes)
        self.leaf_projection_matrix = self.__set_leaf_projection_matrix()

        # Transform for bottom-up transform. Y_{bu} = self.bottom_up_transform @ Y_{all}
        # Shape (self.num_nodes, self.num_nodes)
        self.bottom_up_transform = self.sum_matrix @ self.leaf_projection_matrix

        # Constraint matrix. self.constraint_matrix @ Y_{all} is 0 if Y_{all} satisfies the constraints.
        # Shape (self.num_internal_nodes, self.num_nodes)
        self.constraint_matrix = self.__set_constraint_matrix()

    def get_level_of_node(self, node):
        """Returns a node's level in the tree.
        Level is defined as the length of the path to the root.
        The root is at level 0.

        Parameters
        ----------
        node : `int`
            Index of the node.

        Returns
        -------
        level : `int`
            The level of the node in the tree.
        """
        return max([level for level, start_index in enumerate(self.starting_index_per_level) if node >= start_index])

    def get_child_nodes(self, node):
        """Returns the indices of a node's children in the tree.

        Parameters
        ----------
        node : `int`
            Index of the node.

        Returns
        -------
        child_nodes : `list` [`int`]
            Indices of all the node's children.
        """
        first_child_index = self.starting_child_index_per_parent[node]
        num_children = self.num_children_per_parent[node]
        child_nodes = list(range(first_child_index, first_child_index + num_children))
        return child_nodes

    def __set_sum_matrix(self):
        """Constructs the summing matrix.

        Returns
        -------
        sum_matrix : `numpy.array`, shape (``self.num_nodes``, ``self.num_leaf_nodes``)
            Sum matrix used to compute values of all nodes from the leaf nodes.
        """
        sum_matrix = np.zeros([self.num_nodes, self.num_leaf_nodes])

        def set_matrix_row(i, matrix):
            """Sets the final value of row ``i`` in ``matrix``.

            Consider the entry [i, j] in the matrix.
            matrix[i, j] = 1 if j is a leaf node and (i==j or j is a descendant of i),
            0 otherwise.

            Also recursively sets the values for all children of node ``i``.
            The matrix is defined bottoms-up using DFS.

            Parameters
            ----------
            i : `int`
                Node index. The function updates the row for
                this node and all its descendants.
            matrix : `numpy.array`, shape (self.num_nodes, self.num_leaf_nodes)
                Sum matrix to fill in. Passed by reference in the recursion.

            Returns
            -------
            matrix[i] : `numpy.array`
                Row i of the matrix, after filling in the proper entries.
            """
            if i >= self.num_internal_nodes:
                # leaf node is equal to itself
                leaf_node_index = i - self.num_internal_nodes
                matrix[i] = np.zeros(self.num_leaf_nodes)
                matrix[i, leaf_node_index] = 1.0
            else:
                child_nodes = self.get_child_nodes(i)
                # parent node's leaves are its children's leaves
                matrix[i] = np.sum([set_matrix_row(j, matrix) for j in child_nodes], axis=0)
            return matrix[i]

        set_matrix_row(0, sum_matrix)
        return sum_matrix

    def __set_leaf_projection_matrix(self):
        """Constructs leaf projection matrix.

        Returns
        -------
        leaf_projection_matrix : `numpy.array`, shape (``self.num_leaf_nodes``, ``self.num_nodes``)
            Projection matrix to get leaf nodes.
        """
        return np.concatenate((
            np.zeros([self.num_leaf_nodes, self.num_internal_nodes]),  # removes the parent nodes
            np.eye(self.num_leaf_nodes)  # projects the leaf nodes
        ), axis=1)

    def __set_constraint_matrix(self):
        """Constructs constraint matrix that requires each parent nodes's
        value to be the sum of its leaf nodes' values.

        Returns
        -------
        constraint_matrix : `numpy.array`, shape (``self.num_internal_nodes``, ``self.num_nodes``)
            Constraint matrix representing hierarchical additive constraints,
            where a parent's value equals the sum of its leaf nodes' values.
        """
        assert hasattr(self, "bottom_up_transform") and self.bottom_up_transform is not None
        arr = self.bottom_up_transform - np.eye(self.bottom_up_transform.shape[0])
        return arr[:self.num_internal_nodes, ]
