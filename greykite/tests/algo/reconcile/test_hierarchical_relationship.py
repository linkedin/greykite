import numpy as np
import pytest

from greykite.algo.reconcile.hierarchical_relationship import HierarchicalRelationship
from greykite.common.python_utils import assert_equal


def test_hierarchical_relationship():
    # Invalid input
    with pytest.raises(ValueError, match="`levels` must contain at least one list"):
        HierarchicalRelationship(levels=[])

    with pytest.raises(ValueError, match="Every parent node must have at least one child, "
                                         "so that leaf nodes are at the same depth."):
        HierarchicalRelationship(levels=[[0]])

    with pytest.raises(ValueError, match="Every parent node must have at least one child, "
                                         "so that leaf nodes are at the same depth."):
        HierarchicalRelationship(levels=[[3], [0, 1, 2], [3, 3, 3]])

    with pytest.raises(ValueError, match="The number of children does not match the expected length."):
        HierarchicalRelationship(levels=[[3], [2, 2]])

    # Height 1, edge case (root with single child)
    levels = [[1]]
    tree = HierarchicalRelationship(levels=levels)
    assert tree.levels == [[1]]
    assert tree.num_children_per_parent == [1]
    assert tree.num_internal_nodes == 1
    assert tree.num_leaf_nodes == 1
    assert tree.num_nodes == 2
    assert tree.nodes_per_level == [1, 1]
    assert tree.starting_index_per_level == [0, 1]
    assert tree.starting_child_index_per_parent == [1]
    assert_equal(tree.sum_matrix, np.array([[1.], [1.]]))
    assert_equal(tree.leaf_projection_matrix, np.array([[0., 1.]]))
    assert_equal(tree.constraint_matrix, np.array([[-1.,  1.]]))
    assert_equal(tree.bottom_up_transform, np.array([[0., 1.], [0., 1.]]))

    # Height 1
    levels = [[2]]
    tree = HierarchicalRelationship(levels=levels)
    assert tree.levels == [[2]]
    assert tree.num_children_per_parent == [2]
    assert tree.num_internal_nodes == 1
    assert tree.num_leaf_nodes == 2
    assert tree.num_nodes == 3
    assert tree.nodes_per_level == [1, 2]
    assert tree.starting_index_per_level == [0, 1]
    assert tree.starting_child_index_per_parent == [1]
    assert_equal(tree.sum_matrix, np.array([
        [1., 1.],
        [1., 0.],
        [0., 1.]]))
    assert_equal(tree.leaf_projection_matrix, np.array([
        [0., 1., 0.],
        [0., 0., 1.]]))
    assert_equal(tree.constraint_matrix, np.array([[-1.,  1.,  1.]]))
    assert_equal(tree.bottom_up_transform, np.array([
        [0., 1., 1.],
        [0., 1., 0.],
        [0., 0., 1.]]))

    # Height 2
    levels = [[3], [3, 2, 1]]
    tree = HierarchicalRelationship(levels=levels)
    assert tree.levels == [[3], [3, 2, 1]]
    assert tree.num_children_per_parent == [3, 3, 2, 1]
    assert tree.num_internal_nodes == 4
    assert tree.num_leaf_nodes == 6
    assert tree.num_nodes == 10
    assert tree.nodes_per_level == [1, 3, 6]
    assert tree.starting_index_per_level == [0, 1, 4]
    assert tree.starting_child_index_per_parent == [1, 4, 7, 9]
    assert_equal(tree.sum_matrix, np.array([
        [1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 0., 0., 0.],
        [0., 0., 0., 1., 1., 0.],
        [0., 0., 0., 0., 0., 1.],
        [1., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 1.]]))
    assert_equal(tree.leaf_projection_matrix, np.array([
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]))
    assert_equal(tree.constraint_matrix, np.array([
        [-1., 0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.],
        [0., -1.,  0.,  0.,  1.,  1.,  1.,  0.,  0.,  0.],
        [0.,  0., -1.,  0.,  0.,  0.,  0.,  1.,  1.,  0.],
        [0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  1.]]))
    assert_equal(tree.bottom_up_transform, np.array([
        [0., 0., 0., 0., 1., 1., 1., 1., 1., 1.],
        [0., 0., 0., 0., 1., 1., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]))

    # Height 3
    levels = [[2], [2, 3], [3, 2, 3, 1, 1]]
    tree = HierarchicalRelationship(levels=levels)
    assert tree.levels == [[2], [2, 3], [3, 2, 3, 1, 1]]
    assert tree.num_children_per_parent == [2, 2, 3, 3, 2, 3, 1, 1]
    assert tree.num_internal_nodes == 8
    assert tree.num_leaf_nodes == 10
    assert tree.num_nodes == 18
    assert tree.nodes_per_level == [1, 2, 5, 10]
    assert tree.starting_index_per_level == [0, 1, 3, 8]
    assert tree.starting_child_index_per_parent == [1, 3, 5, 8, 11, 13, 16, 17]
    assert_equal(tree.sum_matrix, np.array([
        [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
        [1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 1., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]))
    assert_equal(tree.leaf_projection_matrix, np.array([
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]))
    assert_equal(tree.constraint_matrix, np.array([
        [-1., 0.,  0.,  0.,  0.,  0.,  0.,  0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        [0., -1.,  0.,  0.,  0.,  0.,  0.,  0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
        [0.,  0., -1.,  0.,  0.,  0.,  0.,  0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
        [0.,  0.,  0., -1.,  0.,  0.,  0.,  0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0.,  0.,  0.,  0., -1.,  0.,  0.,  0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0.],
        [0.,  0.,  0.,  0.,  0., -1.,  0.,  0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0.],
        [0.,  0.,  0.,  0.,  0.,  0., -1.,  0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0.,  0.,  0.,  0.,  0.,  0.,  0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]))
    assert_equal(tree.bottom_up_transform, np.array([
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]))
