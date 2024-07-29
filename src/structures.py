"""
Data Structures.
-----------

This script defines the data structures needed for the algorithm of routine detection.

The module contains the following public classes:

Subsequence
-----------

Basic data structure.

    **Parameters**:
        * ``instance``: np.ndarray, the instance of the subsequence
        * ``date``: datetime.date, the date of the subsequence
        * ``starting_point``: int, the starting point of the subsequence

    **Public methods:**
        * ``get_instance() -> np.ndarray``: returns the instance of the subsequence
        * ``get_date() -> date``: returns the date of the subsequence
        * ``get_starting_point() -> int``: returns the starting point of the subsequence
        * ``to_collection() -> list[dict]``: returns the subsequence as a dictionary
        * ``magnitude() -> float``: returns the magnitude of the subsequence
        * ``inverse_magnitude() -> float``: returns the inverse magnitude of the subsequence
        * ``distance(other: Subsequence | np.ndarray) -> float``: returns the distance between the subsequence and another subsequence or array

    **Examples**:
        >>> subsequence = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
        >>> subsequence.get_instance()
        np.array([1, 2, 3, 4])

        >>> subsequence.get_date()
        datetime.date(2021, 1, 1)

        >>> subsequence.get_starting_point()
        0

        >>> subsequence.to_collection()
        {'instance': np.array([1, 2, 3, 4]), 'date': datetime.date(2021, 1, 1), 'starting_point': 0}

        >>> subsequence.magnitude()
        4

        >>> subsequence.distance(np.array([1, 2, 3, 4]))
        0

Sequence:
---------

Represents a sequence of subsequences

    **Parameters**:
        * ``subsequence``: Optional[Subsequence], the subsequence to add to the sequence. None is the default value

    **Properties**:
        *Getters:*
            * ``length_subsequences``: int, the length of the subsequences in the sequence

    **Public Methods:**
        * ``add_sequence(new: Subsequence)``: adds a Subsequence instance to the Sequence
        * ``get_by_starting_point(starting_point: int) -> Subsequence``: returns the subsequence with the specified starting point
        * ``set_by_starting_point(starting_point: int, new_sequence: Subsequence)``: sets the subsequence with the specified starting point
        * ``get_starting_points() -> list[int]``: returns the starting points of the subsequences
        * ``get_dates() -> list[dates]``: returns the dates of the subsequences
        * ``get_subsequences() -> list[np.ndarray]``: returns the instances of the subsequences
        * ``to_collection() -> list[dict]``: returns the sequence as a list of dictionaries

    **Examples**:

        >>> sequence = Sequence()
        >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
        >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
        >>> sequence.get_by_starting_point(0)
        Subsequence(instance=np.array([1, 2, 3, 4]), date=datetime.date(2021, 1, 1), starting_point=0)

        >>> sequence.get_starting_points()
        [0, 4]

        >>> sequence.get_dates()
        [datetime.date(2021, 1, 1), datetime.date(2021, 1, 2)]

        >>> sequence.get_subsequences()
        [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])]

        >>> sequence.to_collection()
        [{'instance': np.array([1, 2, 3, 4]), 'date': datetime.date(2021, 1, 1), 'starting_point': 0},
         {'instance': np.array([5, 6, 7, 8]), 'date': datetime.date(2021, 1, 2), 'starting_point': 4}]

        >>> sequence.length_subsequences
        4


Cluster
-------

Represents a cluster of subsequences

    **Parameters**:
        * ``centroid``: `np.ndarray`, the centroid of the cluster
        * ``instances``: `Sequence`, the sequences of subsequences

    **Properties**:
        *Getters:*
            * ``centroid: np.ndarray``: returns the centroid of the cluster
            * ``length_cluster_subsequences: int``: returns the length of the subsequences in the cluster
        *Setters:*
            * ``centroid: np.ndarray``: sets the centroid of the cluster

    **Public Methods:**
        * ``add_instance(new_subsequence: Subsequence)``: adds a subsequence to the cluster
        * ``get_sequences() -> Sequence``: returns the sequences of the cluster
        * ``update_centroid()``: updates the centroid of the cluster
        * ``get_starting_points() -> list[int]``: returns the starting points of the subsequences
        * ``get_dates() -> list[dates]``: returns the dates of the subsequences
        * ``cumulative_magnitude() -> float``: returns the cumulative magnitude of the cluster
        * ``fusion(other: Cluster) -> Cluster``: fusions the cluster with another cluster
        * ``is_similar(other: Cluster, distance_threshold: int | float) -> bool``: returns True if the cluster is similar to another cluster, False otherwise

    **Examples**:

        >>> subsequence1 = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
        >>> subsequence2 = Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)

        >>> sequence = Sequence(subsequence=subsequence1)
        >>> cluster = Cluster(np.array([1, 1, 1, 1]), sequence)

        >>> cluster.get_sequences()
        Sequence(
            length_subsequences=4,
            list_sequences=[
                Subsequence(
                    instance=np.array([1, 2, 3, 4]),
                    date=datetime.date(2021, 1, 1),
                    starting_point=0
                )
            ]

        >>> cluster.add_instance(subsequence2)
        >>> cluster.get_sequences()
        Sequence(
            length_subsequences=4,
            list_sequences=[
                Subsequence(
                    instance=np.array([1, 2, 3, 4]),
                    date=datetime.date(2021, 1, 1),
                    starting_point=0
                ),
                Subsequence(
                    instance=np.array([5, 6, 7, 8]),
                    date=datetime.date(2021, 1, 2),
                    starting_point=4
                )
            ]

        >>> cluster.update_centroid()
        >>> cluster.centroid
        np.array([3.0, 4.0, 5.0, 6.0])

        >>> cluster.get_starting_points()
        [0, 4]

        >>> cluster.get_dates()
        [datetime.date(2021, 1, 1), datetime.date(2021, 1, 2)]

        >>> cluster.cumulative_magnitude()
        12.0


Routines
--------

Represents a collection of clusters

    **Parameters**:
        * ``cluster: Optional[Cluster]``, the cluster to add to the collection. Default is None

    **Public Methods**:
        * ``add_routine(new_routine: Cluster)``: adds a cluster to the collection
        * ``drop_indexes(to_drop: list[int])``: drops the clusters with the specified indexes
        * ``get_routines() -> list[Cluster]``: returns the clusters from the `Routines`
        * ``get_centroids() -> list[np.ndarray]``: returns the centroids of the clusters
        * ``to_collection() -> list[dict]``: returns the routines as a list of dictionaries
        * ``is_empty() -> bool``: returns True if the routines are empty, False otherwise
        * ``drop_duplicates()``: drops the duplicates from the routines
        * ``remove_subsets()``: removes the subsets of those clusters from the routines that are subsets of another cluster

    **Examples**:

        >>> subsequence1 = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
        >>> subsequence2 = Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)

        >>> sequence = Sequence(subsequence=subsequence1)
        >>> cluster1 = Cluster(np.array([1, 1, 1, 1]), sequence)

        >>> sequence = Sequence(subsequence=subsequence2)
        >>> cluster2 = Cluster(np.array([5, 5, 5, 5]), sequence)

        >>> routines = Routines(cluster=cluster1)
        >>> routines.add_routine(cluster2)

        >>> routines.get_routines()
        [Cluster(
            centroid=np.array([1, 1, 1, 1]),
            sequences=Sequence(
                length_subsequences=4,
                list_sequences=[
                    Subsequence(
                        instance=np.array([1, 2, 3, 4]),
                        date=datetime.date(2021, 1, 1),
                        starting_point=0
                    )
                ]
            )
        ), Cluster(
            centroid=np.array([5, 5, 5, 5]),
            sequences=Sequence(
                length_subsequences=4,
                list_sequences=[
                    Subsequence(
                        instance=np.array([5, 6, 7, 8]),
                        date=datetime.date(2021, 1, 2),
                        starting_point=4
                    )
                ]
            )
        )]

        >>> routines.get_centroids()
        [np.array([1, 1, 1, 1]), np.array([5, 5, 5, 5])]

        >>> routines.to_collection()
        [{'centroid': np.array([1, 1, 1, 1]), 'sequences': [{'instance': np.array([1, 2, 3, 4]), 'date': datetime.date(2021, 1, 1), 'starting_point': 0}], 'length_subsequences': 4},
         {'centroid': np.array([5, 5, 5, 5]), 'sequences': [{'instance': np.array([5, 6, 7, 8]), 'date': datetime.date(2021, 1, 2), 'starting_point': 4}], 'length_subsequences': 4}]

        >>> routines.is_empty()
        False

        >>> routines.drop_indexes([0])
        >>> routines.get_routines()
        [Cluster(
            centroid=np.array([5, 5, 5, 5]),
            sequences=Sequence(
                length_subsequences=4,
                list_sequences=[
                    Subsequence(
                        instance=np.array([5, 6, 7, 8]),
                        date=datetime.date(2021, 1, 2),
                        starting_point=4
                    )
                ]
            )
        )]

HierarchyRoutine
----------------

Represents a hierarchy of routines,
where the hierarchy corresponds to the length of the subsequences for each cluster in the routine.
For each hierarchy, exists one routine with the correspondent length of the subsequences.

    **Parameters**:
        * ``routine: Optional[Routines]``, the routine to add to the hierarchy. Default is None

    **Public Methods**:
        * ``is_empty() -> bool``: returns True if the hierarchy is empty, False otherwise
        * ``add_routine(new_routine: Routines)``: adds a routine to the hierarchy
        * ``to_dictionary() -> dict``: returns the hierarchy as a dictionary
        * ``to_json(path: str)``: saves the hierarchy as a json file
        * ``from_json(path: str)``: loads the hierarchy from a json file
        * ``convert_to_cluster_tree() -> ClusterTree``: converts the hierarchy to a ClusterTree


    **Properties**:
        *Getters:*
            * ``keys: list[int]``: returns the list with all the hierarchies registered
            * ``values: list[Routines]``: returns a list with all the routines
            * ``items: Iterator[tuple[int, Routines]]``: returns a iterator as a zip object with the hierarchy and the routine

    **Examples**:

            >>> subsequence1 = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> subsequence2 = Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)

            >>> sequence = Sequence(subsequence=subsequence1)
            >>> cluster1 = Cluster(np.array([1, 1, 1, 1]), sequence)

            >>> sequence = Sequence(subsequence=subsequence2)
            >>> cluster2 = Cluster(np.array([5, 5, 5, 5]), sequence)

            >>> routines = Routines(cluster=cluster1)
            >>> routines.add_routine(cluster2)

            >>> hierarchy = HierarchyRoutine(routine=routines)
            >>> hierarchy.to_dictionary()
            {4: [{'centroid': np.array([1, 1, 1, 1]), 'sequences': [{'instance': np.array([1, 2, 3, 4]), 'date': datetime.date(2021, 1, 1), 'starting_point': 0}]},
                 {'centroid': np.array([5, 5, 5, 5]), 'sequences': [{'instance': np.array([5, 6, 7, 8]), 'date': datetime.date(2021, 1, 2), 'starting_point': 4}]}]}

            >>> hierarchy.keys
            [4]

            >>> hierarchy.values
            [Routines(
                centroid=np.array([1, 1, 1, 1]),
                sequences=Sequence(
                    length_subsequences=4,
                    list_sequences=[
                        Subsequence(
                            instance=np.array([1, 2, 3, 4]),
                            date=datetime.date(2021, 1, 1),
                            starting_point=0
                        )
                    ]
                )
            ), Routines(
                centroid=np.array([5, 5, 5, 5]),
                sequences=Sequence(
                    length_subsequences=4,
                    list_sequences=[
                        Subsequence(
                            instance=np.array([5, 6, 7, 8]),
                            date=datetime.date(2021, 1, 2),
                            starting_point=4
                        )
                    ]
                )
            )]

            >>> for key, value in hierarchy.items:
            ...     print(key, value)
            4 Routines(
                centroid=np.array([1, 1, 1, 1]),
                sequences=Sequence(
                    length_subsequences=4,
                    list_sequences=[
                        Subsequence(
                            instance=np.array([1, 2, 3, 4]),
                            date=datetime.date(2021, 1, 1),
                            starting_point=0
                        )
                    ]
                )
            )


ClusterTree
-----------

Represents a tree structure of clusters where each node is a cluster,
and the edges are the hierarchy relationships between the clusters.
Each node can have two children, the left and the right child.
Each child has to have hierarchy greater than the parent.
With this method, we can represent the hierarchy of the clusters in a tree structure
and see how the clusters are grown, if from the left or from the right.

    **Public Methods**:
        * ``is_left_child(parent: Cluster, child: Cluster) -> bool``: checks if the child cluster corresponds to the left child of the parent cluster
        * ``is_right_child(parent: Cluster, child: Cluster) -> bool``: checks if the child cluster corresponds to the right child of the parent cluster
        * ``get_name_node(node: Union[Cluster, int]) -> str``: returns the name of the node from the cluster or the index
        * ``get_node(node: Union[str, int]) -> Cluster``: returns the cluster of the node from the index or the name
        * ``get_index(node: Union[Cluster, str]) -> int``: returns the index of the node from the cluster or the name
        * ``get_nodes_with_hierarchy(hierarchy: int) -> list[Cluster]``: returns the nodes with the specified hierarchy
        * ``to_dictionary() -> dict``: returns a dictionary where the keys are the indexes and the values are the clusters
        * ``children(node: Union[Cluster, int]) -> list[int]``: returns the children of the node
        * ``is_child(parent: Union[Cluster, int], child: Union[Cluster, int]) -> bool``: returns `True` if the child is a child of the parent, `False` otherwise
        * ``is_existent_left_child(parent: Union[Cluster, int], child: Union[Cluster, int]) -> bool``: returns `True` if the child is the left child of the parent in the graph, `False` otherwise
        * ``parents(node: Union[Cluster, int]) -> list[int]``: returns the parents of the node
        * ``has_children(node: Union[Cluster, int]) -> bool``: returns `True` if the node has children, `False` otherwise
        * ``has_parents(node: Union[Cluster, int]) -> bool``: returns `True` if the node has parents, `False` otherwise
        * ``assign_names()``: assigns names to the existent nodes in the graph
        * ``reset_names()``: resets the names of the nodes in the graph
        * ``assign_node(cluster: Cluster)``: assigns a node to the tree
        * ``add_edge(parent: Union[Cluster, int], child: Union[Cluster, int], is_left: bool)``: adds an edge to the tree
        * ``drop_node(node: Union[Cluster, int, str])``: drops all the nodes that depends directly on the node specified (including the node)
        * ``convert_to_hierarchy_routine() -> HierarchyRoutine``: converts the tree to a HierarchyRoutine
        * ``plot_tree(node_size: int = 1000, with_labels: bool = True, figsize: tuple[int, int] = (7, 7), title: Optional[str] = None, title_fontsize: int = 15, save_dir: Optional[str] = None)``: plots the tree

    **Properties**:
        *Getters:*
            * ``indexes: list[int]``: returns the list of indexes
            * ``nodes: list[Cluster]``: returns the list of clusters
            * ``graph: nx.classes.digraph.DiGraph``: returns the graph
            * ``edges: nx.classes.reportviews.OutEdgeDataView``: returns the edges of the graph

    **Examples**:
        >>> tree = ClusterTree()
        >>> parent = Cluster(np.array([1, 2, 3]), Sequence(Subsequence(np.array([1, 2, 3]), datetime.date(2021, 1, 2), 1)))
        >>> left = Cluster(np.array([0, 1, 2, 3]), Sequence(Subsequence(np.array([0, 1, 2, 3]), datetime.date(2021, 1, 1), 0)))
        >>> right = Cluster(np.array([1, 2, 3, 4]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 2), 1)))

        >>> tree.assign_node(parent)
        >>> tree.assign_node(left)
        >>> tree.assign_node(right)

        >>> tree.add_edge(1, 2, is_left=True)
        >>> tree.add_edge(1, 3, is_left=False)

        >>> tree.assign_names()

        >>> tree.indexes
        [1, 2, 3]

        >>> tree.nodes
        [Cluster(
            - centroid = [1, 2, 3],
            - instances = [[1, 2, 3]]
            - starting_points = [1]
            - dates = [datetime.date(2021, 1, 2)]
        ), Cluster(
            - centroid = [0, 1, 2, 3],
            - instances = [[0, 1, 2, 3]]
            - starting_points = [0]
            - dates = [datetime.date(2021, 1, 1)]
        ), Cluster(
            - centroid = [1, 2, 3, 4],
            - instances = [[1, 2, 3, 4]]
            - starting_points = [1]
            - dates = [datetime.date(2021, 1, 2)]
        )]

        >>> tree.name_nodes
        ['3-1', '4-1', '4-2']

        >>> tree.edges
        OutEdgeDataView([(1, 2, {'left': True}), (1, 3, {'left': False})])

        >>> tree.parents(2)
        [1]

        >>> tree.children(1)
        [2, 3]

        >>> tree.is_child(1, 2)
        True

        >>> tree.is_left_child(1, 2)
        True

        >>> tree.is_right_child(1, 3)
        True

        >>> tree.has_children(1)
        True

        >>> tree.has_parents(2)
        True

        >>> tree.hierarchies
        [3, 4]

        >>> tree.get_nodes_with_hierarchy(4)
        [2, 3]

        >>> tree.get_name_node(parent)
        '3-1'

        >>> tree.get_node("4-1")
        Cluster(
            - centroid = [0, 1, 2, 3],
            - instances = [[0, 1, 2, 3]]
            - starting_points = [0]
            - dates = [datetime.date(2021, 1, 1)]
        )

        >>> routine = tree.convert_to_hierarchy_routine()
        >>> print(routine)
        HierarchyRoutine(
            [Hierarchy: 3.
                Routines(
                    list_routines=[
                        Cluster(
                            - centroid = [1, 2, 3],
                            - instances = [[1, 2, 3]]
                            - starting_points = [1]
                            - dates = [datetime.date(2021, 1, 2)]
                        )
                    ]
                )
            ],
            [Hierarchy: 4.
                Routines(
                    list_routines=[
                        Cluster(
                            - centroid = [0, 1, 2, 3],
                            - instances = [[0, 1, 2, 3]]
                            - starting_points = [0]
                            - dates = [datetime.date(2021, 1, 1)]
                        ),
                        Cluster(
                            - centroid = [1, 2, 3, 4],
                            - instances = [[1, 2, 3, 4]]
                            - starting_points = [1]
                            - dates = [datetime.date(2021, 1, 2)]
                        )
                    ]
                )
            ]

        >>> tree.drop_node("4-3")
        >>> tree.nodes
        [1, 2]
"""

from typing import Union, Optional, Iterator
from copy import deepcopy
from warnings import warn

import networkx as nx
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import datetime
import json
import os


class Subsequence:
    """
    Basic data structure to represent a subsequence from a sequence which belongs to time series

    Parameters:
    __________

        * ``instance: np.ndarray``, the instance of the subsequence
        * ``date: datetime.date``, the date of the subsequence
        * ``starting_point: int``, the starting point of the subsequence

    Public Methods:
    __________

        * ``get_instance() -> np.ndarray``: returns the instance of the subsequence
        * ``get_date() -> date``: returns the date of the subsequence
        * ``get_starting_point() -> int``: returns the starting point of the subsequence
        * ``to_collection() -> list[dict]``: returns the subsequence as a dictionary
        * ``magnitude() -> float``: returns the magnitude of the subsequence
        * ``distance(other: Subsequence | np.ndarray) -> float``: returns the distance between the subsequence and another subsequence or array

    Examples:

        >>> subsequence = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
        >>> subsequence.get_instance()
        np.array([1, 2, 3, 4])

        >>> subsequence.get_date()
        datetime.date(2021, 1, 1)

        >>> subsequence.get_starting_point()
        0

        >>> subsequence.to_collection()
        {'instance': np.array([1, 2, 3, 4]), 'date': datetime.date(2021, 1, 1), 'starting_point': 0}

        >>> subsequence.magnitude()
        4

        >>> subsequence.distance(np.array([1, 2, 3, 4]))
        0
    """

    def __init__(self, instance: np.ndarray, date: datetime.date, starting_point: int) -> None:
        """
        Parameters:
            * instance: `np.ndarray`, the instance of the subsequence
            * date: `datetime.date`, the date of the subsequence
            * starting_point: `int`, the starting point of the subsequence

        Raises:
            TypeError: if the parameters are not of the correct type

        Examples:
            >>> subsequence = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
        """

        self.__check_type(instance, date, starting_point)
        self.__instance = instance
        self.__date = date
        self.__starting_point = starting_point

    @staticmethod
    def __check_type(instance: np.ndarray, date: datetime.date, starting_point: int) -> None:
        """
        Checks the type of the parameters

        Parameters:
            * instance: `np.ndarray`, the instance of the subsequence
            * date: `datetime.date`, the date of the subsequence
            * starting_point: `int`, the starting point of the subsequence

        Raises:
            TypeError: if the parameters are not of the correct type
        """

        # Check if the instance is an array
        if not isinstance(instance, np.ndarray):
            raise TypeError(f"Instances must be an arrays. Got {type(instance).__name__} instead")

        # Check if the date is a datetime.date
        if not isinstance(date, datetime.date):
            raise TypeError(f"Date must be a timestamps. Got {type(date).__name__} instead")

        # Check if the starting point is an integer
        if not isinstance(starting_point, int):
            raise TypeError(f"starting_point must be a integer. Got {type(starting_point).__name__} instead")

    def __repr__(self):
        """
        Returns the string representation of the subsequence

        Returns:
            str. The string representation of the subsequence

        Examples:
            >>> subsequence = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> print(subsequence)
            Subsequence(
                instance=np.array([1, 2, 3, 4]),
                date=datetime.date(2021, 1, 1),
                starting_point=0
            )
        """

        return f"Subsequence(\n\t instance={self.__instance} \n\t date={self.__date} \n\t starting point = {self.__starting_point}\n)"

    def __str__(self):
        """
        Returns the string representation of the subsequence

        Returns:
            str. The string representation of the subsequence

        Examples:
            >>> subsequence = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> print(subsequence)
            Subsequence(
                instance=np.array([1, 2, 3, 4]),
                date=datetime.date(2021, 1, 1),
                starting_point=0
            )
        """

        return f"Subsequence(\n\t instances={self.__instance} \n\t date={self.__date} \n\t starting point = {self.__starting_point}\n)"

    def __len__(self) -> int:
        """
        Returns the length of the subsequence

        Returns:
            `int`. The length of the subsequence

        Examples:
            >>> subsequence = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> len(subsequence)
            4
        """
        return len(self.__instance)

    def __getitem__(self, index: int) -> int | float:
        """
        Get the item at the specified index in the subsequence

        Parameters:
            * index: `int`. The index of the item

        Returns:
            `float` | `int`. The item at the specified index in the subsequence

        Raises:
            TypeError: if the parameter is not of the correct type
            IndexError: if the index is out of range

        Examples:
            >>> subsequence = Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 1), 0)
            >>> subsequence[2]
            7
        """

        # Check if the index is an integer
        if not isinstance(index, int):
            raise TypeError(f"index must be an integer. Got {type(index).__name__} instead")

        # Check if the index is within the range of the list
        if not 0 <= index < len(self.__instance):
            raise IndexError(f"index {index} out of range (0, {len(self.__instance) - 1})")

        # If the item is a numpy integer or float, convert it to a Python integer or float and return it
        if isinstance(self.__instance[index], np.int32):
            return int(self.__instance[index])

        return float(self.__instance[index])

    def __eq__(self, other: 'Subsequence') -> bool:
        """
        Check if the subsequence is equal to another subsequence

        Parameters:
            other: `Subsequence`. The subsequence to compare

        Returns:
            `bool`. True if the subsequences are equal, False otherwise

        Raises:
            TypeError: if the parameter is not of the correct type

        Examples:
            >>> subsequence1 = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> subsequence2 = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> subsequence1 == subsequence2
            True

            >>> subsequence1 = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> subsequence2 = Subsequence(np.array([1, 2, 3, 4, 5]), datetime.date(2021, 1, 2), 2)
            >>> subsequence1 == subsequence2
            False
        """

        # Check if the parameter is an instance of Subsequence
        if not isinstance(other, Subsequence):
            raise TypeError(f"other must be an instance of Subsequence. Got {type(other).__name__} instead")

        # Check if they have the same length
        if len(self) != len(other):
            return False

        # Check if the instance, date, and starting point are equal
        if not np.array_equal(self.__instance, other.get_instance()):
            return False

        # Check if the date and starting point are equal
        if self.__date != other.get_date() or self.__starting_point != other.get_starting_point():
            return False

        return True

    def get_instance(self) -> np.ndarray:
        """
        Returns the instance of the subsequence

        Returns:
             `np.ndarray`. The instance of the `Subsequence`

        Examples:
            >>> subsequence = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> subsequence.get_instance()
            np.array([1, 2, 3, 4])
        """

        return self.__instance

    def get_date(self) -> datetime.date:
        """
        Returns the date of the subsequence

        Returns:
            `datetime.date`. The date of the `Subsequence`

        Examples:
            >>> subsequence = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> subsequence.get_date()
            datetime.date(2021, 1, 1)
        """

        return self.__date

    def get_starting_point(self) -> int:
        """
        Returns the starting point of the subsequence

        Returns:
             `int`. The starting point of the `Subsequence`

        Examples:
            >>> subsequence = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> subsequence.get_starting_point()
            0
        """

        return self.__starting_point

    def to_collection(self) -> dict:
        """
        Returns the subsequence as a dictionary

        Returns:
             `dict`. The subsequence as a dictionary

        Examples:
            >>> subsequence = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> subsequence.to_collection()
            {'instance': np.array([1, 2, 3, 4]), 'date': datetime.date(2021, 1, 1), 'starting_point': 0}
        """

        date = self.__date.strftime("%m/%d/%Y, %H:%M:%S")

        return {"instance": self.__instance, "date": date, "starting_point": self.__starting_point}

    def magnitude(self) -> float:
        """
        Returns the magnitude of the subsequence as the maximum value

        Returns:
             `float`. The magnitude of the subsequence

        Examples:
            >>> subsequence = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> subsequence.magnitude()
            4.0
        """

        return np.max(self.__instance)

    def inverse_magnitude(self) -> float:
        """
        Returns the inverse magnitude of the subsequence as the minimum value

        Returns:
            `float`. The inverse magnitude of the subsequence

        Examples:
            >>> subsequence = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> subsequence.inverse_magnitude()
            1.0
        """

        return np.min(self.__instance)

    def distance(self, other: Union['Subsequence', np.ndarray]) -> float:
        """
        Returns the maximum absolute distance between the subsequence and another subsequence or array

        Parameters:
            * other: `Union[Subsequence, np.ndarray]`, the subsequence or array to compare

        Returns:
            `float`. The distance between the subsequence and another subsequence or array

        Raises:
            TypeError: if the parameter is not of the correct type
            ValueError: if the instances have different lengths

        Examples:
            >>> subsequence = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> subsequence.distance(np.array([1, 2, 3, 4]))
            0.0

            >>> subsequence.distance(Subsequence(np.array([1, 2, 3, 6]), datetime.date(2021, 1, 2), 2))
            2.0
        """

        # Check if the parameter is an instance of Subsequence or np.ndarray
        if isinstance(other, Subsequence):
            new_instance = other.get_instance()

        elif isinstance(other, np.ndarray):
            new_instance = other

        # If the parameter is not an instance of Subsequence or np.ndarray, raise an error
        else:
            raise TypeError(
                f"other must be an instance of Subsequence or np.ndarray. Got {type(other).__name__} instead")

        # Check if the instances have the same length
        if len(self.__instance) != len(new_instance):
            raise ValueError(
                f"The instances must have the same length. len(self)={len(self.__instance)} and len(other)={len(new_instance)}")

        return np.max(np.abs(self.__instance - new_instance))


class Sequence:
    """
    Represents a sequence of subsequences

    Parameters:
    _________
        * ``subsequence: Optional[Subsequence]``, the subsequence to add to the sequence. Default is `None`

    Properties:
    _________

    **Getters**
        * ``length_subsequences: int``. The length of the subsequences in the sequence

    Public Methods:
    _________

        * ``add_sequence(new: Subsequence)`` : adds a `Subsequence` instance to the `Sequence`
        * ``get_by_starting_point(starting_point: int)`` -> Subsequence: returns the subsequence with the specified starting point
        * ``set_by_starting_point(starting_point: int, new_sequence: Subsequence):`` sets the subsequence with the specified starting point
        * ``get_starting_points() -> list[int]:`` returns the starting points of the subsequences
        * ``get_dates() -> list[dates]:`` returns the dates of the subsequences
        * ``get_subsequences() -> list[np.ndarray]:`` returns the instances of the subsequences
        * ``to_collection() -> list[dict]:`` returns the sequence as a list of dictionaries

    Examples:
        >>> sequence = Sequence()
        >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
        >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
        >>> sequence.get_by_starting_point(0)
        Subsequence(instance=np.array([1, 2, 3, 4]), date=datetime.date(2021, 1, 1), starting_point=0)
        >>> sequence.get_starting_points()
        [0, 4]
        >>> sequence.get_dates()
        [datetime.date(2021, 1, 1), datetime.date(2021, 1, 2)]
        >>> sequence.get_subsequences()
        [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])]
        >>> sequence.to_collection()
        [{'instance': np.array([1, 2, 3, 4]), 'date': datetime.date(2021, 1, 1), 'starting_point': 0},
         {'instance': np.array([5, 6, 7, 8]), 'date': datetime.date(2021, 1, 2), 'starting_point': 4}]
    """

    def __init__(self, subsequence: Optional[Subsequence] = None) -> None:
        """
        Parameters:
            * subsequence: Optional[Subsequence], the subsequence to add to the sequence

        Raises:
            TypeError: if the parameter is not of the correct type

        Examples:
            >>> sequence1 = Sequence(subsequence=Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence2 = Sequence()
            >>> sequence2.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
        """

        # Initialize the length of the sequence
        self.__length = None

        # Check if the subsequence is a Subsequence instance
        if subsequence is not None:
            self.__check_validity_params(subsequence)

            # Make a deep copy of the subsequence
            new_subsequence = deepcopy(subsequence)
            self.__list_sequences: list[Subsequence] = [new_subsequence]

            # Set the length of the sequence
            self.__length: int = len(subsequence)

        # If the subsequence is None, initialize an empty list
        else:
            self.__list_sequences: list[Subsequence] = []

    def __repr__(self):
        """
        Returns the string representation of the sequence

        Returns:
            str. The string representation of the sequence

        Examples:
            >>> sequence = Sequence(subsequence=Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> print(sequence)
            Sequence(
                length_subsequences=4,
                list_sequences=[
                    Subsequence(
                        instance=np.array([1, 2, 3, 4]),
                        date=datetime.date(2021, 1, 1),
                        starting_point=0
                    )
                ]
            )
        """

        out_string = "Sequence(\n\tlength_subsequences=" + f"{self.__length}" + "\n\tlist_sequences=[\n"
        for seq in self.__list_sequences:
            out_string += f" {seq},\n"

        out_string = out_string[:-2] + out_string[-1] + "]"
        return out_string

    def __str__(self):
        """
        Returns the string representation of the sequence

        Returns:
            `str`. The string representation of the sequence

        Examples:
            >>> sequence = Sequence(length=4, subsequence=Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> print(sequence)
            Sequence(
                length_subsequences=4,
                list_sequences=[
                    Subsequence(
                        instance=np.array([1, 2, 3, 4]),
                        date=datetime.date(2021, 1, 1),
                        starting_point=0
                    )
                ]
            )
        """

        out_string = "Sequence(\n\tlength_subsequences=" + f"{self.__length}" + "\n\tlist_sequences=[\n"
        for seq in self.__list_sequences:
            out_string += f" {seq},\n"

        out_string = out_string[:-2] + out_string[-1] + "]"
        return out_string

    def __len__(self) -> int:
        """
        Returns the number of `Subsequence` instances in the `Sequence`

        Returns:
            `int`. The number of subsequences in the sequence

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> len(sequence)
            2
        """

        return len(self.__list_sequences)

    def __getitem__(self, index: int) -> 'Subsequence':
        """
        Get the subsequence at the specified index in the sequence

        Parameters:
            * index: `int`. The index of the subsequence

        Returns:
            `Subsequence`. The subsequence at the specified index in the sequence

        Raises:
            TypeError: if the parameter is not of the correct type
            IndexError: if the index is out of range

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence[0]
            Subsequence(instance=np.array([1, 2, 3, 4]), date=datetime.date(2021, 1, 1), starting_point=0)
        """
        # Check if the index is an integer
        if not isinstance(index, int):
            raise TypeError(f"index must be an integer. Got {type(index).__name__} instead")

        # Check if the index is within the range of the list
        if not 0 <= index < len(self.__list_sequences):
            raise IndexError(f"index {index} out of range (0, {len(self.__list_sequences) - 1})")

        return self.__list_sequences[index]

    def __setitem__(self, index: int, value: 'Subsequence') -> None:
        """
        Set the value of the subsequence at the specified index in the sequence

        Parameters:
            * index: int. The index of the subsequence
            * value: Subsequence. The new subsequence

        Raises:
            TypeError: if the parameter is not of the correct type
            IndexError: if the index is out of range

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence[0] = Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)
            >>> sequence[0]
            Subsequence(instance=np.array([5, 6, 7, 8]), date=datetime.date(2021, 1, 2), starting_point=4)
        """

        # Check if the new_sequence is a Subsequence instance
        if not isinstance(value, Subsequence):
            raise TypeError(f"new_sequence must be an instance of Subsequence. Got {type(value).__name__} instead")

        # Check if the index is an integer
        if not isinstance(index, int):
            raise TypeError(f"index must be an integer. Got {type(index).__name__} instead")

        # Check if the index is within the range of the list
        if not 0 <= index < len(self.__list_sequences):
            raise IndexError(f"index {index} out of range (0, {len(self.__list_sequences) - 1})")

        self.__list_sequences[index] = value

    def __iter__(self):
        """
        Returns an iterator for each subsequence in the sequence

        Returns:
            iter. An iterator for each subsequence in the sequence

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> for subsequence in sequence:
            ...     print(subsequence)
            Subsequence(instance=np.array([1, 2, 3, 4]), date=datetime.date(2021, 1, 1), starting_point=0)
            Subsequence(instance=np.array([5, 6, 7, 8]), date=datetime.date(2021, 1, 2), starting_point=4)
        """

        return iter(self.__list_sequences)

    def __contains__(self, item: 'Subsequence') -> bool:
        """
        Check if the subsequence exists in the sequence

        Parameters:
            * item: `Subsequence`. The subsequence to check

        Returns:
            `bool`. `True` if the subsequence exists, `False` otherwise

        Raises:
            TypeError: if the parameter is not an instance of Subsequence

        Examples:
            >>> sequence = Sequence(subsequence=Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 1))
            >>> Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0) in sequence
            True

        """
        # Check if the new_sequence is a Subsequence instance
        if not isinstance(item, Subsequence):
            raise TypeError(f"new_sequence must be an instance of Subsequence. Got {type(item).__name__} instead")

        return item in self.__list_sequences

    def __delitem__(self, index: int) -> None:
        """
        Deletes the subsequence at the specified index in the sequence

        Parameters:
            * index: `int`. The index of the subsequence to delete

        Raises:
            TypeError: if the parameter is not of the correct type
            IndexError: if the index is out of range
        """
        if not isinstance(index, int):
            raise TypeError(f"index must be an integer. Got {type(index).__name__} instead")

        if not 0 <= index < len(self.__list_sequences):
            raise IndexError(f"index {index} out of range (0, {len(self.__list_sequences) - 1})")

        del self.__list_sequences[index]

    def __add__(self, other: 'Sequence') -> 'Sequence':
        """
        Concatenates two sequences together with the operator +

        Parameters:
            * other: `Sequence`. The sequence to concatenate

        Returns:
            `Sequence`. The concatenated sequence

        Raises:
            TypeError: if the parameter is not an instance of `Sequence`

        Examples:
            >>> sequence1 = Sequence()
            >>> sequence1.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence2 = Sequence()
            >>> sequence2.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 1))
            >>> new_sequence = sequence1 + sequence2
            >>> print(new_sequence)
            Sequence(
                list_sequences=[
                    Subsequence(
                        instance=np.array([1, 2, 3, 4]),
                        date=datetime.date(2021, 1, 1),
                        starting_point=0
                    ),
                    Subsequence(
                        instance=np.array([5, 6, 7, 8]),
                        date=datetime.date(2021, 1, 2),
                        starting_point=1
                    )
                ]
            )
        """
        if not isinstance(other, Sequence):
            raise TypeError(f"other must be an instance of Sequence. Got {type(other).__name__} instead")

        if self.__length != other.length_subsequences:
            raise ValueError(
                f"The length of the subsequences must be the same. Got {self.__length} and {other.length_subsequences} instead")

        # new_sequence = Sequence()
        # new_sequence.__list_sequences = self.__list_sequences + other.__list_sequences
        # new_sequence.__length = self.__length
        new_sequence = Sequence()
        starting_points_self = self.get_starting_points()
        starting_points_other = other.get_starting_points()
        min_sp = min(starting_points_self + starting_points_other)
        max_sp = max(starting_points_self + starting_points_other)
        for sp in range(min_sp, max_sp + 1):
            self_subsequence = self.get_by_starting_point(sp)
            other_subsequence = other.get_by_starting_point(sp)

            if self_subsequence is not None and other_subsequence is not None:
                if not self_subsequence == other_subsequence:
                    raise ValueError(
                        f"The subsequences must be the same at starting point {sp}. Got {self_subsequence} and {other_subsequence} instead")

                new_sequence.add_sequence(self_subsequence)

            if self_subsequence is not None and other_subsequence is None:
                new_sequence.add_sequence(self_subsequence)

            if other_subsequence is not None and self_subsequence is None:
                new_sequence.add_sequence(other_subsequence)

        return new_sequence

    def __eq__(self, other: 'Sequence') -> bool:
        """
        Check if the sequence is equal to another sequence

        Parameters:
            * other: `Sequence`. The sequence to compare

        Returns:
            `bool`. True if the sequences are equal, False otherwise

        Raises:
            TypeError: if the parameter is not of the correct type

        Examples:
            >>> sequence1 = Sequence()
            >>> sequence1.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence2 = Sequence()
            >>> sequence2.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence1 == sequence2
            True
        """

        # Check if the parameter is an instance of Sequence
        if not isinstance(other, Sequence):
            raise TypeError(f"other must be an instance of Sequence. Got {type(other).__name__} instead")

        # Check if the subsequences are equal
        return np.array_equal(self.get_subsequences(), other.get_subsequences())

    @property
    def length_subsequences(self) -> int:
        """
        Getter that returns the length of the subsequences in the sequence

        Returns:
            `int`. The length of the subsequences in the sequence

        Examples:
            >>> sequence = Sequence(subsequence=Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.length_subsequences
            4
        """

        return self.__length

    def __check_validity_params(self, subsequence: Subsequence) -> None:
        """
        Check if the parameters are valid

        Parameters:
            * subsequence: `Subsequence`. The subsequence to add to the sequence

        Raises:
            TypeError: if the parameters are not of the correct type
            ValueError: if the length of the subsequence is not the same as the length of the sequence
        """

        # Check if the subsequence is a Subsequence instance
        if not isinstance(subsequence, Subsequence):
            raise TypeError(f"subsequence must be an instance of Subsequence. Got {type(subsequence).__name__} instead")

        # Check if the length of the subsequence is the same as the length of the sequence
        if self.__length is not None and len(subsequence) != self.__length:
            raise ValueError(
                f"The length of the subsequence must be the same as the length of the Sequence. Got {len(subsequence)} instead of {self.__length}")

    def _already_exists(self, subsequence: 'Subsequence') -> bool:
        """
        Check if the subsequence already exists in the sequence

        Parameters:
            * subsequence: `Subsequence`. The subsequence to check

        Returns:
            `bool`. True if the `subsequence` already exists, `False` otherwise

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence._already_exists(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            True
        """

        self_collection = self.to_collection()
        new_self_collection = []

        # Is necessary to convert the arrays to list for checking properly if the new sequence exists
        for idx, dictionary in enumerate(self_collection):
            dictionary["instance"] = dictionary["instance"]
            new_self_collection.append(dictionary)

        # convert to collection and transform from array to list
        collection = subsequence.to_collection()
        collection = {"instance": collection["instance"].tolist()}

        return collection in new_self_collection

    def add_sequence(self, new: 'Subsequence') -> None:
        """
        Adds a subsequence to the sequence

        Parameters:
            * new: `Subsequence`. The subsequence to add

        Raises:
            TypeError: if the parameter is not of the correct type
            RuntimeError: if the subsequence already exists
            ValueError: if the length of the subsequence is not the same as the length of the sequence

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> print(sequence)
            Sequence(
                length_subsequences=4,
                list_sequences=[
                    Subsequence(
                        instance=np.array([1, 2, 3, 4]),
                        date=datetime.date(2021, 1, 1),
                        starting_point=0
                    ),
                    Subsequence(
                        instance=np.array([5, 6, 7, 8]),
                        date=datetime.date(2021, 1, 2),
                        starting_point=4
                    )
                ]
            )
        """
        # Check if the new sequence is a Subsequence instance
        if not isinstance(new, Subsequence):
            raise TypeError(f"new has to be an instance of Subsequence. Got {type(new).__name__} instead")

        # Check if the new sequence already exists
        if self._already_exists(new):
            raise RuntimeError("new sequence already exists ")

        # Check if the length of the subsequence is the same as the length of the sequence
        if self.__length is not None and len(new) != self.__length:
            raise ValueError(
                f"The length of the subsequence must be the same as the length of the Sequence. Got {len(new)} instead of {self.__length}")

        # If the sequence is empty, set the length of the sequence
        if len(self.__list_sequences) == 0:
            self.__length = len(new)

        self.__list_sequences.append(new)

    def get_by_starting_point(self, starting_point: int) -> Optional['Subsequence']:
        """
        Returns the subsequence with the specified starting point

        Parameters:
            * starting_point: `int`. The starting point of the subsequence

        Returns:
            Optional[Subsequence]. The subsequence with the specified starting point if it exists. Otherwise, None

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 1))
            >>> sequence.get_by_starting_point(0)
            Subsequence(instance=np.array([1, 2, 3, 4]), date=datetime.date(2021, 1, 1), starting_point=0)

            >>> sequence.get_by_starting_point(2)
            None
        """

        for subseq in self.__list_sequences:
            if subseq.get_starting_point() == starting_point:
                return subseq

        return None

    def set_by_starting_point(self, starting_point: int, new_sequence: 'Subsequence') -> None:
        """
        Sets the subsequence with the specified starting point

        Parameters:
            * starting_point: int. The starting point of the subsequence
            * new_sequence: Subsequence. The new subsequence

        Raises:
            TypeError: if the parameter is not of the correct type
            ValueError: if the starting point does not exist

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 1))
            >>> sequence.set_by_starting_point(0, Subsequence(np.array([9, 10, 11, 12]), datetime.date(2021, 1, 3), 0))
            >>> sequence.get_by_starting_point(0)
            Subsequence(instance=np.array([9, 10, 11, 12]), date=datetime.date(2021, 1, 3), starting_point=0)
        """

        # Check if the new_sequence is a Subsequence instance
        if not isinstance(new_sequence, Subsequence):
            raise TypeError(
                f"new_sequence must be an instance of Subsequence. Got {type(new_sequence).__name__} instead")

        # Iterate through the list to find the subsequence with the matching starting point
        for i, subseq in enumerate(self.__list_sequences):
            if subseq.get_starting_point() == starting_point:
                # Replace the found subsequence with the new one
                self.__list_sequences[i] = new_sequence
                return

        # If not found, raise an error indicating the starting point does not exist
        raise ValueError(
            f"The starting point {starting_point} doesn't exist. The available starting points are {self.get_starting_points()}")

    def get_starting_points(self, to_array: bool = False) -> list[int]:
        """
        Returns the starting points of the subsequences

        Parameters:
            * to_array: `bool`. If True, returns the starting points as a numpy array. Default is `False`

        Returns:
             `list[int]`. The starting points of the subsequences

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> sequence.get_starting_points()
            [0, 4]
        """

        sequence_starting_points = [subseq.get_starting_point() for subseq in self.__list_sequences]

        if to_array:
            return np.array(sequence_starting_points)

        return sequence_starting_points

    def get_dates(self) -> list[datetime.date]:
        """
        Returns the dates of the subsequences

        Returns:
             `list[datetime.date]`. The dates of the subsequences

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 1))
            >>> sequence.get_dates()
            [datetime.date(2021, 1, 1), datetime.date(2021, 1, 2)]
        """

        return [subseq.get_date() for subseq in self.__list_sequences]

    def get_subsequences(self, to_array: bool = False) -> Union[list[np.ndarray], np.ndarray]:
        """
        Returns the instances of the subsequences

        Returns:
             `list[np.ndarray]`. The instances of the subsequences

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 1))
            >>> sequence.get_subsequences()
            [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])]

            >>> sequence.get_subsequences(to_array=True)
            array([np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8]])
        """
        subsequences = [subseq.get_instance() for subseq in self.__list_sequences]

        if to_array:
            return np.array(subsequences)

        return subsequences

    def remove_subsequence(self, subsequence: Union[Subsequence, int]) -> None:
        not_exists_msg = "subsequence does not exist in the sequence"

        if not isinstance(subsequence, (Subsequence, int)):
            raise TypeError(
                f"subsequence must be an instance of Subsequence or integer. Got {type(subsequence).__name__} instead")

        if isinstance(subsequence, int):
            if subsequence not in self.get_starting_points():
                raise IndexError(not_exists_msg)

            idx = self.get_starting_points().index(subsequence)
            del self.__list_sequences[idx]

        if isinstance(subsequence, Subsequence):
            if subsequence not in self.__list_sequences:
                raise IndexError(not_exists_msg)

            idx = self.__list_sequences.index(subsequence)
            del self.__list_sequences[idx]

    def extract_components(self, flatten: bool = False) -> tuple[np.ndarray, list[datetime.date], list[int]]:
        """
        Extract the components of a sequence.

        Returns:
            `tuple[np.ndarray, list[datetime.date], list[int]]`. The subsequence, dates, and starting points of the sequence.

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3]), datetime.date(2024, 1, 1), 0))
            >>> subsequence, dates, starting_points = sequence.extract_components()
            >>> print(subsequence)
            [1 2 3]

            >>> print(dates)
            [datetime.date(2024, 1, 1)]

            >>> print(starting_points)
            [0]
        """

        subsequence = self.get_subsequences(to_array=True)

        if flatten:
            subsequence = subsequence.flatten()

        dates = self.get_dates()
        starting_points = self.get_starting_points()

        return subsequence, dates, starting_points

    def to_collection(self) -> list[dict]:
        """
        Returns the sequence as a list of dictionaries

        Returns:
             `list[dict]`. The sequence as a list of dictionaries

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 1))
            >>> sequence.to_collection()
            [{'instance': np.array([1, 2, 3, 4]), 'date': datetime.date(2021, 1, 1), 'starting_point': 0},
             {'instance': np.array([5, 6, 7, 8]), 'date': datetime.date(2021, 1, 2), 'starting_point': 1}]
        """

        collection = []
        for subseq in self.__list_sequences:
            collection.append({
                'instance': subseq.get_instance().tolist(),
                'date': subseq.get_date().strftime("%Y/%m/%d, %H:%M:%S"),
                'starting_point': subseq.get_starting_point()
            })

        return collection


class Cluster:
    """
    Represents a cluster of subsequences from a sequence and a centroid.

    Parameters:
        * ``centroid: np.ndarray``, the centroid of the cluster
        * ``instances: Sequence``, the sequence of subsequences

    Properties:
    ________
        **Getters**:
            * ``centroid: np.ndarray``, the centroid of the cluster
            * ``length_cluster_subsequences: int``, the length of each subsequence in the cluster

        **Setters**:
            * ``centroid: np.ndarray | Subsequence``, the centroid of the cluster


    Public Methods:
    ________

        * ``add_instance(new_subsequence: Subsequence)``: adds a subsequence to the cluster
        * ``update_centroid()``: updates the centroid of the cluster
        * ``get_sequences() -> Sequence``: returns the sequences of the cluster
        * ``get_starting_points() -> list[int]``: returns the starting points of the subsequences
        * ``get_dates() -> list[date]``: returns the dates of the subsequences
        * ``cumulative_magnitude() -> float``: returns the cumulative magnitude of the cluster
        * ``fusion(other: Cluster) -> Cluster``: fuses two clusters together
        * ``is_similar(other: Cluster, distance_threshold: Union[float, int]=0.001) -> bool``: checks if two clusters are similar based on the distance threshold


    Examples:

        >>> sequence = Sequence()
        >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
        >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))

        >>> cluster = Cluster(np.array([1, 1, 1, 1]), sequence)
        >>> cluster.get_sequences().to_collection()
        [{'instance': np.array([1, 2, 3, 4]), 'date': datetime.date(2021, 1, 1), 'starting_point': 0},
         {'instance': np.array([5, 6, 7, 8]), 'date': datetime.date(2021, 1, 2), 'starting_point': 4}]

        >>> cluster.get_starting_points()
        [0, 4]

        >>> cluster.get_dates()
        [datetime.date(2021, 1, 1), datetime.date(2021, 1, 2)]

        >>> cluster.centroid
        np.array([1, 1, 1, 1])

        >>> cluster.centroid = np.array([1, 2, 3, 4])
        >>> cluster.centroid
        np.array([1, 2, 3, 4])

        >>> cluster.update_centroid()
        >>> cluster.centroid
        np.array([3, 4, 5, 6])

        >>> cluster.cumulative_magnitude()
        12.0

        >>> cluster2 = Cluster(np.array([1, 1, 1, 1]), sequence)
        >>> cluster.is_similar(cluster2, distance_threshold=0.001)
        True

        >>> cluster3 = Cluster(np.array([5,5,5,5]), sequence)
        >>> cluster3.is_similar(cluster2, distance_threshold=0.001)
        False
    """

    def __init__(self, centroid: np.ndarray, instances: 'Sequence') -> None:
        """
        Parameters:
            * centroid: `np.ndarray`, the centroid of the cluster
            * instances: `Sequence`, the sequence of subsequences

        Raises:
            TypeError: if the centroid is not an instance of `np.ndarray` or the instances are not an instance of `Sequence`

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> cluster = Cluster(4, np.array([3, 4, 5, 6]), sequence)
        """

        # Check the validity of the parameters
        self.__check_validity_params(centroid, instances)

        # Make a deep copy of the instances to avoid modifying the original sequence
        new_instances = deepcopy(instances)

        # Set the length centroid and the instances
        self.__length: int = new_instances.length_subsequences
        self.__centroid: np.ndarray = centroid
        self.__instances: Sequence = new_instances

    def __str__(self):
        """
        Returns the string representation of the cluster

        Returns:
            `str`. The string representation of the cluster

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> cluster = Cluster(np.array([3, 4, 5, 6]), sequence)
            >>> print(cluster)
            Cluster(
                -Centroid: np.array([3, 4, 5, 6])
                -Instances: [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])]
                -Dates: [datetime.date(2021, 1, 1), datetime.date(2021, 1, 2)]
                -Starting Points: [0, 4]
            )
        """

        out_string = f"Cluster(\n\t -Centroid: {self.__centroid} \n"
        out_string += f"\t -Instances: {[instance for instance in self.__instances.get_subsequences()]}\n"
        out_string += f"\t -Dates: {[date for date in self.__instances.get_dates()]}\n"
        out_string += f"\t -Starting Points: {[sp for sp in self.__instances.get_starting_points()]}\n)"
        return out_string

    def __repr__(self):
        """
        Returns the string representation of the cluster

        Returns:
            `str`. The string representation of the cluster

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> cluster = Cluster(np.array([3, 4, 5, 6]), sequence)
            >>> print(cluster)
            Cluster(
                -Centroid: np.array([3, 4, 5, 6])
                -Instances: [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])]
                -Dates: [datetime.date(2021, 1, 1), datetime.date(2021, 1, 2)]
                -Starting Points: [0, 4]
            )
        """

        out_string = f"Cluster(\n\t -Centroid: {self.__centroid} \n"
        out_string += f"\t -Instances: {[instance for instance in self.__instances.get_subsequences()]}\n"
        out_string += f"\t -Dates: {[date for date in self.__instances.get_dates()]}\n"
        out_string += f"\t -Starting Points: {[sp for sp in self.__instances.get_starting_points()]}\n)"
        return out_string

    def __len__(self) -> int:
        """
        Returns the number of instances in the cluster

        Returns:
            `int`. The number of instances in the cluster

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> cluster = Cluster(np.array([3, 4, 5, 6]), sequence)
            >>> len(cluster)
            2
        """

        return len(self.__instances)

    def __getitem__(self, index: int) -> 'Subsequence':
        """
        Get the subsequence at the specified index in the cluster

        Parameters:
            * index: `int`. The index of the subsequence

        Returns:
            `Subsequence`. The subsequence at the specified index in the cluster

        Raises:
            TypeError: if the parameter is not of the correct type
            IndexError: if the index is out of range

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> cluster = Cluster(np.array([3, 4, 5, 6]), sequence)
            >>> cluster[0]
            Subsequence(instance=np.array([1, 2, 3, 4]), date=datetime.date(2021, 1, 1), starting_point=0)
        """

        # Check if the index is an integer
        if not isinstance(index, int):
            raise TypeError(f"index must be an integer. Got {type(index).__name__} instead")

        # Check if the index is within the range of the list
        if not 0 <= index < len(self.__instances):
            raise IndexError(f"index {index} out of range (0, {len(self.__instances) - 1})")

        return self.__instances[index]

    def __setitem__(self, index: int, value: 'Subsequence') -> None:
        """
        Set the value of the subsequence at the specified index in the cluster

        Parameters:
            * index: `int`. The index of the subsequence
            * value: `Subsequence`. The new subsequence

        Raises:
            TypeError: if the parameter is not of the correct type
            IndexError: if the index is out of range

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> cluster = Cluster(np.array([3, 4, 5, 6]), sequence)
            >>> cluster[0] = Subsequence(np.array([9, 10, 11, 12]), datetime.date(2021, 1, 3), 0)
            >>> cluster[0]
            Subsequence(instance=np.array([9, 10, 11, 12]), date=datetime.date(2021, 1, 3), starting_point=0)
        """

        # Check if the new_sequence is a Subsequence instance
        if not isinstance(value, Subsequence):
            raise TypeError(f"new_sequence must be an instance of Subsequence. Got {type(value).__name__} instead")

        # Check if the index is an integer
        if not isinstance(index, int):
            raise TypeError(f"index must be an integer. Got {type(index).__name__} instead")

        # Check if the index is within the range of the list
        if not 0 <= index < len(self.__instances):
            raise IndexError(f"index {index} out of range (0, {len(self.__instances) - 1})")

        self.__instances[index] = value

    def __iter__(self) -> iter:
        """
        Returns an iterator for each subsequence in the cluster's instances

        Returns:
            `iter`. An iterator for each subsequence in the cluster's instances

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> cluster = Cluster(np.array([3, 4, 5, 6]), sequence)
            >>> for subsequence in cluster:
            ...     print(subsequence)
            Subsequence(instance=np.array([1, 2, 3, 4]), date=datetime.date(2021, 1, 1), starting_point=0)
            Subsequence(instance=np.array([5, 6, 7, 8]), date=datetime.date(2021, 1, 2), starting_point=4)
        """

        return iter(self.__instances)

    def __contains__(self, item: 'Subsequence') -> bool:
        """
        Check if the subsequence exists in the cluster's instances

        Parameters:
            * item: `Subsequence`. The subsequence to check

        Returns:
            `bool`. `True` if the subsequence exists, `False` otherwise

        Raises:
            TypeError: if the parameter is not an instance of `Subsequence`

        Examples:
            >>> sequence = Sequence(subsequence=Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> cluster = Cluster(np.array([3, 4, 5, 6]), sequence)
            >>> item = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> item in cluster
            True
        """

        # Check if the new_sequence is a Subsequence instance
        if not isinstance(item, Subsequence):
            raise TypeError(f"new_sequence must be an instance of Subsequence. Got {type(item).__name__} instead")

        return item in self.__instances

    def __delitem__(self, index: int) -> None:
        """
        Deletes the subsequence at the specified index in the cluster's instances

        Parameters:
            * index: `int`. The index of the subsequence to delete

        Raises:
            TypeError: if the parameter is not of the correct type
            IndexError: if the index is out of range

        Examples:
            >>> sequence = Sequence(subsequence=Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> cluster = Cluster(np.array([3, 4, 5, 6]), sequence)
            >>> del cluster[0]
            >>> print(cluster)
            Cluster(
                -Centroid: np.array([3, 4, 5, 6])
                -Instances: [np.array([5, 6, 7, 8])]
                -Dates: [datetime.date(2021, 1, 2)]
                -Starting Points: [4]
            )
        """

        if not isinstance(index, int):
            raise TypeError(f"index must be an integer. Got {type(index).__name__} instead")

        if not 0 <= index < len(self.__instances):
            raise IndexError(f"index {index} out of range (0, {len(self.__instances) - 1})")

        del self.__instances[index]

    def __add__(self, other: 'Cluster') -> 'Cluster':
        """
        Concatenates two clusters together with the operator + and updates the centroid

        Parameters:
            * other: `Cluster`. The cluster to concatenate

        Returns:
            `Cluster`. The concatenated cluster

        Raises:
            TypeError: if the parameter is not an instance of `Cluster`
            ValueError: if the clusters do not have the same length of instances in each `Subsequence`

        Examples:
            >>> sequence1 = Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence2 = Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 1))
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), sequence1)
            >>> cluster2 = Cluster(np.array([7, 8, 9, 10]), sequence2)
            >>> new_cluster = cluster1 + cluster2
            >>> print(new_cluster)
            Cluster(
                -Centroid: np.array([5, 6, 7, 8])
                -Instances: [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])]
                -Dates: [datetime.date(2021, 1, 1), datetime.date(2021, 1, 2)]
                -Starting Points: [0, 1]
            )
        """

        if not isinstance(other, Cluster):
            raise TypeError(f"other must be an instance of Cluster. Got {type(other).__name__} instead")

        # Check if the lengths of the subsequences from the instances of each cluster match
        if len(self.__instances[0]) != len(other.get_sequences()[0]):
            raise ValueError(
                f"clusters do not have the same length of instances in each Subsequence. Expected len={len(self.__instances[0])} but got len={len(other.get_sequences()[0])}")

        new_instances = self.__instances + other.get_sequences()
        new_centroid = np.mean(new_instances.get_subsequences(), axis=0)
        return Cluster(centroid=new_centroid, instances=new_instances)

    def __eq__(self, other: Union['Cluster', None]) -> bool:
        """
        Check if the cluster is equal to another cluster with the operator ==

        Parameters:
            * other: `Cluster`. The cluster to check

        Returns:
            `bool`. `True` if the clusters are equal, `False` otherwise

        Raises:
            TypeError: if the parameter is not an instance of `Cluster`

        Examples:
            >>> sequence1 = Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence2 = Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 1))
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), sequence1)
            >>> cluster2 = Cluster(np.array([3, 4, 5, 6]), sequence2)
            >>> cluster1 == cluster2
            False

            >>> sequence3 = Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence4 = Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 1))
            >>> cluster3 = Cluster(np.array([3, 4, 5, 6]), sequence3)
            >>> cluster4 = Cluster(np.array([3, 4, 5, 6]), sequence4)
            >>> cluster3 == cluster4
            True
        """

        # Check if the other is a Cluster instance
        if not isinstance(other, Cluster):
            if other is None:
                return False

            raise TypeError(f"other must be an instance of Cluster. Got {type(other).__name__} instead")

        # Check if the centroid and the instances are equal
        if not np.array_equal(self.__centroid, other.centroid):
            return False

        # Check if the number of instances is equal
        if len(self.__instances) != len(other.get_sequences()):
            return False

        # Check if the length of the instances is equal
        if self.__length != other.length_cluster_subsequences:
            return False

        # Check if the instances are equal
        if not np.array_equal(self.__instances.get_subsequences(), other.get_sequences().get_subsequences()):
            return False

        # Check if the dates are equal
        if not np.array_equal(self.__instances.get_dates(), other.get_sequences().get_dates()):
            return False

        # Check if the starting points are equal
        if not np.array_equal(self.__instances.get_starting_points(), other.get_sequences().get_starting_points()):
            return False

        return True

    @staticmethod
    def __check_validity_params(centroid: np.ndarray, instances: 'Sequence') -> None:
        """
        Check if the parameters are valid

        Parameters:
            * centroid: `np.ndarray`. The centroid of the cluster
            * instances: `Sequence`. The sequence of subsequences

        Raises:
            TypeError: if the parameters are not of the correct type
            ValueError: if the length of the centroid is different from the length of the subsequences
        """

        # Check if the centroid is an instance of np.ndarray
        if not isinstance(centroid, np.ndarray):
            raise TypeError(f"centroid must be an instance of np.ndarray. Got {type(centroid).__name__}")

        # Check if the instances is an instance of Sequence
        if not isinstance(instances, Sequence):
            raise TypeError(f"instances must be an instance of Sequence. Got {type(instances).__name__}")

        # Check if the length of the centroid is the same as the length of the subsequences
        if len(centroid) != instances.length_subsequences:
            raise ValueError(
                f"The length of the centroid must be equal to the length of the subsequences. Got {len(centroid)} and {instances.length_subsequences} instead")

    @property
    def centroid(self) -> np.ndarray:
        """
        Returns the centroid of the cluster

        Returns:
             np.ndarray. The centroid of the cluster

        Examples:
            >>> sequence = Sequence(Subsequence(np.array([1, 2, 3, 4, 5]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8, 9]), datetime.date(2021, 1, 2), 4))
            >>> cluster = Cluster(np.array([3, 4, 5, 6, 7]), sequence)
            >>> cluster.centroid
            np.array([3, 4, 5, 6, 7])
        """
        return self.__centroid

    @centroid.setter
    def centroid(self, subsequence: np.ndarray | Subsequence) -> None:
        """
        Sets the value of the centroid from the cluster with a subsequence

        Parameters:
            * subsequence: `Union[Subsequence|np.ndarray]`. The subsequence to set as the centroid

        Raises:
            TypeError: if the parameter is not a `Subsequence` or a numpy array
            ValueError: if the length of the subsequence is different from the length of the subsequences

        Examples:
            >>> sequence = Sequence(Subsequence(np.array([1, 2, 3, 4, 5]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8, 9]), datetime.date(2021, 1, 2), 4))
            >>> cluster = Cluster(np.array([3, 4, 5, 6, 7]), sequence)
            >>> cluster.centroid = np.array([1, 2, 3, 4, 5])
            >>> cluster.centroid
            np.array([1, 2, 3, 4, 5])
        """

        # Check if the length of the subsequence is the same as the length of the subsequences
        if len(subsequence) != self.__length:
            raise ValueError(f"the length of the subsequence must be {self.__length}. Got {len(subsequence)} instead")

        # Set the centroid if it is an instance of Subsequence
        if isinstance(subsequence, Subsequence):
            self.__centroid = subsequence.get_instance()

        # Set the centroid if it is a numpy array
        if isinstance(subsequence, np.ndarray):
            self.__centroid = subsequence

        # Raise an error if the parameter is not a Subsequence or a numpy array
        if not isinstance(subsequence, Subsequence) and not isinstance(subsequence, np.ndarray):
            raise TypeError(
                f"subsequence must be an instance of Subsequence or a numpy array. Got {type(subsequence).__name__} instead")

    @property
    def length_cluster_subsequences(self) -> int:
        """
        Getter that returns the length of the instances in the cluster

        Returns:
            `int`. The length of the instances in the cluster

        Examples:
            >>> sequence = Sequence(Subsequence(np.array([1, 2, 3, 4, 5]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8, 9]), datetime.date(2021, 1, 2), 4))
            >>> cluster = Cluster(np.array([3, 4, 5, 6, 7]), sequence)
            >>> cluster.length_cluster_subsequences
            5
        """

        return self.__length

    def add_instance(self, new_instance: 'Subsequence') -> None:
        """
        Adds a subsequence to the instances of the cluster

        Parameters:
            * new_instance: `Subsequence`. The subsequence to add

        Raises:
            TypeError: if the parameter is not of the correct type
            ValueError: if the subsequence is already an instance of the cluster

        Examples:
            >>> sequence = Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> cluster = Cluster(np.array([3, 4, 5, 6]), sequence)
            >>> cluster.add_instance(Subsequence(np.array([9, 10, 11, 12]), datetime.date(2021, 1, 3), 0))
            >>> print(cluster)
            Cluster(
                -Centroid: np.array([3, 4, 5, 6])
                -Instances: [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8]), np.array([9, 10, 11, 12])]
                -Dates: [datetime.date(2021, 1, 1), datetime.date(2021, 1, 2), datetime.date(2021, 1, 3)]
                -Starting Points: [0, 4, 0]
            )
        """

        # Check if the new_sequence is a Subsequence instance
        if not isinstance(new_instance, Subsequence):
            raise TypeError(
                f"new sequence must be an instance of Subsequence. Got {type(new_instance).__name__} instead")

        # Check if the new sequence is already an instance of the cluster
        if self.__instances._already_exists(new_instance):
            raise ValueError(f"new sequence is already an instance of the cluster")

        # Check if the length of the new instance is the same as the length of the subsequences
        if len(new_instance) != self.__length:
            raise ValueError(
                f"the length of the subsequence must be {self.__length}. Got {len(new_instance)} instead")

        self.__instances.add_sequence(new_instance)

    def get_sequences(self) -> 'Sequence':
        """
        Returns the sequence of the cluster

        Returns:
             `Sequence`. The sequence of the cluster

        Examples:
            >>> sequence = Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> cluster = Cluster(np.array([3, 4, 5, 6]), sequence)
            >>> cluster.get_sequences()
            Sequence(
                length_subsequences=4,
                list_sequences=[
                    Subsequence(
                        instance=np.array([1, 2, 3, 4]),
                        date=datetime.date(2021, 1, 1),
                        starting_point=0
                    ),
                    Subsequence(
                        instance=np.array([5, 6, 7, 8]),
                        date=datetime.date(2021, 1, 2),
                        starting_point=4
                    )
                ]
            )
        """

        return self.__instances

    def update_centroid(self) -> None:
        """
        Updates the centroid of the cluster with the mean of the instances
        """

        self.__centroid = np.mean(self.__instances.get_subsequences(), axis=0)

    def get_starting_points(self, to_array: bool = False) -> Union[list[int], np.ndarray]:
        """
        Returns the starting points of the subsequences

        Parameters:
            * to_array: `bool`. If True, the starting points are returned as a numpy array. Default is False

        Returns:
             `list[int] | np.ndarray`. The starting points of the subsequences

        Examples:
            >>> sequence = Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> cluster = Cluster(np.array([3, 4, 5, 6]), sequence)
            >>> cluster.get_starting_points()
            [0, 4]
        """

        # Return the starting points as a numpy array
        if to_array:
            return np.array(self.__instances.get_starting_points())

        # Return the starting points as a list
        return self.__instances.get_starting_points()

    def get_dates(self) -> list[datetime.date]:
        """
        Returns the dates of the subsequences

        Returns:
             `list[datetime.date]`. The dates of the subsequences

        Examples:
            >>> sequence = Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> cluster = Cluster(np.array([3, 4, 5, 6]), sequence)
            >>> cluster.get_dates()
            [datetime.date(2021, 1, 1), datetime.date(2021, 1, 2)]
        """

        return self.__instances.get_dates()

    def cumulative_magnitude(self) -> float | int:
        """
        Returns the magnitude's sum of the subsequences that belongs to the instances within the cluster

        Returns:
             `float`. The magnitude's sum of the subsequences

        Examples:
            >>> sequence = Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> cluster = Cluster(np.array([3, 4, 5, 6]), sequence)
            >>> cluster.cumulative_magnitude()
            12.0
        """

        return sum([subsequence.magnitude() for subsequence in self.__instances])

    def fusion(self, other: 'Cluster') -> 'Cluster':
        """
        Fusion of two clusters.

        Parameters:
            * other: `Cluster`. The first cluster to fuse.

        Returns:
            `Cluster`. The fusion of the two clusters.

        Raises:
            TypeError: If the other is not an instance of Cluster.

        Examples:
            >>> cluster1 = Cluster(centroid=np.array([1, 2, 3]), instances=Sequence(Subsequence(np.array([1, 2, 3]), date=datetime.date(2024, 1, 1), starting_point=0)))
            >>> cluster2 = Cluster(centroid=np.array([3, 2, 1]), instances=Sequence(Subsequence(np.array([3, 2, 1]), date=datetime.date(2024, 1, 1), starting_point=1))
            >>> fusion = cluster1.fusion(cluster2)
            >>> print(fusion)
            Cluster(
                - centroid=np.array([2. 2. 2.])
                - instances=[[1, 2, 3], [3, 2, 1]]
                - date=datetime.date(2024, 1, 1)
                - starting_point=[0, 1]
            )
        """

        # Check if the other is a Cluster instance
        if not isinstance(other, Cluster):
            raise TypeError(f"The other must be a Cluster object. Got {type(other).__name__} instead.")

        return self + other

    def is_similar(self, other: 'Cluster', distance_threshold: Union[float, int] = 0.001) -> bool:
        """
        Check if the cluster is similar to another cluster

        Parameters:
            * other: `Cluster`. The cluster to compare
            * distance_threshold: `Union[float, int]`. The distance threshold. Default is 0.001

        Returns:
            `bool`. `True` if the cluster is similar, `False` otherwise

        Examples:
            >>> sequence1 = Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence2 = Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 1))
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), sequence1)
            >>> cluster2 = Cluster(np.array([3, 4, 5, 6]), sequence2)
            >>> cluster1.is_similar(cluster2)
            True

            >>> sequence3 = Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence4 = Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 1))
            >>> cluster3 = Cluster(np.array([3, 4, 5, 6]), sequence3)
            >>> cluster4 = Cluster(np.array([7, 8, 9, 10]), sequence4)
            >>> cluster3.is_similar(cluster4)
            False
        """

        distance = np.max(np.abs(self.centroid - other.centroid))
        return distance < distance_threshold

    def is_overlapping(self, other: 'Cluster', epsilon: Union[int, float]) -> bool:

        other_sequences = other.get_sequences()
        self_sequences = self.get_sequences()

        N_matches = 0
        N_max = min(len(other_sequences), len(self_sequences))
        for subsequence in self_sequences:
            if subsequence in other_sequences:
                N_matches += 1

        return N_matches / N_max >= epsilon


class Routines:
    """
    Represents a collection of clusters, each of them representing a routine.

    Parameters:
    _________
        * ``cluster: Optional[Cluster]``, the cluster to add to the collection. Default is None

    Properties:
    ________
        **Getters**:
            * ``hierarchy: int``, the length of the subsequences in the clusters

    Public Methods:
    _________
        * ``add_routine(new_routine: Cluster)``: adds a cluster to the collection
        * ``drop_indexes(to_drop: list[int])``: drops the clusters at the specified indexes
        * ``get_routines() -> list[Cluster]``: returns the routines as a list of clusters
        * ``get_centroids() -> list[np.ndarray]``: returns the centroids of the clusters as a list of numpy arrays
        * ``to_collection() -> list[dict]``: returns the routines as a list of dictionaries
        * ``is_empty() -> bool``: checks if the collection is empty

    Examples:

        >>> subsequence1 = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
        >>> subsequence2 = Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)

        >>> sequence = Sequence(subsequence=subsequence1)
        >>> cluster1 = Cluster(np.array([1, 1, 1, 1]), sequence)

        >>> sequence = Sequence(subsequence=subsequence2)
        >>> cluster2 = Cluster(np.array([5, 5, 5, 5]), sequence)

        >>> routines = Routines(cluster=cluster1)
        >>> routines.add_routine(cluster2)

        >>> routines.get_routines()
        [Cluster(
            centroid=np.array([1, 1, 1, 1]),
            sequences=Sequence(
                length_subsequences=4,
                list_sequences=[
                    Subsequence(
                        instance=np.array([1, 2, 3, 4]),
                        date=datetime.date(2021, 1, 1),
                        starting_point=0
                    )
                ]
            )
        ), Cluster(
            centroid=np.array([5, 5, 5, 5]),
            sequences=Sequence(
                length_subsequences=4,
                list_sequences=[
                    Subsequence(
                        instance=np.array([5, 6, 7, 8]),
                        date=datetime.date(2021, 1, 2),
                        starting_point=4
                    )
                ]
            )
        )]

        >>> routines.get_centroids()
        [np.array([1, 1, 1, 1]), np.array([5, 5, 5, 5])]

        >>> routines.to_collection()
        [{'centroid': np.array([1, 1, 1, 1]), 'sequences': [{'instance': np.array([1, 2, 3, 4]), 'date': datetime.date(2021, 1, 1), 'starting_point': 0}], 'length_subsequences': 4},
         {'centroid': np.array([5, 5, 5, 5]), 'sequences': [{'instance': np.array([5, 6, 7, 8]), 'date': datetime.date(2021, 1, 2), 'starting_point': 4}], 'length_subsequences': 4}]

        >>> routines.is_empty()
        False

        >>> routines.drop_indexes([0])
        >>> routines.get_routines()
        [Cluster(
            centroid=np.array([5, 5, 5, 5]),
            sequences=Sequence(
                length_subsequences=4,
                list_sequences=[
                    Subsequence(
                        instance=np.array([5, 6, 7, 8]),
                        date=datetime.date(2021, 1, 2),
                        starting_point=4
                    )
                ]
            )
        )]
    """

    def __init__(self, cluster: Optional[Cluster] = None) -> None:
        """
        Parameters:
            * cluster: `Optional[Cluster]`, the cluster to add to the routines. Default is None

        Raises:
             TypeError: if the parameter is not an instance of cluster

        Examples:
             >>> routines = Routines()
             >>> print(routines)
             Routines(
                list_routines = [[]]
             )

             >>> sequence = Sequence(subsequence=Subsequence(np.array([1,2,3], datetime.date(2024, 1, 1), 1)))
             >>> routines = Routines(Cluster(centroid=np.array([1,2,3], instances=sequence)))
             >>> print(routines)
             Routines(
                list_routines = [
                    Cluster(
                        - centroid = [1,2,3],
                        - instances = [[1,2,3]]
                        - starting_points = [1]
                        - dates = [datetime.date(2024, 1, 1)]
                    )
                ]
             )
        """

        if cluster is not None:
            if not isinstance(cluster, Cluster):
                raise TypeError(f"cluster has to be an instance of Cluster. Got {type(cluster).__name__} instead")

            self.__routines: list[Cluster] = [cluster]
            self.__hierarchy = cluster.length_cluster_subsequences

        else:
            self.__routines: list[Cluster] = []
            self.__hierarchy = None

    def __repr__(self):
        """
        Returns the string representation of the routines

        Returns:
            `str`. The string representation of the routines

        Examples:
            >>> routines = Routines()
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> cluster2 = Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> routines.add_routine(cluster1)
            >>> routines.add_routine(cluster2)
            >>> print(routines)
            Routines(
                list_routines=[
                    Cluster(
                        - centroid = [3, 4, 5, 6],
                        - instances = [[1, 2, 3, 4]]
                        - starting_points = [0]
                        - dates = [datetime.date(2021, 1, 1)]
                    ),
                    Cluster(
                        - centroid = [7, 8, 9, 10],
                        - instances = [[5, 6, 7, 8]]
                        - starting_points = [4]
                        - dates = [datetime.date(2021, 1, 2)]
                    )
                ]
            )
        """

        out_string = "Routines(\n\tlist_routines=[\n"
        for routine in self.__routines:
            out_string += f" {routine},\n"

        out_string = out_string[:-2] + out_string[-1] + "])"
        return out_string

    def __str__(self):
        """
        Returns the string representation of the routines

        Returns:
            `str`. The string representation of the routines

        Examples:
            >>> routines = Routines()
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> cluster2 = Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> routines.add_routine(cluster1)
            >>> routines.add_routine(cluster2)
            >>> print(routines)
            Routines(
                list_routines=[
                    Cluster(
                        - centroid = [3, 4, 5, 6],
                        - instances = [[1, 2, 3, 4]]
                        - starting_points = [0]
                        - dates = [datetime.date(2021, 1, 1)]
                    ),
                    Cluster(
                        - centroid = [7, 8, 9, 10],
                        - instances = [[5, 6, 7, 8]]
                        - starting_points = [4]
                        - dates = [datetime.date(2021, 1, 2)]
                    )
                ]
            )
        """

        out_string = "Routines(\n\tlist_routines=[\n"
        for routine in self.__routines:
            out_string += f" {routine},\n"

        out_string = out_string[:-2] + out_string[-1] + "])"
        return out_string

    def __len__(self) -> int:
        """
        Returns the number of clusters in the `Routines`

        Returns:
            `int`. The number of clusters in the `Routines`

        Examples:
            >>> routines = Routines()
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> cluster2 = Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> routines.add_routine(cluster1)
            >>> routines.add_routine(cluster2)
            >>> len(routines)
            2
        """

        return len(self.__routines)

    def __getitem__(self, index: int) -> 'Cluster':
        """
        Get the cluster at the specified index in the collection

        Parameters:
            * index: `int`. The index of the cluster

        Returns:
            `Cluster`. The cluster at the specified index in the collection

        Raises:
            TypeError: if the index is not an integer
            IndexError: if the index is out of range of the routines

        Examples:
            >>> routines = Routines()
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> cluster2 = Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> routines.add_routine(cluster1)
            >>> routines.add_routine(cluster2)
            >>> print(routines[0])
            Cluster(
                - centroid = [3, 4, 5, 6],
                - instances = [[1, 2, 3, 4]]
                - starting_points = [0]
                - dates = [datetime.date(2021, 1, 1)]
            )
        """
        # Check if the index is an integer
        if not isinstance(index, int):
            raise TypeError(f"index must be an integer. Got {type(index).__name__} instead")

        # Check if the index is within the range of the list
        if not 0 <= index < len(self.__routines):
            raise IndexError(f"index {index} out of range (0, {len(self.__routines) - 1})")

        return self.__routines[index]

    def __setitem__(self, index: int, value: 'Cluster') -> None:
        """
        Set the value of the cluster at the specified index in the collection

        Parameters:
            * index: `int`. The index of the cluster
            * value: `Cluster`. The new cluster

        Raises:
            TypeError: if the index is not an integer or the value is not an instance of Cluster
            IndexError: if the index is out of range of the routines

        Examples:
            >>> routines = Routines()
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> cluster2 = Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> routines.add_routine(cluster1)
            >>> routines.add_routine(cluster2)
            >>> routines[0] = Cluster(np.array([11, 12, 13, 14]), Sequence(Subsequence(np.array([9, 10, 11, 12]), datetime.date(2021, 1, 3), 4))
            >>> print(routines[0])
            Cluster(
                - centroid = [11, 12, 13, 14],
                - instances = [[9, 10, 11, 12]]
                - starting_points = [4]
                - dates = [datetime.date(2021, 1, 3)]
            )
        """
        # Check if the value is a Cluster instance
        if not isinstance(value, Cluster):
            raise TypeError(f"value has to be an instance of Cluster. Got {type(value).__name__} instead")

        # Check if the index is an integer
        if not isinstance(index, int):
            raise TypeError(f"index has to be an integer. Got {type(index).__name__} instead")

        # Check if the index is within the range of the list
        if not 0 <= index < len(self.__routines):
            raise IndexError(f"index {index} out of range. Got (0, {len(self.__routines) - 1})")

        self.__routines[index] = value

    def __iter__(self) -> iter:
        """
        Returns an iterator for each cluster in the `Routines`

        Returns:
            iter. An iterator for each cluster in the `Routines`

        Examples:
            >>> routines = Routines()
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> cluster2 = Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> routines.add_routine(cluster1)
            >>> routines.add_routine(cluster2)
            >>> for cluster in routines:
            ...     print(cluster)
            Cluster(
                - centroid = [3, 4, 5, 6],
                - instances = [[1, 2, 3, 4]]
                - starting_points = [0]
                - dates = [datetime.date(2021, 1, 1)]
            )
            Cluster(
                - centroid = [7, 8, 9, 10],
                - instances = [[5, 6, 7, 8]]
                - starting_points = [4]
                - dates = [datetime.date(2021, 1, 2)]
            )
        """

        return iter(self.__routines)

    def __contains__(self, item: 'Cluster') -> bool:
        """
        Check if the cluster exists in the collection

        Parameters:
            * item: `Cluster`. The cluster to check

        Returns:
            `bool`. `True` if the cluster exists, `False` otherwise

        Raises:
            TypeError: if the parameter is not an instance of Cluster
        """

        # Check if the item is a Cluster instance
        if not isinstance(item, Cluster):
            raise TypeError(f"item has to be an instance of Cluster. Got {type(item).__name__} instead")

        return item in self.__routines

    def __delitem__(self, index: int) -> None:
        """
        Deletes the cluster at the specified index in the collection

        Parameters:
            * index: `int`. The index of the cluster to delete

        Returns:
            `Cluster`. The cluster at the specified index in the collection

        Raises:
            TypeError: if the index is not an integer
            IndexError: if the index is out of range of the routines
        """

        # Check if the index is an integer
        if not isinstance(index, int):
            raise TypeError(f"index has to be an integer. Got {type(index).__name__} instead")

        # Check if the index is within the range of the list
        if not 0 <= index < len(self.__routines):
            raise IndexError(f"index {index} out of range (0, {len(self.__routines) - 1})")

        del self.__routines[index]

    def __add__(self, other: 'Routines') -> 'Routines':
        """
        Concatenates two routines together with the operator + and returns a new collection

        Parameters:
            * other: `Routines`. The collection to concatenate

        Returns:
            `Routines`. The concatenated `Routines`

        Raises:
            TypeError: if the parameter is not an instance of Routines
            ValueError: if the hierarchy of the routines is not the same

        Examples:
            >>> routines1 = Routines()
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> cluster2 = Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> routines1.add_routine(cluster1)
            >>> routines1.add_routine(cluster2)
            >>> routines2 = Routines()
            >>> cluster3 = Cluster(np.array([11, 12, 13, 14]), Sequence(Subsequence(np.array([9, 10, 11, 12]), datetime.date(2021, 1, 3), 4))
            >>> routines2.add_routine(cluster3)
            >>> new_routines = routines1 + routines2
            >>> print(new_routines)
            Routines(
                list_routines=[
                    Cluster(
                        - centroid = [3, 4, 5, 6],
                        - instances = [[1, 2, 3, 4]]
                        - starting_points = [0]
                        - dates = [datetime.date(2021, 1, 1)]
                    ),
                    Cluster(
                        - centroid = [7, 8, 9, 10],
                        - instances = [[5, 6, 7, 8]]
                        - starting_points = [4]
                        - dates = [datetime.date(2021, 1, 2)]
                    ),
                    Cluster(
                        - centroid = [11, 12, 13, 14],
                        - instances = [[9, 10, 11, 12]]
                        - starting_points = [4]
                        - dates = [datetime.date(2021, 1, 3)]
                    )
                ]
            )
        """
        # Check if the other is a Routines instance
        if not isinstance(other, Routines):
            raise TypeError(f"other has to be an instance of Routines. Got {type(other).__name__} instead")

        # Check if the routines are empty
        if other.is_empty() and not self.is_empty():
            return self

        # Check if the routines are empty
        if not other.is_empty() and self.is_empty():
            return other

        # Check if the hierarchy is the same
        if not other.is_empty() and self.__hierarchy != other[0].length_cluster_subsequences:
            raise ValueError(
                f"the hierarchy of the routines must be the same. Expected {self.__hierarchy}, got {other.__hierarchy} instead")

        # Concatenate the routines if both are not empty
        new_routines = Routines()
        new_routines.__routines = self.__routines + other.__routines
        new_routines.__hierarchy = self.__hierarchy
        return new_routines

    def __eq__(self, other: 'Routines') -> bool:
        """
        Check if the self routine is equal to another routine with the operator ==

        Parameters:
            * other: `Routines`. The routine to check

        Returns:
            `bool`. `True` if the routines are equal, `False` otherwise

        Raises:
            TypeError: if the parameter is not an instance of `Routines`

        Examples:
            >>> routines1 = Routines()
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> cluster2 = Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> routines1.add_routine(cluster1)
            >>> routines1.add_routine(cluster2)
            >>> routines2 = Routines()
            >>> cluster3 = Cluster(np.array([11, 12, 13, 14]), Sequence(Subsequence(np.array([9, 10, 11, 12]), datetime.date(2021, 1, 3), 4))
            >>> routines2.add_routine(cluster3)
            >>> routines1 == routines2
            False

            >>> routines3 = Routines()
            >>> cluster4 = Cluster(np.array([3, 4, 5, 6]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> cluster5 = Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> routines3.add_routine(cluster4)
            >>> routines3.add_routine(cluster5)
            >>> routines1 == routines3
            True
        """

        # Check if the other is a Routines instance
        if not isinstance(other, Routines):
            raise TypeError(f"other has to be an instance of Routines. Got {type(other).__name__} instead")

        # Check if the number of clusters is equal
        if len(self.__routines) != len(other.__routines):
            return False

        if self.__hierarchy != other.__hierarchy:
            return False

        # Check if the clusters are equal
        for idx, routine in enumerate(self.__routines):
            if routine != other.__routines[idx]:
                return False

        return True

    @property
    def hierarchy(self) -> int:
        """
        Returns the hierarchy of the routines

        Returns:
            `int`. The hierarchy of the routines
        """

        return self.__hierarchy

    def add_routine(self, new_routine: 'Cluster') -> None:
        """
        Adds a cluster to the collection

        Parameters:
            new_routine: `Cluster`. The cluster to add

        Raises:
             TypeError: if the parameter is not of the correct type

        Examples:
            >>> routines = Routines()
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> cluster2 = Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> routines.add_routine(cluster1)
            >>> routines.add_routine(cluster2)
            >>> print(routines)
            Routines(
                list_routines=[
                    Cluster(
                        - centroid = [3, 4, 5, 6],
                        - instances = [[1, 2, 3, 4]]
                        - starting_points = [0]
                        - dates = [datetime.date(2021, 1, 1)]
                    ),
                    Cluster(
                        - centroid = [7, 8, 9, 10],
                        - instances = [[5, 6, 7, 8]]
                        - starting_points = [4]
                        - dates = [datetime.date(2021, 1, 2)]
                    )
                ]
            )
        """
        # Check if the new_routine is a Cluster instance
        if not isinstance(new_routine, Cluster):
            raise TypeError(f"new_routine has to be an instance of Cluster. Got {type(new_routine).__name__} instead")

        # Check if the hierarchy is not initialized
        if self.__hierarchy is None:
            self.__hierarchy = new_routine.length_cluster_subsequences

        # Check if the length of the subsequences is the same as the hierarchy
        if new_routine.length_cluster_subsequences != self.__hierarchy:
            raise ValueError(
                f"the length of the subsequences must be {self.__hierarchy}. Got {new_routine.length_cluster_subsequences} instead")

        self.__routines.append(new_routine)

    def drop_indexes(self, to_drop: list[int]) -> 'Routines':
        """
        Drops the clusters with the specified indexes

        Parameters:
            to_drop: `list[int]`. The indexes of the clusters to drop

        Returns:
             Routines. The collection without the dropped clusters

        Examples:
            >>> routines = Routines()
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)))
            >>> cluster2 = Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 2)))
            >>> cluster3 = Cluster(np.array([11, 12, 13, 14]), Sequence(Subsequence(np.array([9, 10, 11, 12]), datetime.date(2021, 1, 3), 4)))
            >>> routines.add_routine(cluster1)
            >>> routines.add_routine(cluster2)
            >>> routines.add_routine(cluster3)
            >>> filtered_routines = routines.drop_indexes([0, 2])
            >>> print(filtered_routines)
            Routines(
                list_routines=[
                    Cluster(
                        - centroid = [11, 12, 13, 14],
                        - instances = [[9, 10, 11, 12]]
                        - starting_points = [4]
                        - dates = [datetime.date(2021, 1, 3)]
                    )
                ]
            )

        Notes:
            This method does not modify the original `Routine`, it returns a new one without the dropped clusters
        """

        # Check if the indexes to drop are a list of integers
        if not isinstance(to_drop, list):
            raise TypeError(
                f"to_drop has to be a list of integers. Got {type(to_drop).__name__} instead")

        # Check if the indexes are integers
        if not all(isinstance(idx, int) for idx in to_drop):
            raise TypeError(
                f"the indexes to drop have to be integers. Got list {[type(v).__name__ for v in to_drop]}, instead")

        # Check if the indexes are integers
        new_routines = Routines()

        # Checks if the index of the cluster is in the list of indexes to drop
        for idx, cluster in enumerate(self.__routines):
            if idx not in to_drop:
                new_routines.add_routine(cluster)

        return new_routines

    def get_centroids(self) -> list[np.ndarray]:
        """
        Returns the centroids of the clusters

        Returns:
             `list[np.ndarray]`. The centroids of the clusters

        Examples:
            >>> routines = Routines()
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> cluster2 = Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> routines.add_routine(cluster1)
            >>> routines.add_routine(cluster2)
            >>> routines.get_centroids()
            [np.array([3, 4, 5, 6]), np.array([7, 8, 9, 10])]
        """

        return [cluster.centroid for cluster in self.__routines]

    def to_collection(self) -> list[dict]:
        """
        Returns the collection as a list of dictionaries

        Returns:
             `list[dict]`. The routines as a list of dictionaries

        Examples:
            >>> routines = Routines()
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> cluster2 = Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> routines.add_routine(cluster1)
            >>> routines.add_routine(cluster2)
            >>> routines.to_collection()
            [{ 'centroid': np.array([3, 4, 5, 6]),
              'instances': [{'instance': np.array([1, 2, 3, 4]), 'date': datetime.date(2021, 1, 1), 'starting_point': 0}]},
              {'centroid': np.array([7, 8, 9, 10]),
               'instances': [{'instance': np.array([5, 6, 7, 8]), 'date': datetime.date(2021, 1, 2), 'starting_point': 4}]}
            ]
        """

        collection = []
        for routine in self.__routines:
            collection.append({
                'centroid': routine.centroid.tolist(),
                'instances': routine.get_sequences().to_collection()
            })
        return collection

    def is_empty(self) -> bool:
        """
        Returns `True` if routines is empty, `False` otherwise

        Returns:
            `bool`. `True` if the collection is empty, `False` otherwise

        Examples:
            >>> routines = Routines()
            >>> routines.is_empty()
            True

            >>> cluster = Cluster(np.array([3, 4, 5, 6]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> routines.add_routine(cluster)
            >>> routines.is_empty()
            False
        """

        return len(self.__routines) == 0

    def drop_duplicates(self) -> 'Routines':
        """
        Filter the repeated clusters in the routines and returns

        Returns:
            `Routines`. The routines without the repeated clusters
        """

        new_routines = Routines()

        for cluster in self.__routines:
            if cluster not in new_routines:
                new_routines.add_routine(cluster)

        return new_routines

    # def remove_repeated_instances(self) -> 'Routines':
    #     """
    #     If some instance is repeated in two clusters,
    #     the method removes the repeated instance from the cluster with less instances.
    #     If both clusters have the same number of instances,
    #     the method removes the instance from the cluster with less cumulative magnitude
    #
    #     Returns:
    #         `Routines`. The routines without the repeated instances
    #     """
    #
    #     new_routines = Routines()
    #     existent_sequences: list[list[Sequence]] = []
    #     for i in range(len(self.__routines)):
    #         for j in range(i + 1, len(self.__routines)):
    #             sequence_i = self.__routines[i].get_sequences()
    #             sequence_j = self.__routines[j].get_sequences()
    #             for id, subsequence_i in enumerate(sequence_i):
    #                 if subsequence_i in sequence_j:
    #                     if len(sequence_i) < len(sequence_j):
    #                         sequence_j.remove(subsequence_i)
    #                     elif len(sequence_i) > len(sequence_j):
    #                         sequence_i.remove(subsequence_i)
    #                     else:
    #                         if sum([np.sum(subsequence.instance) for subsequence in sequence_i]) < sum([np.sum(subsequence.instance) for subsequence in sequence_j]):
    #                             sequence_j.remove(subsequence_i)
    #                         else:
    #                             sequence_i.remove(subsequence_i)
    #
    #
    #     return new_routines

    def remove_subsets(self) -> 'Routines':
        """
        Filter the clusters that are subsets of another cluster in the routines and returns the filtered routines

        Returns:
            `Routines`. The routines without the clusters that are subsets of another cluster

        Examples:
            >>> routines = Routines()
            >>> cluster1 = Cluster(np.array([1, 1, 1, 1]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> cluster2 = Cluster(np.array([1, 1, 1, 1]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> cluster3 = Cluster(np.array([1, 1, 1, 1]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> routines.add_routine(cluster1)
            >>> routines.add_routine(cluster2)
            >>> routines.add_routine(cluster3)
            >>> new_routines = routines.remove_subsets()
            >>> print(new_routines)
            Routines(
                list_routines=[
                    Cluster(
                        - centroid = [1, 1, 1, 1],
                        - instances = [[1, 2, 3, 4]]
                        - starting_points = [0]
                        - dates = [datetime.date(2021, 1, 1)]
                    )
                ]
            )
        """
        # Get the starting points of each cluster
        list_starting_points = [cluster.get_starting_points() for cluster in self.__routines]
        new_routine = Routines()
        to_drop_list = []

        # Check if the starting points of the first routine are a subset of the starting points of the second routine
        for i in range(len(self.__routines)):
            starting_points = list_starting_points[i]

            for j in range(i + 1, len(self.__routines)):
                # Check if the starting points of the first routine are a subset of the starting points of the second routine
                if len(starting_points) < len(list_starting_points[j]):
                    # If each starting point is on the starting points from the second routine, then is a subset
                    if all([x in list_starting_points[j] for x in starting_points]):
                        to_drop_list.append(i)
                        break
                else:
                    # Check if the starting points of the second routine are a subset of the starting points of the first routine
                    if all([x in starting_points for x in list_starting_points[j]]):
                        to_drop_list.append(j)
                        break

        # Add the clusters that are not subsets
        for i in range(len(self.__routines)):
            if i not in to_drop_list:
                new_routine.add_routine(self.__routines[i])

        return new_routine


class HierarchyRoutine:
    """
    Represents a hierarchy of routines,
    where the hierarchy corresponds to the length of the subsequences for each cluster in the routine.
    For each hierarchy, exists one routine with the correspondent length of the subsequences.

    Parameters:
    _________
        * ``routine: Optional[Routines]``, the routine to add to the hierarchy. Default is None

    Public Methods:
    _______________

        * ``add_routine(new_routine: Routines)``: adds a routine to the hierarchy
        * ``to_dictionary() -> dict``: returns the hierarchy as a dictionary

    Properties:
    ___________
        **Getters:**
            * ``keys: list[int]``: returns the list with all the hierarchies registered
            * ``values: list[Routines]``: returns a list with all the routines
            * ``items: Iterator[tuple[int, Routines]]``: returns a iterator as a zip object with the hierarchy and the routine

    Examples:

            >>> subsequence1 = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> subsequence2 = Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)

            >>> sequence = Sequence(subsequence=subsequence1)
            >>> cluster1 = Cluster(np.array([1, 1, 1, 1]), sequence)

            >>> sequence = Sequence(subsequence=subsequence2)
            >>> cluster2 = Cluster(np.array([5, 5, 5, 5]), sequence)

            >>> routines = Routines(cluster=cluster1)
            >>> routines.add_routine(cluster2)

            >>> hierarchy = HierarchyRoutine(routines=routines)
            >>> hierarchy.to_dictionary()
            {4: [{'centroid': array([1, 1, 1, 1]), 'instances': [{'instance': array([1, 2, 3, 4]), 'date': datetime.date(2021, 1, 1), 'starting_point': 0}]},
                 {'centroid': array([5, 5, 5, 5]), 'instances': [{'instance': array([5, 6, 7, 8]), 'date': datetime.date(2021, 1, 2), 'starting_point': 4}]}]}

            >>> hierarchy.keys
            [4]

            >>> hierarchy.values
            [Routines(
                list_routines=[
                  Cluster(
                     -Centroid: [1 1 1 1]
                     -Instances: [array([1, 2, 3, 4])]
                     -Dates: [datetime.date(2021, 1, 1)]
                     -Starting Points: [0]
                 ),
                  Cluster(
                     -Centroid: [5 5 5 5]
                     -Instances: [array([5, 6, 7, 8])]
                     -Dates: [datetime.date(2021, 1, 2)]
                     -Starting Points: [4]
                 )
            ])]

            >>> for key, value in hierarchy.items:
            ...     print(key, value)
            4 Routines(
                list_routines=[
                     Cluster(
                         -Centroid: [1 1 1 1]
                         -Instances: [array([1, 2, 3, 4])]
                         -Dates: [datetime.date(2021, 1, 1)]
                         -Starting Points: [0]
                    ),
                     Cluster(
                         -Centroid: [5 5 5 5]
                         -Instances: [array([5, 6, 7, 8])]
                         -Dates: [datetime.date(2021, 1, 2)]
                         -Starting Points: [4]
                    )
            ])
    """

    def __init__(self, routines: Optional[Routines] = None) -> None:
        """
        Initializes the HierarchyRoutine with the routines

        Parameters:
            * routines: ``Optional[Routines]``, the routines to add to the collection. Default is `None`

        Raises:
            TypeError: if the routines is not an instance of `Routines`
            ValueError: if the routines are empty

        Examples:

            >>> sequence = Sequence(Subsequence(np.array([1, 2, 3]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7]), datetime.date(2021, 1, 2), 4))
            >>> cluster = Cluster(np.array([3, 4, 5]), sequence)
            >>> routines = Routines(cluster=cluster)

            >>> hierarchy_routine = HierarchyRoutine(routines)

            >>> hierarchy_routine = HierarchyRoutine()
            >>> hierarchy_routine.add_routine(routines)

            >>> print(hierarchy_routine)
            HierarchyRoutine(
                [Hierarchy: 3.
                    Routines(
                        list_routines=[
                            Cluster(
                                - centroid = [3, 4, 5],
                                - instances = [[1, 2, 3], [5, 6, 7]]
                                - starting_points = [0, 4]
                                - dates = [datetime.date(2021, 1, 1)]
                            )
                        ]
                    )
                ]
            )
        """

        self.__list_routines: list[Routines] = []
        self.__hierarchy: list[int] = []

        # check if a routine is provided
        if routines is not None:
            # Check if the routines is an instance of Routines
            if not isinstance(routines, Routines):
                raise TypeError(f"routines has to be an instance of Routines. Got {type(routines).__name__} instead")

            # Check if the routines are not empty
            if len(routines) == 0:
                raise ValueError("routines cannot be empty")

            # Add the routine to the hierarchy routine
            self.__hierarchy = [routines.hierarchy]
            self.__list_routines.append(routines)

    def __str__(self):
        out_string = "HierarchyRoutine(\n"
        for idx, routine in enumerate(self.__list_routines):
            out_string += f" [Hierarchy: {self.__hierarchy[idx]}. \n\t{routine} ], \n"

        out_string = out_string[:-2] + out_string[-1] + ")"
        return out_string

    def __repr__(self):
        out_string = "HierarchyRoutine(\n"
        for idx, routine in enumerate(self.__list_routines):
            out_string += f" [Hierarchy: {self.__hierarchy[idx]}. \n\t{routine} ], \n"

        out_string = out_string[:-2] + out_string[-1] + ")"
        return out_string

    def __setitem__(self, hierarchy: int, routine: Routines) -> None:
        """
        Sets the routine at the specified hierarchy

        Parameters:
            * hierarchy: `int`. The hierarchy of the routine
            * routine: `Routines`. The routine to set

        Raises:
            TypeError: if the hierarchy is not an integer or the routine is not an instance of Routines
            ValueError: if the routine is empty or the hierarchy is not the same as the routine hierarchy

        Examples:
            >>> routine = Routines(cluster=Cluster(np.array([3, 4, 5]), Sequence(Subsequence(np.array([1, 2, 3]), datetime.date(2021, 1, 1), 0)))
            >>> routine2 = Routines(cluster=Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)))

            >>> hierarchy_routine = HierarchyRoutine(routine)
            >>> hierarchy_routine[4] = routine2
        """
        # Check if the hierarchy is an integer
        if not isinstance(hierarchy, int):
            raise TypeError(f"hierarchy has to be an integer. Got {type(hierarchy).__name__} instead")

        # Check if the routine is an instance of Routines
        if not isinstance(routine, Routines):
            raise TypeError(f"routine has to be an instance of Routines. Got {type(routine).__name__} instead")

        # Check if the routine is empty
        if routine.is_empty():
            raise ValueError("routine cannot be empty")

        # Check if the hierarchy is the same as the routine hierarchy
        if hierarchy != routine.hierarchy:
            raise ValueError(
                f"the hierarchy of the routines must be the same. Expected {hierarchy}. Got {routine.hierarchy} instead")

        # If the hierarchy exists, we update the value
        if hierarchy in self.__hierarchy:
            idx = self.__hierarchy.index(hierarchy)
            self.__list_routines[idx] = routine

        # If the hierarchy doesn't exist, we create a new tuple key, value
        else:
            self.__hierarchy.append(hierarchy)
            self.__list_routines.append(routine)

    def __getitem__(self, hierarchy: int) -> Routines:
        """
        Get the routine at the specified hierarchy

        Parameters:
            * hierarchy: `int`. The hierarchy of the routine

        Returns:
            `Routines`. The routine at the specified hierarchy

        Raises:
            TypeError: if the hierarchy is not an integer
            KeyError: if the hierarchy is not found in the routines

        Examples:
            >>> routine = Routines(cluster=Cluster(np.array([3, 4, 5]), Sequence(Subsequence(np.array([1, 2, 3]), datetime.date(2021, 1, 1), 0)))
            >>> routine2 = Routines(cluster=Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)))

            >>> hierarchy_routine = HierarchyRoutine(routine)
            >>> hierarchy_routine.add_routine(routine2)
            >>> print(hierarchy_routine[3])
            Cluster(
                - centroid = [3, 4, 5],
                - instances = [[1, 2, 3]]
                - starting_points = [0]
                - dates = [datetime.date(2021, 1, 1)]
            )
        """

        # Check if the hierarchy is an integer
        if not isinstance(hierarchy, int):
            raise TypeError(f"hierarchy has to be an integer. Got {type(hierarchy).__name__} instead")

        # Check if the hierarchy exists
        if hierarchy not in self.__hierarchy:
            raise KeyError(f"hierarchy {hierarchy} not found in {self.__hierarchy}")

        # Get the index of the hierarchy
        idx = self.__hierarchy.index(hierarchy)
        return self.__list_routines[idx]

    def __len__(self) -> int:
        """
        Returns the number of routines in the hierarchical routines

        Returns:
            `int`. The number of routines in the hierarchical routines

        Examples:
            >>> routine = Routines(cluster=Cluster(np.array([3, 4, 5]), Sequence(Subsequence(np.array([1, 2, 3]), datetime.date(2021, 1, 1), 0)))
            >>> routine2 = Routines(cluster=Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)))

            >>> hierarchy_routine = HierarchyRoutine(routine)
            >>> hierarchy_routine.add_routine(routine2)
            >>> len(hierarchy_routine)
            2
        """

        return len(self.__list_routines)

    def __contains__(self, routine: Routines) -> bool:
        """
        Check if the routine exists in the hierarchical routines

        Parameters:
            * routine: `Routines`. The routine to check

        Returns:
            `bool`. `True` if the routine exists, `False` otherwise

        Examples:
            >>> routine = Routines(cluster=Cluster(np.array([3, 4, 5]), Sequence(Subsequence(np.array([1, 2, 3]), datetime.date(2021, 1, 1), 0)))
            >>> routine2 = Routines(cluster=Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)))
            >>> hierarchy_routine = HierarchyRoutine(routine)
            >>> hierarchy_routine.add_routine(routine2)
            >>> routine in hierarchy_routine
            True

            >>> routine3 = Routines(cluster=Cluster(np.array([11, 12, 13]), Sequence(Subsequence(np.array([9, 10, 11]), datetime.date(2021, 1, 3), 0)))
            >>> routine3 in hierarchy_routine
            False
        """

        # Check if the routine is an instance of Routines
        if not isinstance(routine, Routines):
            raise TypeError(f"routine has to be an instance of Routines. Got {type(routine).__name__} instead")

        return routine in self.__list_routines

    def __eq__(self, other: 'HierarchyRoutine') -> bool:
        # Check if the other is a HierarchyRoutine instance
        if not isinstance(other, HierarchyRoutine):
            raise TypeError(f"other has to be an instance of HierarchyRoutine. Got {type(other).__name__} instead")

        # Check if the number of routines and hierarchies are equal
        if len(self.__list_routines) != len(other.__list_routines) or len(self.__hierarchy) != len(other.__hierarchy):
            return False

        # Check if the hierarchy is the same
        if any([hierarchy not in other.keys for hierarchy in self.__hierarchy]):
            return False

        # Check if the routines are equal
        for idx, routine in enumerate(self.__list_routines):
            if routine != other.__list_routines[idx]:
                return False

        return True

    @property
    def keys(self) -> list[int]:
        """
        Returns the hierarchy of the routines

        Returns:
            `list[int]`. The hierarchy of the routines

        Examples:
            >>> routine = Routines(cluster=Cluster(np.array([3, 4, 5]), Sequence(Subsequence(np.array([1, 2, 3]), datetime.date(2021, 1, 1), 0)))
            >>> routine2 = Routines(cluster=Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)))

            >>> hierarchy_routine = HierarchyRoutine(routine)
            >>> hierarchy_routine[4] = routine2
            >>> hierarchy_routine.keys
            [3, 4]
        """

        return self.__hierarchy

    @property
    def values(self) -> list[Routines]:
        """
        Returns the routines

        Returns:
            `list[Routines]`. The routines

        Examples:
            >>> routine = Routines(cluster=Cluster(np.array([3, 4, 5]), Sequence(Subsequence(np.array([1, 2, 3]), datetime.date(2021, 1, 1), 0)))
            >>> routine2 = Routines(cluster=Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)))

            >>> hierarchy_routine = HierarchyRoutine(routine)
            >>> hierarchy_routine[4] = routine2
            >>> hierarchy_routine.values
            [Routines(
                list_routines=[
                    Cluster(
                        - centroid = [3, 4, 5],
                        - instances = [[1, 2, 3]]
                        - starting_points = [0]
                        - dates = [datetime.date(2021, 1, 1)]
                    )
                ]
            ), Routines(
                list_routines=[
                    Cluster(
                        - centroid = [7, 8, 9, 10],
                        - instances = [[5, 6, 7, 8]]
                        - starting_points = [4]
                        - dates = [datetime.date(2021, 1, 2)]
                    )
                ]
            )]
        """

        return self.__list_routines

    @property
    def items(self) -> Iterator[tuple[int, Routines]]:
        """
        Getter that returns the hierarchy and the routines as a zip object iterator

        Returns:
            `Iterator[tuple[int, Routines]]`. The hierarchy and the routines as a zip object iterator

        Examples:
            >>> routine = Routines(cluster=Cluster(np.array([3, 4, 5]), Sequence(Subsequence(np.array([1, 2, 3]), datetime.date(2021, 1, 1), 0)))
            >>> routine2 = Routines(cluster=Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)))

            >>> hierarchy_routine = HierarchyRoutine(routine)
            >>> hierarchy_routine[4] = routine2
            >>> for hierarchy, routine in hierarchy_routine.items:
            ...     print(hierarchy, routine)
            3 Routines(
                list_routines=[
                    Cluster(
                        - centroid = [3, 4, 5],
                        - instances = [[1, 2, 3]]
                        - starting_points = [0]
                        - dates = [datetime.date(2021, 1, 1)]
                    )
                ]
            )
            4 Routines(
                list_routines=[
                    Cluster(
                        - centroid = [7, 8, 9, 10],
                        - instances = [[5, 6, 7, 8]]
                        - starting_points = [4]
                        - dates = [datetime.date(2021, 1, 2)]
                    )
                ]
            )

        """
        return zip(self.__hierarchy, self.__list_routines)

    def is_empty(self) -> bool:
        return len(self.__list_routines) == 0

    def add_routine(self, routine: Routines) -> None:
        """
        Adds a routine to the HierarchyRoutine.
        If the key (hierarchy) already exists, it updates the value (Routine).
        If not, it creates a new key, value

        Parameters:
            * routine: `Routines`. The routine to add

        Raises:
            TypeError: if the routine is not an instance of Routines
            ValueError: if the routine is empty

        Examples:
            >>> routine = Routines(cluster=Cluster(np.array([3, 4, 5]), Sequence(Subsequence(np.array([1, 2, 3]), datetime.date(2021, 1, 1), 0)))
            >>> routine2 = Routines(cluster=Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)))

            >>> hierarchy_routine = HierarchyRoutine(routine)
            >>> hierarchy_routine.add_routine(routine2)
            HierachyRoutine(
                [Hierarchy: 3.
                    Routines(
                        list_routines=[
                            Cluster(
                                - centroid = [3, 4, 5],
                                - instances = [[1, 2, 3]]
                                - starting_points = [0]
                                - dates = [datetime.date(2021, 1, 1)]
                            )
                        ]
                    )
                ],
                [Hierarchy: 4.
                    Routines(
                        list_routines=[
                            Cluster(
                                - centroid = [7, 8, 9, 10],
                                - instances = [[5, 6, 7, 8]]
                                - starting_points = [4]
                                - dates = [datetime.date(2021, 1, 2)]
                            )
                        ]
                    )
                ]
            )
        """

        # Check if the routine is an instance of Routines
        if not isinstance(routine, Routines):
            raise TypeError(f"routine has to be an instance of Routines. Got {type(routine).__name__}")

        # Check if the routine is empty
        if routine.is_empty():
            raise ValueError("routine cannot be empty")

        # Get the hierarchy of the routine
        length_clusters = routine.hierarchy

        # If doesn't exist, we create a new tuple key, value
        if length_clusters not in self.__hierarchy:
            self.__list_routines.append(routine)
            self.__hierarchy.append(length_clusters)

        # If it exists, we update the value
        else:
            idx = self.__hierarchy.index(length_clusters)
            self.__list_routines[idx] = routine

    def to_dictionary(self) -> dict:
        """
        Returns the routines as a dictionary

        Returns:
            `dict`. The routines as a dictionary

        Examples:
            >>> routine = Routines(cluster=Cluster(np.array([3, 4, 5]), Sequence(Subsequence(np.array([1, 2, 3]), datetime.date(2021, 1, 1), 0)))
            >>> routine2 = Routines(cluster=Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)))

            >>> hierarchy_routine = HierarchyRoutine(routine)
            >>> hierarchy_routine[4] = routine2
            >>> hierarchy_routine.to_dictionary()
            {3: Routines(
                list_routines=[
                    Cluster(
                        - centroid = [3, 4, 5],
                        - instances = [[1, 2, 3]]
                        - starting_points = [0]
                        - dates = [datetime.date(2021, 1, 1)]
                    )
                ]
            ), 4: Routines(
                list_routines=[
                    Cluster(
                        - centroid = [7, 8, 9, 10],
                        - instances = [[5, 6, 7, 8]]
                        - starting_points = [4]
                        - dates = [datetime.date(2021, 1, 2)]
                    )
                ]
            )}
        """

        out_dict = {}
        for idx, hierarchy in enumerate(self.__hierarchy):
            out_dict[hierarchy] = self.__list_routines[idx].to_collection()
        return out_dict

    def to_json(self, path_out: str):

        if not isinstance(path_out, str):
            raise TypeError(f"The path_out must be a string, got {type(path_out).__name__} instead")

        if not path_out.endswith('.json'):
            raise ValueError(f"The output file must be a json file format, got {path_out} instead")

        dictionary = self.to_dictionary()

        with open(path_out, 'w') as file:
            json.dump(dictionary, file, indent=3)

    def from_json(self, path_in: str) -> None:
        """
        Parameters:
            * path_in: `str`. The path to the json file

        Raises:
            TypeError: if the path_in is not a string
            ValueError: if the input file is not a json file format
            FileNotFoundError: if the file does not exist in the directory
            JSONDecodeError: if the file is not a valid json file
        """

        if not isinstance(path_in, str):
            raise TypeError(f"The path_in must be a string, got {type(path_in).__name__} instead")

        if not path_in.endswith('.json'):
            raise ValueError(f"The input file must be a json file format, got {path_in} instead")

        if not os.path.exists(path_in):
            raise FileNotFoundError(
                f"The file {path_in} does not exist in the directory {os.getcwd()} with the files: \n {os.listdir()}")

        self.__hierarchy: list[int] = []
        self.__list_routines: list[Routines] = []

        with open(path_in, 'r') as file:
            try:
                dictionary = json.load(file)

            except json.JSONDecodeError:
                raise ValueError(f"The file {path_in} is not a valid json file")

            for hierarchy, routines in dictionary.items():
                self.__hierarchy.append(int(hierarchy))
                new_routine = Routines()
                for cluster in routines:
                    centroid = np.array([float(x) for x in cluster['centroid']])
                    all_instances = cluster['instances']
                    sequence = Sequence()
                    for instance in all_instances:
                        subseq = np.array([float(x) for x in instance['instance']])
                        starting_point = int(instance['starting_point'])
                        date = datetime.datetime.strptime(instance['date'], "%Y/%m/%d, %H:%M:%S")

                        subsequence = Subsequence(subseq, date, starting_point)
                        sequence.add_sequence(subsequence)

                    new_routine.add_routine(Cluster(centroid, sequence))

                self.__list_routines.append(new_routine)

            print(f"File {path_in} imported successfully")

    def convert_to_cluster_tree(self) -> 'ClusterTree':

        if self.is_empty():
            warn("The hierarchy routine is empty. Returning an empty ClusterTree object")
            return ClusterTree()

        # Create a ClusterTree object
        cluster_tree = ClusterTree()

        # Assign the nodes to the cluster tree and iterate over all hierarchical routines
        for length, routine in self.items:
            for id_clust, cluster in enumerate(routine):
                # Assign the node to the cluster tree
                cluster_tree.assign_node(cluster)

                # If the length is greater than the minimum length, add the edges to the cluster tree
                if length > self.keys[0]:
                    # Iterate over the parent clusters
                    for id_parent, parent_cluster in enumerate(self[length - 1]):

                        # Check if the current cluster is the left child from the parent cluster and add the edge
                        try:
                            if cluster_tree.is_left_child(parent_cluster, cluster):
                                cluster_tree.add_edge(parent_cluster, cluster, is_left=True)

                            # Check if the current cluster is the right child from the parent cluster and add the edge
                            if cluster_tree.is_right_child(parent_cluster, cluster):
                                cluster_tree.add_edge(parent_cluster, cluster, is_left=False)

                        except Exception as e:
                            print(
                                f"error on parent: {length - 1}-{id_parent + 1}; and child: {length}-{id_clust + 1}, {e}")

        cluster_tree.assign_names()
        return cluster_tree


class ClusterTree:
    """
    Represents a tree of clusters, where each cluster is a node in the tree.
    The tree is a directed graph where the edges represent the parent-child relationship between the clusters.
    This class is used to represent the hierarchical structure of the clusters in the routines and how they are related.

    Public Methods:
    _______________

        * ``is_left_child(parent: Cluster, child: Cluster) -> bool``: checks if the child cluster corresponds to the left child of the parent cluster
        * ``is_right_child(parent: Cluster, child: Cluster) -> bool``: checks if the child cluster corresponds to the right child of the parent cluster
        * ``get_name_node(node: Union[Cluster, int]) -> str``: returns the name of the node from the cluster or the index
        * ``get_node(node: Union[str, int]) -> Cluster``: returns the cluster of the node from the index or the name
        * ``get_index(node: Union[Cluster, str]) -> int``: returns the index of the node from the cluster or the name
        * ``get_nodes_with_hierarchy(hierarchy: int) -> list[Cluster]``: returns the nodes with the specified hierarchy
        * ``to_dictionary() -> dict``: returns a dictionary where the keys are the indexes and the values are the clusters
        * ``children(node: Union[Cluster, int]) -> list[int]``: returns the children of the node
        * ``is_child(parent: Union[Cluster, int], child: Union[Cluster, int]) -> bool``: returns `True` if the child is a child of the parent, `False` otherwise
        * ``is_existent_left_child(parent: Union[Cluster, int], child: Union[Cluster, int]) -> bool``: returns `True` if the child is the left child of the parent in the graph, `False` otherwise
        * ``parents(node: Union[Cluster, int]) -> list[int]``: returns the parents of the node
        * ``has_children(node: Union[Cluster, int]) -> bool``: returns `True` if the node has children, `False` otherwise
        * ``has_parents(node: Union[Cluster, int]) -> bool``: returns `True` if the node has parents, `False` otherwise
        * ``assign_names()``: assigns names to the existent nodes in the graph
        * ``reset_names()``: resets the names of the nodes in the graph
        * ``assign_node(cluster: Cluster)``: assigns a node to the tree
        * ``add_edge(parent: Union[Cluster, int], child: Union[Cluster, int], is_left: bool)``: adds an edge to the tree
        * ``drop_node(node: Union[Cluster, int, str])``: drops all the nodes that depends directly on the node specified (including the node)
        * ``convert_to_hierarchy_routine() -> HierarchyRoutine``: converts the tree to a HierarchyRoutine
        * ``plot_tree(node_size: int = 1000, with_labels: bool = True, figsize: tuple[int, int] = (7, 7), title: Optional[str] = None, title_fontsize: int = 15, save_dir: Optional[str] = None)``: plots the tree

    Properties
    ___________
        **Getters:**
            * ``indexes: list[int]``: returns the list of indexes
            * ``nodes: list[Cluster]``: returns the list of clusters
            * ``graph: nx.classes.digraph.DiGraph``: returns the graph
            * ``edges: nx.classes.reportviews.OutEdgeDataView``: returns the edges of the graph

    Examples:
    _________
        >>> tree = ClusterTree()
        >>> parent = Cluster(np.array([1, 2, 3]), Sequence(Subsequence(np.array([1, 2, 3]), datetime.date(2021, 1, 2), 1)))
        >>> left = Cluster(np.array([0, 1, 2, 3]), Sequence(Subsequence(np.array([0, 1, 2, 3]), datetime.date(2021, 1, 1), 0)))
        >>> right = Cluster(np.array([1, 2, 3, 4]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 2), 1)))

        >>> tree.assign_node(parent)
        >>> tree.assign_node(left)
        >>> tree.assign_node(right)

        >>> tree.add_edge(1, 2, is_left=True)
        >>> tree.add_edge(1, 3, is_left=False)

        >>> tree.assign_names()

        >>> tree.indexes
        [1, 2, 3]

        >>> tree.nodes
        [Cluster(
            - centroid = [1, 2, 3],
            - instances = [[1, 2, 3]]
            - starting_points = [1]
            - dates = [datetime.date(2021, 1, 2)]
        ), Cluster(
            - centroid = [0, 1, 2, 3],
            - instances = [[0, 1, 2, 3]]
            - starting_points = [0]
            - dates = [datetime.date(2021, 1, 1)]
        ), Cluster(
            - centroid = [1, 2, 3, 4],
            - instances = [[1, 2, 3, 4]]
            - starting_points = [1]
            - dates = [datetime.date(2021, 1, 2)]
        )]

        >>> tree.name_nodes
        ['3-1', '4-1', '4-2']

        >>> tree.edges
        OutEdgeDataView([(1, 2, {'left': True}), (1, 3, {'left': False})])

        >>> tree.parents(2)
        [1]

        >>> tree.children(1)
        [2, 3]

        >>> tree.is_child(1, 2)
        True

        >>> tree.is_left_child(1, 2)
        True

        >>> tree.is_right_child(1, 3)
        True

        >>> tree.has_children(1)
        True

        >>> tree.has_parents(2)
        True

        >>> tree.hierarchies
        [3, 4]

        >>> tree.get_nodes_with_hierarchy(4)
        [2, 3]

        >>> tree.get_name_node(parent)
        '3-1'

        >>> tree.get_node("4-1")
        Cluster(
            - centroid = [0, 1, 2, 3],
            - instances = [[0, 1, 2, 3]]
            - starting_points = [0]
            - dates = [datetime.date(2021, 1, 1)]
        )

        >>> routine = tree.convert_to_hierarchy_routine()
        >>> print(routine)
        HierarchyRoutine(
            [Hierarchy: 3.
                Routines(
                    list_routines=[
                        Cluster(
                            - centroid = [1, 2, 3],
                            - instances = [[1, 2, 3]]
                            - starting_points = [1]
                            - dates = [datetime.date(2021, 1, 2)]
                        )
                    ]
                )
            ],
            [Hierarchy: 4.
                Routines(
                    list_routines=[
                        Cluster(
                            - centroid = [0, 1, 2, 3],
                            - instances = [[0, 1, 2, 3]]
                            - starting_points = [0]
                            - dates = [datetime.date(2021, 1, 1)]
                        ),
                        Cluster(
                            - centroid = [1, 2, 3, 4],
                            - instances = [[1, 2, 3, 4]]
                            - starting_points = [1]
                            - dates = [datetime.date(2021, 1, 2)]
                        )
                    ]
                )
            ]

        >>> tree.drop_node("4-3")
        >>> tree.nodes
        [1, 2]
    """

    def __init__(self):
        self.__graph: nx.DiGraph = nx.DiGraph()
        self.__nodes: list[Cluster] = []
        self.__list_of_index: list[int] = []
        self.__name_node: list[str] = []

    @property
    def indexes(self) -> list[int]:
        """
        Returns the list of indexes from the graph

        Returns:
            `list[int]`. The list of indexes

        Examples:
            >>> tree = ClusterTree()
            >>> tree.assign_node(Cluster(np.array([1, 1, 1, 1]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)))
            >>> tree.assign_node(Cluster(np.array([5, 5, 5, 5]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)))
            >>> tree.indexes
            [1, 2]
        """

        return self.__list_of_index

    @property
    def name_nodes(self) -> list[str]:
        """
        Returns the list of names from the graph

        Returns:
            `list[str]`. The list of names

        """

        return self.__name_node

    @property
    def nodes(self) -> list[Cluster]:
        """
        Returns the list of clusters from the graph

        Returns:
            `list[Cluster]`. The list of clusters

        Examples:
            >>> tree = ClusterTree()
            >>> cluster1 = Cluster(np.array([1, 1, 1, 1]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> cluster2 = Cluster(np.array([5, 5, 5, 5]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))

            >>> tree.assign_node(cluster1)
            >>> tree.assign_node(cluster2)

            >>> tree.nodes
            [Cluster(
                - centroid = [1, 1, 1, 1],
                - instances = [[1, 2, 3, 4]]
                - starting_points = [0]
                - dates = [datetime.date(2021, 1, 1)]
            ), Cluster(
                - centroid = [5, 5, 5, 5],
                - instances = [[5, 6, 7, 8]]
                - starting_points = [4]
                - dates = [datetime.date(2021, 1, 2)]
            )]
        """

        return self.__nodes

    @property
    def graph(self) -> nx.classes.digraph.DiGraph:
        """
        Returns the graph

        Returns:
            `DiGraph`. The graph

        Examples:
            >>> tree = ClusterTree()
            >>> tree.assign_node(Cluster(np.array([1, 1, 1, 1]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)))
            >>> tree.assign_node(Cluster(np.array([5, 5, 5, 5]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)))
            >>> tree.graph
            <networkx.classes.digraph.DiGraph at 0x7f8c3c8f9d90>
        """

        return self.__graph

    @property
    def edges(self) -> nx.classes.reportviews.OutEdgeDataView:
        """
        Returns the edges of the graph

        Returns:
            `OutEdgeDataView`. The edges of the graph

        Examples:
            >>> tree = ClusterTree()
            >>> tree.assign_node(Cluster(np.array([1, 1, 1, 1]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)))
            >>> tree.assign_node(Cluster(np.array([5, 5, 5, 5]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)))
            >>> tree.add_edge(0, 1, is_left=True)
            >>> tree.edges
            OutEdgeDataView([(0, 1, {'left': True})])
        """

        return self.__graph.edges.data()

    @property
    def hierarchies(self) -> list[int]:
        """
        Returns the list of hierarchies from the graph

        Returns:
            `list[int]`. The list of hierarchies

        Examples:
            >>> tree = ClusterTree()
            >>> tree.assign_node(Cluster(np.array([4, 4, 4, 4]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)))
            >>> tree.assign_node(Cluster(np.array([5, 5, 5, 5, 5]), Sequence(Subsequence(np.array([5, 6, 7, 8, 9]), datetime.date(2021, 1, 2), 4)))
            >>> tree.hierarchies
            [4, 5]
        """

        hierarchies = [cluster.length_cluster_subsequences for cluster in self.__nodes]
        return list(set(hierarchies))

    @staticmethod
    def is_left_child(parent: Cluster, child: Cluster) -> bool:
        """
        Check if a cluster is the left child of another cluster.

        Parameters:
            * parent: `Cluster`. The parent cluster.
            * child: `Cluster`. The child cluster.

        Returns:
            `bool`. True if the child cluster is the left child of the parent cluster, False otherwise.

        Raises:
            TypeError: If the parent or child clusters are not Cluster objects.

        Examples:
            >>> parent = Cluster(centroid=np.array([1, 2, 3]), instances=Sequence(Subsequence(np.array([1, 2, 3]), date=datetime.date(2024, 1, 1), starting_point=0)))
            >>> child = Cluster(centroid=np.array([3, 2, 1]), instances=Sequence(Subsequence(np.array([3, 2, 1]), date=datetime.date(2024, 1, 1), starting_point=0))
            >>> drgs = DRGS(length_range=(3, 8), R=2, C=3, G=4, epsilon=0.5)
            >>> is_left = drgs.is_left_child(parent, child)
            >>> print(is_left)
            False
        """

        # Get the starting points of the parent and child clusters
        parent_sp, child_sp = ClusterTree().__get_parent_child_starting_points(parent, child)

        # If its left child and is not a right child
        if all([x + 1 in parent_sp for x in child_sp]) and not all([x in parent_sp for x in child_sp]):
            return True

        # If its both left and right childs
        if all([x + 1 in parent_sp for x in child_sp]) and all([x in parent_sp for x in child_sp]):
            return parent_sp[0] == child_sp[0]

        return False

    @staticmethod
    def is_right_child(parent: Cluster, child: Cluster) -> bool:
        """
        Check if a cluster is the right child of another cluster.

        Parameters:
            * parent: `Cluster`. The parent cluster.
            * child: `Cluster`. The child cluster.

        Returns:
            `bool`. True if the child cluster is the right child of the parent cluster, False otherwise.

        Raises:
            TypeError: If the parent or child clusters are not Cluster objects.

        Examples:
            >>> parent = Cluster(centroid=np.array([1, 2, 3]), instances=Sequence(Subsequence(np.array([1, 2, 3]), date=datetime.date(2024, 1, 1), starting_point=0)))
            >>> child = Cluster(centroid=np.array([3, 2, 1]), instances=Sequence(Subsequence(np.array([3, 2, 1]), date=datetime.date(2024, 1, 1), starting_point=0))
            >>> drgs = DRGS(length_range=(3, 8), R=2, C=3, G=4, epsilon=0.5)
            >>> is_right = drgs.is_right_child(parent, child)
            >>> print(is_right)
            False
        """

        # Get the starting points of the parent and child clusters
        parent_sp, child_sp = ClusterTree().__get_parent_child_starting_points(parent, child)

        # If its right child and is not a left child
        if all([x in parent_sp for x in child_sp]) and not all([x + 1 in parent_sp for x in child_sp]):
            return True

        # If its both left and right childs
        if all([x in parent_sp for x in child_sp]) and all([x + 1 in parent_sp for x in child_sp]):
            return parent_sp[0] != child_sp[0]

        return False

    @staticmethod
    def __check_plot_params(**kwargs):
        """
        Checks the plot parameters

        Parameters:
            * kwargs: `dict`. The plot parameters

        Raises:
            ValueError: if the plot parameters are not valid

        Examples:
            >>> ClusterTree.__check_plot_params()
            >>> ClusterTree.__check_plot_params(color='blue', edge_color='red')
        """

        integer_params = ["node_size", "title_fontsize"]
        string_params = ["title", "save_dir"]
        tuple_params = ["figsize"]
        boolean_params = ["with_labels"]

        for key, value in kwargs.items():
            if key in integer_params and not isinstance(value, int):
                raise TypeError(f"{key} has to be an integer. Got {type(value).__name__} instead")

            if key in string_params and not isinstance(value, str) and value is not None:
                raise TypeError(f"{key} has to be a string. Got {type(value).__name__} instead")

            if key in boolean_params and not isinstance(value, bool):
                raise TypeError(f"{key} has to be a boolean. Got {type(value).__name__} instead")

            if key in tuple_params:
                if not isinstance(value, tuple):
                    raise TypeError(f"{key} has to be a tuple. Got {type(value).__name__} instead")

                if len(value) != 2:
                    raise ValueError(f"{key} has to have two elements. Got {len(value)} instead")

                if not all(isinstance(val, int) for val in value):
                    raise TypeError(
                        f"{key} has to be a tuple of integers. Got tuple[{', '.join([type(val).__name__ for val in value])}] instead")

    @staticmethod
    def __convert_edges_to_list(edges: nx.classes.reportviews.OutEdgeDataView) -> list[tuple[int, int, bool]]:
        """
        Takes the edges and converts them to a list of tuples

        Parameters:
            * edges: `OutEdgeDataView`. The edges to convert

        Returns:
            `list[tuple[int, int, bool]]`. The list of edges as tuples

        Raises:
            TypeError: if the edges is not an instance of OutEdgeDataView

        Examples:
            >>> tree = ClusterTree()
            >>> tree.assign_node(Cluster(np.array([1, 1, 1, 1]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)))
            >>> tree.assign_node(Cluster(np.array([5, 5, 5, 5]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)))
            >>> tree.add_edge(0, 1, is_left=True)
            >>> edges = tree.edges
            >>> tree.__convert_edges_to_list(edges)
            [(0, 1, True)]
        """

        # Check if the edges is an instance of OutEdgeDataView
        if not isinstance(edges, nx.classes.reportviews.OutEdgeDataView):
            raise TypeError(f"edges has to be an instance of OutEdgeDataView. Got {type(edges).__name__} instead")

        return list(edges)

    @staticmethod
    def __get_parent_child_starting_points(parent: Cluster, child: Cluster) -> tuple[list[int], list[int]]:
        # Check if the parent and child clusters are Cluster objects
        if not isinstance(parent, Cluster):
            raise TypeError(f"The parent must be a Cluster object. Got {type(parent).__name__} instead.")

        if not isinstance(child, Cluster):
            raise TypeError(f"The child must be a Cluster object. Got {type(child).__name__} instead.")

        # Get the parent and child sequences
        parent_sequences = parent.get_sequences()
        child_sequences = child.get_sequences()

        # Get the starting points of the parent and child sequences
        parent_starting_points = parent_sequences.get_starting_points()
        child_starting_points = child_sequences.get_starting_points()

        return parent_starting_points, child_starting_points

    def __check_and_convert_node(self, node: Union[Cluster, int, str]) -> int:
        """
        Checks if the node is a Cluster instance or an integer.

        Parameters:
            * node: `Union[Cluster, int]`. The node to check

        Returns:
            `int`. The index of the node

        Raises:
            TypeError: if the node is not an instance of Cluster or an integer
            IndexError: if the node is not found in the list of clusters

        Examples:
            >>> tree = ClusterTree()
            >>> cluster = Cluster(np.array([1, 1, 1, 1]), Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> tree.assign_node(cluster)
            >>> tree.__check_and_convert_node(0)
            0

            >>> tree.__check_and_convert_node(cluster)
            0
        """

        # Check if the node is an integer or a Cluster instance
        if not isinstance(node, (Cluster, int, str)):
            raise TypeError(
                f"node has to be either an integer or a Cluster instance. Got {type(node).__name__} instead")

        if isinstance(node, int) and node not in self.__list_of_index:
            raise IndexError(f"Node {node} not found in the list of clusters")

        if isinstance(node, str) and node not in self.__name_node:
            raise IndexError(f"Node Name {node} not found in the list of clusters")

        if isinstance(node, str):
            idx = self.__name_node.index(node)
            return self.__list_of_index[idx]

        # If the node is a Cluster instance, we get the index
        if isinstance(node, Cluster):
            # Check if the node exists in the list of clusters
            if node not in self.__nodes:
                raise IndexError(f"Cluster {node} not found in the list of clusters {self.__nodes}")

            # Get the index of the node
            idx = self.__nodes.index(node)
            return self.__list_of_index[idx]

        return node

    def __assign_edge_color(self) -> list[str]:
        """
        Assigns the color to the edges of the graph to visualize the hierarchy and the left and right children.
        The left children are colored in blue, and the right children are colored in red.

        Returns:
            `list[str]`. The list of colors for the edges

        Examples:
            >>> tree = ClusterTree()
            >>> tree.assign_node(Cluster(np.array([1, 1, 1, 1]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)))
            >>> tree.assign_node(Cluster(np.array([5, 5, 5, 5]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)))
            >>> tree.assign_node(Cluster(np.array([7, 7, 7, 7]), Sequence(Subsequence(np.array([9, 10, 11, 12]), datetime.date(2021, 1, 3), 8)))

            >>> tree.add_edge(1, 2, is_left=True)
            >>> tree.add_edge(1, 3, is_left=False)

            >>> tree.__assign_edge_color()
            ['blue', 'red']

        """
        # Initialize the colors as red (assume that all the edges are right children)
        colors = ['red'] * len(self.__graph.edges)

        # Get the formated edges
        left = self.__graph.edges.data("left")
        left_formatted = self.__convert_edges_to_list(left)

        # Assign the color to the left children
        for idx, (parent, child, is_left) in enumerate(left_formatted):
            if is_left:
                colors[idx] = "blue"

        return colors

    def __assign_nodes_color(self) -> list[np.ndarray]:
        available_hierarchies = self.hierarchies
        n_clusters = [len(self.get_nodes_with_hierarchy(hier)) for hier in available_hierarchies]
        n_max = max(n_clusters)

        if len(self.__name_node) > 0:
            n_max = max([int(name.split("-")[1]) for name in self.__name_node])

        base_colors = cm.rainbow(np.linspace(0, 1, n_max))
        colors = []
        for hier in available_hierarchies:
            if len(self.__name_node) > 0:
                clusters = [int(name.split("-")[1]) for name in self.__name_node if int(name.split("-")[0]) == hier]
                for idx in clusters:
                    colors.append(base_colors[idx - 1])

            else:
                clusters = self.get_nodes_with_hierarchy(hier)
                for cluster in clusters:
                    idx = clusters.index(cluster)
                    colors.append(base_colors[idx])

        return list(colors)

    def __assign_labels(self) -> dict:
        """
        Assigns the labels to the nodes of the graph

        Returns:
            `dict`. A dictionary where the keys are the indexes and the values are the labels
        """

        labels = []
        available_hierarchies = self.hierarchies

        if len(self.__name_node) > 0:
            for name in self.__name_node:
                labels.append(name)

        else:
            for hierarchy in available_hierarchies:
                clusters = self.get_nodes_with_hierarchy(hierarchy)
                for cluster in clusters:
                    idx = clusters.index(cluster)
                    labels.append(f"{hierarchy}-{idx + 1}")

        return {node: label for node, label in zip(self.__list_of_index, labels)}

    def __check_clusters_from_edges(self, parent: Cluster, child: Cluster) -> None:
        """
        Checks if the parent and child clusters are in the list of clusters and if the hierarchy is correct.
        Also, checks if exists and edge between the parent and child clusters.

        Parameters:
            * parent: `Cluster`. The parent cluster
            * child: `Cluster`. The child cluster

        Raises:
            IndexError: if the parent or child is not in the list of clusters
            ValueError: if it happens some of the following cases:
                        the parent and child are the same cluster;
                        the child hierarchy is not the parent hierarchy + 1;
                        the child is not a child of the parent
        """

        # Get the hierarchy from the parent and child
        parent_hierarchy = parent.length_cluster_subsequences
        child_hierarchy = child.length_cluster_subsequences

        index_parent = self.get_index(parent)
        index_child = self.get_index(child)

        # Check if the parent is in the list of clusters
        if parent not in self.__nodes:
            raise IndexError(f"Cluster {parent} not found in the list of clusters")

        # Check if the child is in the list of clusters
        if child not in self.__nodes:
            raise IndexError(f"Cluster {child} not found in the list of clusters")

        # Check if the parent and child are the same cluster
        if parent == child:
            raise ValueError("The parent and child cannot be the same cluster")

        # Check if the child hierarchy is the parent hierarchy + 1
        if child_hierarchy != parent_hierarchy + 1:
            raise ValueError(
                f"The child hierarchy must be the parent hierarchy + 1. Expected {parent_hierarchy + 1}. Got {child_hierarchy} instead")

        # Check if the child is a child of the parent
        children = [child_idx for parent_idx, child_idx, _ in self.edges if parent_idx == index_parent]

        # Check if the child is in the children
        if index_child in children:
            raise ValueError(f"The edge between {index_parent} and {index_child} already exists")

    def __check_and_return_edge(self, parent: Union[Cluster, int], child: Union[Cluster, int]) -> tuple[int, int]:
        """
        Checks if the parent and child are in the list of clusters and returns the index of the clusters

        Parameters:
            * parent: `Union[Cluster, int]`. The parent cluster
            * child: `Union[Cluster, int]`. The child cluster

        Returns:
            `tuple[int, int]`. The clusters indexes from the parent and child

        Raises:
            TypeError: if the parent or child is not an instance of Cluster or an integer
            IndexError: if the parent or child is not in the list of clusters
            ValueError: if the parent and child are the same cluster or if the child hierarchy is not the parent hierarchy + 1

        Examples:
            >>> tree = ClusterTree()
            >>> cluster1 = Cluster(np.array([1, 1, 1, 1]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> cluster2 = Cluster(np.array([5, 5, 5, 5]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))

            >>> tree.assign_node(cluster1)
            >>> tree.assign_node(cluster2)

            >>> tree.__check_and_return_edge(0, 1)
            (Cluster(
                - centroid = [1, 1, 1, 1],
                - instances = [[1, 2, 3, 4]]
                - starting_points = [0]
                - dates = [datetime.date(2021, 1, 1)]
            ), Cluster(
                - centroid = [5, 5, 5, 5],
                - instances = [[5, 6, 7, 8]]
                - starting_points = [4]
                - dates = [datetime.date(2021, 1, 2)]
            ))
        """

        # Check if the parent and child are instances of Cluster or integers
        if not isinstance(parent, (int, Cluster)) or not isinstance(child, (int, Cluster)):
            raise TypeError(
                f"The parent and child must be either an integer or a Cluster instance. Got {type(parent).__name__} and {type(child).__name__} instead")

        # If the parent is a Cluster instance, we get the index
        if type(parent) != type(child):
            raise TypeError(
                f"The parent and child must be the same type. Got {type(parent).__name__} and {type(child).__name__} instead")

        # If the parent is a Cluster instance, we get the index
        if isinstance(parent, Cluster):
            self.__check_clusters_from_edges(parent, child)
            return self.get_index(parent), self.get_index(child)

        else:
            # Check if the parent is in the list of clusters
            if parent not in self.__list_of_index:
                raise IndexError(f"Index {parent} not found in the list of indexes {self.__list_of_index}")

            # Check if the child is on the list of clusters
            if child not in self.__list_of_index:
                raise IndexError(f"Index {child} not found in the list of indexes {self.__list_of_index}")

            # Get the parent and child clusters indexes
            parent_clust = self.get_node(parent)
            child_clust = self.get_node(child)

            # Check if the parent and child clusters are correct
            self.__check_clusters_from_edges(parent_clust, child_clust)

            return parent, child

    def __check_no_left_right_repeat(self, parent: int, is_left: bool) -> None:
        """
        Checks if the parent has already a left or right child

        Parameters:
            * parent: `int`. The parent index
            * is_left: `bool`. If the child is the left child

        Raises:
            ValueError: raises value error on the following cases:
                        the parent already has a left or right child;
                        the left child already exists for the parent;
                        the right child already exists for the parent

        Examples:
            >>> parent = Cluster(np.array([1, 2, 3]), Sequence(Subsequence(np.array([1, 2, 3]), datetime.date(2021, 1, 1), 1))
            >>> left = Cluster(np.array([0, 1, 2, 3]), Sequence(Subsequence(np.array([0, 1, 2, 3]), datetime.date(2021, 1, 2), 0))
            >>> right = Cluster(np.array([1, 2, 3, 4]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 3), 1))

            >>> tree = ClusterTree()
            >>> tree.assign_node(parent)
            >>> tree.assign_node(left)
            >>> tree.assign_node(right)

            >>> tree.add_edge(1, 2, is_left=True)
            >>> tree.add_edge(1, 3, is_left=False)
            >>> tree.__check_no_left_right_repeat(1, is_left=False)
            ValueError: Parent Cluster(
                already has a left or right child

            >>> tree.__check_no_left_right_repeat(1, is_left=True)
            ValueError: Left child already exists for parent 0

        """

        existent_children = self.children(parent)

        # If there are name nodes, we get the name of the parent and children nodes
        if len(self.name_nodes) > 0:
            parent = self.get_name_node(parent)
            existent_children = [self.get_name_node(child) for child in existent_children]

        # Check if the parent already has two children: left and right child
        if len(existent_children) == 2:
            raise ValueError(
                f"Parent {parent} already has two children: left: {existent_children[0]}; right: {existent_children[1]}")

        # Check if the left child already exists for the parent
        if is_left and any(self.__graph.edges[parent, child]['left'] for child in existent_children):
            raise ValueError(f"Left child already exists for parent {parent}")

        # Check if the right child already exists for the parent
        if not is_left and any(not self.__graph.edges[parent, child]['left'] for child in existent_children):
            raise ValueError(f"Right child already exists for parent {parent}")

    def get_name_node(self, node: Union[Cluster, int, str]) -> str:
        """
        Returns the name of the node

        Parameters:
            * node: Union[`Cluster`, `int`, `str`]. The node to get the name

        Returns:
            * `str`. The name of the node

        Raises:
            TypeError: if the node is not an instance of Cluster, integer or string
            ValueError: if the node doesn't exist

        Examples:
            >>> tree = ClusterTree()
            >>> parent = Cluster(np.array([1, 2, 3]), Sequence(Subsequence(np.array([1, 2, 3]), datetime.date(2021, 1, 1), 1))
            >>> left = Cluster(np.array([0, 1, 2, 3]), Sequence(Subsequence(np.array([0, 1, 2, 3]), datetime.date(2021, 1, 2), 0))
            >>> right = Cluster(np.array([1, 2, 3, 4]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 3), 1)

            >>> tree.assign_node(parent)
            >>> tree.assign_node(left)
            >>> tree.assign_node(right)

            >>> tree.get_name_node(parent)
            '3-1'

            >>> tree.get_name_node(left)
            '4-1'

            >>> tree.get_name_node(right)
            '4-2'
        """

        # Check if the node is an instance of Cluster
        if not isinstance(node, (Cluster, int, str)):
            raise TypeError(f"node has to be an instance of Cluster. Got {type(node).__name__} instead")

        # Check if the node is in the list of nodes
        if isinstance(node, Cluster):
            if node not in self.__nodes:
                raise ValueError(f"Node {node} not found in the list of nodes")

            idx = self.__nodes.index(node)
            return self.__name_node[idx]

        # If the node is an integer, we get the name of the node
        if isinstance(node, int):
            if node not in self.__list_of_index:
                raise ValueError(f"Node {node} not found in the list of nodes")

            idx = self.__list_of_index.index(node)
            return self.__name_node[idx]

        # If the node is a string, we check if it exists in the list of names
        if node not in self.__name_node:
            raise ValueError(f"Node {node} not found in the list of nodes")

        return node

    def get_node(self, node: Union[int, str, Cluster]) -> Cluster:
        # Check if the node is an instance of Cluster, string or integer
        if not isinstance(node, (int, str, Cluster)):
            raise TypeError(f"node has to be an integer, string or Cluster. Got {type(node).__name__} instead")

        # If the node is a string, we get the index from the name nodes
        if isinstance(node, str):
            if node not in self.__name_node:
                raise ValueError(f"Node {node} not found in the list of nodes {self.__name_node}")

            idx = self.__name_node.index(node)
            return self.__nodes[idx]

        # If the node is an integer, we check if it exists in the list of indexes
        if isinstance(node, int):
            if node not in self.__list_of_index:
                raise ValueError(f"Node {node} not found in the list of nodes {self.__list_of_index}")

            idx = self.__list_of_index.index(node)
            return self.__nodes[idx]

        # If the node is a Cluster, we check if it exists in the list of nodes
        if node not in self.__nodes:
            raise ValueError(f"Node {node} not found in the list of nodes")

        return node

    def get_index(self, node: Union[Cluster, str, int]) -> int:

        # Check if the node is an instance of Cluster, string or integer
        if not isinstance(node, (Cluster, str, int)):
            raise TypeError(f"node has to be a Cluster, string or integer instance. Got {type(node).__name__} instead")

        # If the node is a string, we get the index from the name nodes
        if isinstance(node, str):
            if node not in self.__name_node:
                raise ValueError(f"Node {node} not found in the list of nodes {self.__name_node}")

            idx = self.__name_node.index(node)
            return self.__list_of_index[idx]

        # If the node is a Cluster, we get the index from the list of nodes
        if isinstance(node, Cluster):
            if node not in self.__nodes:
                raise ValueError(f"Node {node} not found in the list of nodes")

            idx = self.__nodes.index(node)
            return self.__list_of_index[idx]

        # If the node is an integer, we check if it exists in the list of indexes
        if node not in self.__list_of_index:
            raise ValueError(f"Node {node} not found in the list of nodes {self.__list_of_index}")

        return node

    def get_nodes_with_hierarchy(self, hierarchy: int) -> list[Cluster]:
        """
        Returns the list of clusters with the specified hierarchy

        Parameters:
            * hierarchy: `int`. The hierarchy of the clusters

        Returns:
            `list[Cluster]`. The list of clusters with the specified hierarchy
        """

        available_hierarchies = self.hierarchies

        # Check if the hierarchy is an integer
        if not isinstance(hierarchy, int):
            raise TypeError(f"hierarchy has to be an integer. Got {type(hierarchy).__name__} instead")

        # Check if the hierarchy exists
        if hierarchy not in available_hierarchies:
            raise IndexError(f"Hierarchy {hierarchy} not found in the available hierarchies {available_hierarchies}")

        list_cluster = []
        for cluster in self.__nodes:
            if cluster.length_cluster_subsequences == hierarchy:
                list_cluster.append(cluster)

        return list_cluster

    def to_dictionary(self) -> dict:
        """
        Returns a dictionary where the keys are the index of the nodes and the values are the clusters

        Returns:
            `dict`. The graph as a dictionary

        Examples:
            >>> tree = ClusterTree()
            >>> tree.assign_node(Cluster(np.array([1, 1, 1, 1]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)))
            >>> tree.assign_node(Cluster(np.array([5, 5, 5, 5]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)))
            >>> tree.add_edge(0, 1, is_left=True)
            >>> tree.to_dictionary()
            {0: Cluster(
                - centroid = [1, 1, 1, 1],
                - instances = [[1, 2, 3, 4]]
                - starting_points = [0]
                - dates = [datetime.date(2021, 1, 1)]
            ), 1: Cluster(
                - centroid = [5, 5, 5, 5],
                - instances = [[5, 6, 7, 8]]
                - starting_points = [4]
                - dates = [datetime.date(2021, 1, 2)]
            )}
        """

        return dict(zip(self.__list_of_index, self.__nodes))

    def children(self, node: Union[Cluster, int, str]) -> list[int]:
        """
        Returns the index of clusters from the children of the node

        Parameters:
            * node: `Union[Cluster, int]`. The node to get the children

        Returns:
            `list[int]`. The indexes of the children clusters

        Raises:
            TypeError: if the node is not an instance of Cluster or an integer
            IndexError: if the node is not in the list of clusters

        Examples:
            >>> tree = ClusterTree()
            >>> cluster1 = Cluster(np.array([1, 1, 1, 1]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)))
            >>> cluster2 = Cluster(np.array([5, 5, 5, 5]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)))
            >>> cluster3 = Cluster(np.array([7, 7, 7, 7]), Sequence(Subsequence(np.array([9, 10, 11, 12]), datetime.date(2021, 1, 3), 8)))

            >>> tree.assign_node(cluster1)
            >>> tree.assign_node(cluster2)
            >>> tree.assign_node(cluster3)

            >>> tree.add_edge(1, 2, is_left=True)
            >>> tree.add_edge(1, 3, is_left=False)

            >>> tree.children(1)
            [2, 3]
        """

        # Check and convert the node to index
        node = self.__check_and_convert_node(node)

        # Get the edges from the graph
        edges = list(self.edges)

        # Get the children from the node
        children = [child for parent, child, _ in edges if parent == node]

        return children

    def is_child(self, parent: Union[Cluster, int], child: Union[Cluster, int]) -> bool:
        """
        Indicating the parent and child nodes, returns if the child is a child from the parent

        Parameters:
            * parent: `Union[Cluster, int]`. The parent node
            * child: `Union[Cluster, int]`. The child node

        Returns:
            `bool`. `True` if the child is a child from the parent, `False` otherwise

        Raises:
            TypeError: if the parent or child is not an instance of Cluster or an integer
            IndexError: if the parent or child is not in the list of clusters

        Examples:
            >>> tree = ClusterTree()
            >>> cluster1 = Cluster(np.array([1, 1, 1, 1]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> cluster2 = Cluster(np.array([5, 5, 5, 5]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))

            >>> tree.assign_node(cluster1)
            >>> tree.assign_node(cluster2)

            >>> tree.add_edge(0, 1, is_left=True)
            >>> tree.is_child(0, 1)
            True

            >>> tree.is_child(1, 0)
            False
        """

        # Initialize the variable
        is_child_value = False

        parent = self.get_index(parent)
        child = self.get_index(child)

        # Get the edges from the graph
        edges = list(self.edges)

        # Get the children from the parent
        children = [child for parent_idx, child_idx, _ in edges if parent_idx == parent]

        # Check if the child is in the children
        if child in children:
            is_child_value = True

        return is_child_value

    def is_existent_left_child(self, parent: Union[Cluster, int], child: Union[Cluster, int]) -> bool:
        """

        Indicating the parent and child nodes, returns if the child is the left child from the parent

        Parameters:
            * parent: `Union[Cluster, int]`. The parent node
            * child: `Union[Cluster, int]`. The child node

        Returns:
            `bool`. `True` if the child is the left child from the parent, `False` if right

        Raises:
            TypeError: if the parent or child is not an instance of Cluster or an integer
            IndexError: if the parent or child is not in the list of clusters
            ValueError: if the child is not a child from the parent

        Examples:
            >>> tree = ClusterTree()
            >>> cluster1 = Cluster(np.array([1, 1, 1, 1]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> cluster2 = Cluster(np.array([5, 5, 5, 5]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))

            >>> tree.assign_node(cluster1)
            >>> tree.assign_node(cluster2)

            >>> tree.add_edge(0, 1, is_left=True)
            >>> tree.is_left_child(0, 1)
            True
        """

        # Check and convert the parent and child to index
        # parent, child = self.__check_and_return_edge(parent, child)
        # print("llego hasta aqui")
        # If the child is not a child from the parent, we raise a ValueError
        if not self.is_child(parent, child):
            raise ValueError(f"The child {child} is not a child from the parent {parent}")

        # Get the edges from the graph
        edges = list(self.edges)
        parent, child = self.get_index(parent), self.get_index(child)

        # Get the left parameter from the child
        for parent_idx, child_idx, is_left in edges:
            if parent_idx == parent and child_idx == child:
                return is_left

    def parents(self, node: Union[Cluster, int, str]) -> list[int]:
        """
        Returns the index of clusters from the parents of the node

        Parameters:
            * node: `Union[Cluster, int]`. The node to get the parents

        Returns:
            `list[int]`. The indexes of the parent clusters

        Raises:
            TypeError: if the node is not an instance of Cluster or an integer
            IndexError: if the node is not in the list of clusters

        Examples:
            >>> tree = ClusterTree()
            >>> cluster1 = Cluster(np.array([1, 2, 3]), Sequence(Subsequence(np.array([1, 2, 3]), datetime.date(2021, 1, 1), 1))
            >>> cluster2 = Cluster(np.array([0, 1, 2, 3]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 0))

            >>> tree.assign_node(cluster1)
            >>> tree.assign_node(cluster2)
            >>> tree.add_edge(cluster1, cluster2, is_left=True)

            >>> tree.parents(cluster2)
            [1]

            >>> tree.parents(cluster1)
            []
        """

        # Check and convert the node to index
        node = self.__check_and_convert_node(node)

        # Get the edges from the graph
        edges = list(self.edges)

        # Get the parents from the node
        parents = [parent for parent, child, _ in edges if child == node]
        return parents

    def has_children(self, node: Union[Cluster, int, str]) -> bool:
        """
        Indicates if the node has children

        Returns:
            `bool`. `True` if the node has children, `False` otherwise

        Raises:
            TypeError: if the node is not an instance of Cluster or an integer
            IndexError: if the node is not in the list of clusters

        Examples:
            >>> tree = ClusterTree()
            >>> cluster1 = Cluster(np.array([1, 2, 3]), Sequence(Subsequence(np.array([1, 2, 3]), datetime.date(2021, 1, 1), 1))
            >>> cluster2 = Cluster(np.array([0, 1, 2, 3]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 0))

            >>> tree.assign_node(cluster1)
            >>> tree.assign_node(cluster2)
            >>> tree.add_edge(1, 2, is_left=True)

            >>> tree.has_children(1)
            True

            >>> tree.has_children(2)
            False
        """

        return len(self.children(node)) > 0

    def has_parents(self, node: Union[Cluster, int, str]) -> bool:
        """
        Indicates if the node has parents

        Returns:
            `bool`. `True` if the node has parents, `False` otherwise

        Examples:
            >>> tree = ClusterTree()
            >>> cluster1 = Cluster(np.array([1, 2, 3]), Sequence(Subsequence(np.array([1, 2, 3]), datetime.date(2021, 1, 1), 1))
            >>> cluster2 = Cluster(np.array([0, 1, 2, 3]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 0))

            >>> tree.assign_node(cluster1)
            >>> tree.assign_node(cluster2)
            >>> tree.add_edge(0, 1, is_left=True)

            >>> tree.has_parents(cluster1)
            False

            >>> tree.has_parents(cluster2)
            True
        """

        return len(self.parents(node)) > 0

    def assign_names(self):
        """
        Assigns names to the nodes of the graph based on the hierarchy and
        the order of assignment as follows: **hierarchy-index**

        Examples:
            >>> cluster1 = Cluster(np.array([1, 2, 3]), Sequence(Subsequence(np.array([1, 2, 3]), datetime.date(2021, 1, 1), 1)))
            >>> cluster2 = Cluster(np.array([0, 1, 2, 3]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 0)))

            >>> tree = ClusterTree()
            >>> tree.assign_node(cluster1)
            >>> tree.assign_node(cluster2)
            >>> tree.assign_names()

            >>> for cluster in [cluster1, cluster2]:
            ...     name = tree.get_name_node(cluster)
            ...     print(name)
            3-1
            4-1
        """

        available_hierarchies = self.hierarchies

        # For each hierarchy, we assign a name to the nodes
        for hierarchy in available_hierarchies:
            # Get the clusters from the hierarchy
            clusters = self.get_nodes_with_hierarchy(hierarchy)

            # Assign for each node belonging to the hierarchy a name
            for k in range(len(clusters)):
                self.__name_node.append(f"{hierarchy}-{k + 1}")

    def reset_names(self):
        """
        Resets the names of the nodes

        Examples:
            >>> cluster1 = Cluster(np.array([1, 2, 3]), Sequence(Subsequence(np.array([1, 2, 3]), datetime.date(2021, 1, 1), 1)))
            >>> cluster2 = Cluster(np.array([0, 1, 2, 3]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 0)))

            >>> tree = ClusterTree()
            >>> tree.assign_node(cluster1)
            >>> tree.assign_node(cluster2)
            >>> tree.assign_names()

            >>> for name in tree.__name_node:
            ...     print(name)
            3-1
            4-1

            >>> tree.reset_names()
            >>> print(tree.__name_node)
            []
        """

        self.__name_node: list[str] = []

    def assign_node(self, cluster: Cluster) -> None:
        """
        Assigns a node to the graph.
        The node is a cluster that will be added to the list of clusters and,
        automatically, it will assign a unique index to the node that starts from 1.

        Parameters:
            * cluster: `Cluster`.
            The cluster to assign to the graph

        Raises:
            TypeError: if the cluster is not an instance of Cluster
            ValueError: if the cluster already exists in the list of clusters

        Examples:
            >>> tree = ClusterTree()
            >>> cluster1 = Cluster(np.array([1, 2, 3]), Sequence(Subsequence(np.array([1, 2, 3]), datetime.date(2021, 1, 1), 1))
            >>> cluster2 = Cluster(np.array([0, 1, 2, 3]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 0))

            >>> tree.assign_node(cluster1)
            >>> tree.assign_node(cluster2)
            >>> for node in tree.nodes:
            ...     print(node)
            Cluster(
                - centroid = [1, 2, 3]
                - instances = [[1, 2, 3]]
                - starting_points = [1]
                - dates = [datetime.date(2021, 1, 1)]
            )
            Cluster(
                - centroid = [0, 1, 2, 3]
                - instances = [[5, 6, 7, 8]]
                - starting_points = [0]
                - dates = [datetime.date(2021, 1, 2)]
            )
        """

        # Check if the cluster is an instance of Cluster
        if not isinstance(cluster, Cluster):
            raise TypeError(f"cluster has to be an instance of Cluster. Got {type(cluster).__name__} instead")

        # Check if the cluster already exists in the list of clusters
        if cluster in self.__nodes:
            raise ValueError(f"Cluster {cluster} already exists in the list of clusters")

        idx_to_add = len(self.__nodes) + 1

        # Add the node to the graph
        self.__graph.add_node(idx_to_add, cluster=cluster)
        self.__nodes.append(cluster)
        self.__list_of_index.append(idx_to_add)

    def add_edge(self, vertex_parent: Union[Cluster, int], vertex_child: Union[Cluster, int], is_left: bool) -> None:
        """
        Adds an edge to the graph, indicating the parent and child clusters and if the child is the left or right child

        Parameters:
            * vertex_parent: `Union[Cluster, int]`. The parent cluster
            * vertex_child: `Union[Cluster, int]`. The child cluster
            * is_left: `bool`. If the child is the left child from the parent

        Raises:
            TypeError: if the parent or child is not an instance of Cluster or an integer
            IndexError: if the parent or child is not in the list of clusters
            ValueError: raises a value error in the following cases:
                        the parent and child are the same cluster;
                        the child hierarchy is not the parent hierarchy + 1;
                        the parent already has two children;
                        the left or right child already exists

        Examples:
            >>> tree = ClusterTree()
            >>> cluster1 = Cluster(np.array([1, 2, 3]), Sequence(Subsequence(np.array([1, 2, 3]), datetime.date(2021, 1, 1), 1))
            >>> cluster2 = Cluster(np.array([0, 1, 2, 3]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 0))

            >>> tree.assign_node(cluster1)
            >>> tree.assign_node(cluster2)

            >>> tree.add_edge(1, 2, is_left=True)
            >>> tree
        """

        # Check the validity of the parent and child clusters
        parent, child = self.__check_and_return_edge(vertex_parent, vertex_child)

        # Check if the parent already has two and only one left and right child
        self.__check_no_left_right_repeat(parent, is_left)

        # Add the edge to the graph
        self.__graph.add_edge(parent, child, left=is_left)

    def drop_node(self, node: Union[Cluster, int, str]):
        """
        Drops the node from the graph and those nodes that depends on it
        (children that has no more parents than the dropped node)

        Parameters:
            * node: `Union[Cluster, int, str]`. The node to drop

        Raises:
            TypeError: if the node is not an instance of Cluster or an integer
            IndexError: if the node is not in the list of clusters

        Examples:
            >>> tree = ClusterTree()
            >>> cluster1 = Cluster(np.array([1, 2, 3]), Sequence(Subsequence(np.array([1, 2, 3]), datetime.date(2021, 1, 1), 0)))
            >>> cluster2 = Cluster(np.array([5, 6, 7, 8]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)))
            >>> cluster3 = Cluster(np.array([10, 11, 12, 13]), Sequence(Subsequence(np.array([10, 11, 12, 13]), datetime.date(2021, 1, 3), 5)))
            >>> cluster4 = Cluster(np.array([15, 16, 17, 18, 19]), Sequence(Subsequence(np.array([15, 16, 17, 18, 19]), datetime.date(2021, 1, 4), 6)))
            >>> cluster5 = Cluster(np.array([20, 21, 22, 23, 24]), Sequence(Subsequence(np.array([20, 21, 22, 23, 24]), datetime.date(2021, 1, 5), 7)))
            >>> cluster6 = Cluster(np.array([25, 26, 27, 28, 29]), Sequence(Subsequence(np.array([25, 26, 27, 28, 29]), datetime.date(2021, 1, 6), 8)))
            >>> cluster7 = Cluster(np.array([30, 31, 32, 33, 34]), Sequence(Subsequence(np.array([30, 31, 32, 33, 34]), datetime.date(2021, 1, 7), 9)))

            >>> tree.assign_node(cluster1)
            >>> tree.assign_node(cluster2)
            >>> tree.assign_node(cluster3)
            >>> tree.assign_node(cluster4)
            >>> tree.assign_node(cluster5)
            >>> tree.assign_node(cluster6)
            >>> tree.assign_node(cluster7)

            >>> tree.add_edge(1, 2, is_left=True)
            >>> tree.add_edge(1, 3, is_left=False)
            >>> tree
        """

        # Check if the node is an instance of Cluster or an integer
        node = self.__check_and_convert_node(node)

        def __get_nodes_to_drop_recursive(node: int, nodes_to_drop: list[int]) -> list[int]:
            """
            Gets the nodes to drop recursively

            Parameters:
                * node: `int`. The node to drop
                * nodes_to_drop: `list[int]`. The list of nodes to drop

            Returns:
                `list[int]`. The list of nodes to drop recursively
            """

            # Get the node and edges
            node: int = self.__check_and_convert_node(node)
            edges = list(self.__graph.edges.data())

            # Get the nodes to drop recursively for those children that have only the node as parent
            for parent, child, _ in edges:
                # If the parent is the node, we check if the child has only one parent
                if parent == node:
                    if len(self.parents(child)) == 1:
                        # If the child has only one parent, we add it to the list of nodes to drop
                        nodes_to_drop += __get_nodes_to_drop_recursive(child, [child])

            return nodes_to_drop

        # Get the nodes to drop recursively
        nodes_to_drop = __get_nodes_to_drop_recursive(node, [node])

        for node in nodes_to_drop:
            # Remove the node from the graph
            self.__graph.remove_node(node)

            # Remove the node from the list of nodes
            idx_to_remove = self.__list_of_index.index(node)
            self.__nodes = [node for idx, node in enumerate(self.__nodes) if idx != idx_to_remove]

            # Remove the index from the list of indexes
            self.__list_of_index = [idx for idx in self.__list_of_index if idx != node]

            # Remove the name from the list of names if it exists
            if len(self.__name_node) > 0:
                self.__name_node = [name for idx, name in enumerate(self.__name_node) if idx != idx_to_remove]

    def convert_to_hierarchical_routines(self) -> HierarchyRoutine:
        """
        Converts the tree to a hierarchical routine

        Returns:
            `HierarchyRoutine`. The hierarchical routine

        Examples:
            >>> tree = ClusterTree()
            >>> cluster1 = Cluster(np.array([1, 2, 3]), Sequence(Subsequence(np.array([1, 2, 3]), datetime.date(2021, 1, 1), 0))
            >>> cluster2 = Cluster(np.array([5, 6, 7, 8]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> cluster3 = Cluster(np.array([10, 11, 12, 13]), Sequence(Subsequence(np.array([10, 11, 12, 13]), datetime.date(2021, 1, 3), 5))

            >>> tree.assign_node(cluster1)
            >>> tree.assign_node(cluster2)
            >>> tree.assign_node(cluster3)

            >>> tree.add_edge(1, 2, is_left=True)
            >>> tree.add_edge(1, 3, is_left=False)

            >>> hierarchy_routine = tree.convert_to_hierarchical_routines()
            >>> hierarchy_routine
        """

        hierarchy_routine = HierarchyRoutine()
        available_hierarchies = self.hierarchies
        for hierarchy in available_hierarchies:
            new_routine = Routines()
            clusters = self.get_nodes_with_hierarchy(hierarchy)
            for cluster in clusters:
                new_routine.add_routine(cluster)

            hierarchy_routine[hierarchy] = new_routine

        return hierarchy_routine

    def plot_tree(self, node_size: int = 1000, with_labels: bool = True,
                  figsize: tuple[int, int] = (7, 7),
                  title_fontsize: int = 15,
                  show_plot: bool = True,
                  title: Optional[str] = None,
                  save_dir: Optional[str] = None):

        """
        Plots the tree with the left and right children colored in blue and red, respectively.
        Each level on the tree corresponds to a hierarchy level of the clusters.

        Parameters:
            * node_size: `int`. The size of the nodes
            * with_labels: `bool`. If True, the labels are shown
            * figsize: `tuple[int, int]`. The size of the figure
            * title: `Optional[str]`. The title of the plot
            * title_fontsize: `int`. The fontsize of the title
            * save_dir: `Optional[str]`. The directory to save the plot

        Examples:
            >>> tree = ClusterTree()
            >>> cluster1 = Cluster(np.array([1, 2, 3]), Sequence(Subsequence(np.array([1, 2, 3]), datetime.date(2021, 1, 1), 0))
            >>> cluster2 = Cluster(np.array([5, 6, 7, 8]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> cluster3 = Cluster(np.array([10, 11, 12, 13]), Sequence(Subsequence(np.array([10, 11, 12, 13]), datetime.date(2021, 1, 3), 5))

            >>> tree.assign_node(cluster1)
            >>> tree.assign_node(cluster2)
            >>> tree.assign_node(cluster3)

            >>> tree.add_edge(1, 2, is_left=True)
            >>> tree.add_edge(1, 3, is_left=False)

            >>> tree.plot_tree(node_size=1000, with_labels=True, figsize=(7, 7), title="Cluster Tree", title_fontsize=15, save_dir="results.png")
        """

        # Check the validity of the plot parameters
        args = locals()
        self.__check_plot_params(**args)

        # Assign the colors to the nodes and edges
        colors_edges = self.__assign_edge_color()
        colors_nodes = self.__assign_nodes_color()

        # Get the labels
        labels = self.__assign_labels()

        # Initialize the figure
        plt.figure(figsize=figsize)

        # Add the title if it is indicated
        if title is not None:
            plt.title(title, fontsize=title_fontsize)

        # Plot the graph
        nx.draw(self.__graph, pos=nx.nx_agraph.graphviz_layout(self.__graph, prog='dot'), with_labels=with_labels,
                node_size=node_size, node_color=colors_nodes, edge_color=colors_edges, labels=labels)

        # Save the plot if it is indicated
        if save_dir is not None:
            format = save_dir.split('.')[-1]
            plt.savefig(save_dir, format=format)

        # Show the plot
        if show_plot:
            plt.show()

        plt.close()
