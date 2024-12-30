from typing import List, Union, Tuple, Optional
import numpy as np
from tree import Tree, Node
from scipy.linalg import expm

COMPARISON_TYPES = {'equal': 0, 'less': 1, 'more': 2}


def set_names_to_array(data_array: List[List[Union[int, float]]], row_col_names: Union[List[str],
                       Tuple[str, ...]] = None) -> List[List[Union[None, str, int, float]]]:
    row_col_names = get_name(row_col_names)
    size = len(data_array[0])
    result_array: List[List[Union[None, str, int, float]]] = ([[None] + [next(row_col_names) for _ in range(size)]] +
                                                              [0] * size)
    for i in range(size):
        result_array[i + 1] = [result_array[0][i + 1]] + data_array[i]

    return result_array


def get_column_width(quantity: int, header_width: int = 5, table_width: int = 100) -> int:
    return (table_width - header_width) // (quantity - 1)


def get_html_table(data_array: List[List[Union[None, str, int, float]]], header_width: int = 5,
                   table_width: int = 100) -> str:
    str_result = '<table id="newickTable" class="w-100 my-6 table-danger bg-light">\n'
    column_width = get_column_width(len(data_array[0]), header_width, table_width)
    for row in data_array:
        str_result += '<tr>\n'
        row_index = data_array.index(row)
        row_name = row[0]
        for val in row:
            val_index = row.index(val)
            val = val if type(val) is float else val
            if row_index and val_index:
                str_result += (f'<td><input id="{row_name}{data_array[0][val_index]}" type="text" placeholder='
                               f'"Default input" class ="w-{table_width} form-control" label = "" value = "'
                               f'{val:.6f}" readonly></td>\n')
            else:
                str_result += (f'<th class ="w-{column_width if val_index > 0 else header_width} text-center" >'
                               f'{val if val else ""}</th>\n')
        str_result += '</tr>\n'
    str_result += '</table>'

    return str_result


def get_jukes_cantor_probabilities_amino_acids_matrix(branch_length: float, alphabet_size: int) -> np.ndarray:
    qmatrix = np.ones((alphabet_size, alphabet_size))
    np.fill_diagonal(qmatrix, 1 - alphabet_size)
    qmatrix = qmatrix * 1 / (alphabet_size - 1)

    return expm(qmatrix * branch_length)


def get_pij(qmatrix: np.ndarray, p_time: float, ij: Tuple[int, ...]) -> np.ndarray:
    gain = qmatrix[0][1]
    loss = qmatrix[1][0]
    gl = gain + loss
    if ij == (0, 0):
        return loss / gl + np.exp(-gl * p_time) * gain / gl
    elif ij == (0, 1):
        return gain / gl * (1 - np.exp(-gl * p_time))
    elif ij == (1, 0):
        return loss / gl * (1 - np.exp(-gl * p_time))
    elif ij == (1, 1):
        return gain / gl + np.exp(-gl * p_time) * loss / gl


def get_one_parameter_qmatrix(p0: Optional[float], p1: Optional[float]) -> np.ndarray:
    qmatrix = np.zeros((2, 2), dtype='float32')
    p1 = p1 if p1 else 1 - p0

    qmatrix[0, 0] = - 1 / (2 * (1 - p1))
    qmatrix[0, 1] = 1 / (2 * (1 - p1))
    qmatrix[1, 0] = 1 / (2 * p1)
    qmatrix[1, 1] = - 1 / (2 * p1)

    return qmatrix


def lq_to_qmatrix(lg: str) -> Tuple[np.ndarray, ...]:
    qmatrix = np.zeros((20, 20), dtype='float32')
    lg_list = lg.split('\n')
    amino_acids_frequencies = np.array(1)
    for i in enumerate(lg_list):
        if i[0] < 20:
            line_list = i[1].strip().split()
            for j in enumerate(line_list):
                qmatrix[i[0]][j[0]] = float(j[1])
        elif i[0] == 21:
            amino_acids_frequencies = np.array([float(j) for j in i[1].strip().split()], dtype='float32')
    qmatrix = qmatrix + np.tril(qmatrix, -1).T
    for i in list(range(20)):
        for j in list(range(20)):
            if i != j:
                qmatrix[i][j] = qmatrix[i][j] * amino_acids_frequencies[j]
        qmatrix[i][i] = - np.sum(qmatrix[i])
    divisor = np.sum(np.diag(qmatrix))
    qmatrix_nn = qmatrix.copy()
    for i in list(range(20)):
        for j in list(range(20)):
            qmatrix[i][j] = - qmatrix[i][j] / divisor
    replacement_frequencies = np.array([[0.0 if x == i else np.round((- qmatrix[i][x]) * 1 / qmatrix[i][i], 9)
                                        for x in range(20)] for i in range(20)])
    replacement_frequencies = np.array([[0.0 if x == i else replacement_frequencies[i][x] + (1 -
                                       sum(replacement_frequencies[i])) / 19 for x in range(20)] for i in range(20)])

    return qmatrix_nn, qmatrix, amino_acids_frequencies, replacement_frequencies


def get_min(data_array: np.ndarray) -> float:
    return np.min(data_array[np.where((data_array > 0))])


def join_array(data_array: List[np.ndarray], min_indexes: np.ndarray, divider: int, multiplier: int,
               axis: int = 0) -> np.ndarray:
    result_array = result_data = (data_array[min_indexes[0]] * multiplier + data_array[min_indexes[1]]) / divider

    for i in enumerate(data_array):
        if i[0] == 0 == min(min_indexes):
            result_array = result_data
        elif i[0] == max(min_indexes):
            continue
        elif i[0] == 0 != min(min_indexes):
            result_array = i[1]
        elif i[0] == min(min_indexes) != 0:
            result_array = np.hstack((result_array, result_data)) if axis else np.vstack((result_array, result_data))
        else:
            result_array = np.hstack((result_array, i[1])) if axis else np.vstack((result_array, i[1]))

    return result_array


def get_index(data_array: np.ndarray, value: float, comparison_type: int = 0) -> np.ndarray:
    if comparison_type == 1:
        result_array = np.int8(np.argwhere(data_array < value))
    elif comparison_type == 2:
        result_array = np.int8(np.argwhere(data_array > value))
    else:
        result_array = np.int8(np.argwhere(data_array == value))

    return result_array


def get_name(names: Union[List[str], Tuple[str, ...]] = None):
    """This method is for internal use only."""
    if names:
        for name in names:
            yield name
    else:
        names = 'abcdefghijklmnopqrstuvwxyz'
        len_names = len(names)
        i = 0
        j = 1
        k = 1
        while True:
            yield names[i:j]
            i += 1
            j += 1
            if j >= len_names:
                k += 1
                j = 1 * k
                i = 0


def counter(count: Optional[int] = None):
    """This method is for internal use only."""
    count = count if count else 0
    while True:
        yield 'nd' + str(count).rjust(4, '0')
        count += 1


def clustering(data_array: List[List[float]]) -> Tree:
    return __clustering(np.array(data_array, dtype='float32'))


def set_node_children(node: Node, nodes: Tuple[Node, Node], distances: Tuple[float, float]) -> None:
    for child in enumerate(nodes):
        node.add_child(child[1], distances[child[0]])


def __clustering(data_array: np.ndarray) -> Tree:
    node, node_a, min_index = None, None, None
    node_names, node_nums = get_name(), counter()
    while True:
        upper_triangle = np.triu(data_array, 1)
        min_value = get_min(upper_triangle)
        distance = min_value / 2
        min_indexes = get_index(upper_triangle, min_value, COMPARISON_TYPES['equal'])[0]

        key = get_index(min_indexes, min_index, COMPARISON_TYPES['equal']).size if min_index is not None else 0
        node_new = Node(next(node_nums))
        divider, multiplier = 2, 1

        if node is None or (node is not None and not key):
            set_node_children(node_new, (Node(next(node_names)), Node(next(node_names))), (distance,
                                                                                           distance))
            node_a = node
        elif node is not None and key and node_a is None:
            set_node_children(node_new, (node, Node(next(node_names))),
                              (distance - node.get_full_distance_to_leafs(), distance))
            divider, multiplier = 3, 2
        elif node is not None and key and node_a is not None:
            set_node_children(node_new, (node_a, node), (distance - node_a.get_full_distance_to_leafs(),
                                                         distance - node.get_full_distance_to_leafs()))
            node_a = None

        if data_array[0].size <= 2:
            tree = Tree(node_new)
            return tree

        min_index = np.min(min_indexes)
        min_index = min_index - (1 if min_index == data_array.size - 1 else 0)
        data_rows = join_array(np.vsplit(data_array, data_array.shape[0]), min_indexes, divider, multiplier, axis=0)
        data_array = join_array(np.hsplit(data_rows, data_array.shape[0]), min_indexes, divider, multiplier, axis=1)
        node = node_new
