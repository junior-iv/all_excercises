from math import exp, log
from typing import List, Union, Tuple, Optional, Dict, Set
from numpy import random as rnd
from time import time
from datetime import timedelta
from flask import url_for
from tree import Tree
from node import Node
from scipy.optimize import Bounds, minimize as mz

import numpy as np
import os
import random
import array_functions as af
from itertools import product

from main import RESULT_DATA_PATH as RESULT_DATA_PATH
from main import AMINO_ACIDS as AMINO_ACIDS
from main import DNA as DNA
from main import BINARY as BINARY

CHARACTERS = {1: DNA, 2: AMINO_ACIDS[0], 3: BINARY}


def minus_func_decorator(func):
    def wrapper(*args, **kwargs):
        return -func(*args, **kwargs)
    return wrapper


@minus_func_decorator
def func_for_ex7_task2(x: List[float]) -> float:
    return 5 - (x[0] - 2) ** 2


@minus_func_decorator
def __get_sequences_log_likelihood_for_optimization(branch_length: float, dna1: str, dna2: str) -> float:
    dna_len = len(dna1)
    different_char = dna_len * get_sequences_difference(dna1, dna2)
    same_char = dna_len - different_char
    p_same, p_change = get_jukes_cantor_probabilities(branch_length)

    return (same_char * log(p_same, 10)) + (different_char * log(p_change, 10)) + (dna_len * log(0.25, 10))


def func_for_ex7_task1(x: List[float]) -> float:
    return (x[0] - 3) ** 2 - 4


def __get_minimized(func, parameter_x, *args, **kwargs):
    if kwargs.get('bounds'):
        bounds = kwargs.pop('bounds')
        kwargs.update({'bounds': Bounds(bounds[0], bounds[1])})
    return mz(func, parameter_x, *args, **kwargs)


def __get_sequences_log_likelihood(branch_length: float, dna1: str, dna2: str) -> Tuple[List[float], ...]:
    dna_len = len(dna1)
    different_char = dna_len * get_sequences_difference(dna1, dna2)
    same_char = dna_len - different_char
    p_same, p_change = get_jukes_cantor_probabilities(branch_length)
    result = ([same_char * log(p_same, 10)], [different_char * log(p_change, 10)], [dna_len * log(0.25, 10)],
              [(same_char * log(p_same, 10)) + (different_char * log(p_change, 10)) + (dna_len * log(0.25, 10))])

    return result


def get_distance(dna1, dna2) -> float:
    return -3/4 * log(1 - 4/3 * get_sequences_difference(dna1, dna2))


def get_sequences_log_likelihood(branch_length: float, dna1: str, dna2: str, variant: int, limits_x:
                                 Optional[Tuple[float, float]] = None) -> Dict[str, Union[str, float, int]]:
    start_time = time()
    if variant == 1:
        log_likelihood = __get_sequences_log_likelihood(branch_length, dna1, dna2)
        result = {f'execution_time': convert_seconds(time() - start_time),
                  f'The_log_likelihood_of_same_characters_is': log_likelihood[0][0],
                  f'The_log_likelihood_of_different_characters_is': log_likelihood[1][0],
                  f'The_log_likelihood_of_characters_is': log_likelihood[2][0],
                  f'The_log_likelihood_of_these_sequences_is': log_likelihood[3][0]}
    else:
        print(limits_x)
        maximized_function_results = __get_minimized(__get_sequences_log_likelihood_for_optimization, branch_length,
                                                     (dna1, dna2), bounds=limits_x, method='Powell')
        result = {f'execution_time': convert_seconds(time() - start_time),
                  f'status_(max)': 'maximized',
                  f'The_function_obtains_its_maximum_in_X': float(maximized_function_results.x[0]),
                  f'The_value_of_the_function_in_this_maximum_is': -maximized_function_results.fun}

    return result


def get_maximized(parameter_x, limits_x: Optional[Tuple[float, float]] = None) -> Dict[str, Union[str, float, int]]:
    start_time = time()
    maximized_function_results = __get_minimized(func_for_ex7_task2, parameter_x, bounds=limits_x)
    maximized_function_results.x[0] = round(maximized_function_results.x[0], 7)
    dict_results = {f'<i>f(x)</i>': '5 - (x - 2)<sup>2</sup>',
                    f'The_function_obtains_its_maximun_in_X': maximized_function_results.x,
                    f'The_value_of_the_function_in_this_maximun_is': -maximized_function_results.fun}
    result = {'execution_time': convert_seconds(time() - start_time)}
    result.update(dict_results)

    return result


def get_minimized(parameter_x) -> Dict[str, Union[str, float, int]]:
    start_time = time()
    minimized_function_results = __get_minimized(func_for_ex7_task1, parameter_x)
    minimized_function_results.x[0] = round(minimized_function_results.x[0], 7)
    dict_results = {f'<i>f(x)</i>': '(x - 3)<sup>2</sup> - 4',
                    f'The_function_obtains_its_minimun_in_X': minimized_function_results.x,
                    f'The_value_of_the_function_in_this_minimun_is': minimized_function_results.fun}
    result = {'execution_time': convert_seconds(time() - start_time)}
    result.update(dict_results)

    return result


def convert_seconds(seconds: float) -> str:
    return str(timedelta(seconds=seconds))


def get_sequences_difference(sequence1: str, sequence2: str) -> float:
    return sum(a != b for a, b in zip(sequence1, sequence2)) / len(sequence1)


def get_random_sequence(sequence_length: Optional[Union[int, str]] = 1, exclusion_index: Optional[int] = None,
                        variant: int = 1) -> str:
    sequence = CHARACTERS[variant] if exclusion_index is None else (CHARACTERS[variant][:exclusion_index] +
                                                                    CHARACTERS[variant][:exclusion_index + 1])

    return ''.join(rnd.choice(sequence, int(sequence_length)))


def get_random_amino_acid_sequence(aa_length: Optional[Union[int, str]] = 1, aa_frequencies: Optional[np.ndarray] = None
                                   ) -> str:
    return ''.join(rnd.choice(AMINO_ACIDS[0], int(aa_length), True, aa_frequencies))


def get_jukes_cantor_probabilities(branch_length) -> Tuple[float, float]:
    return (1 / 4) + (3 / 4) * exp((-4 / 3) * branch_length), (1 / 4) - (1 / 4) * exp((-4 / 3) * branch_length)


def get_jukes_cantor_probabilities_amino_acids(branch_length) -> Tuple[float, float]:
    return 1/20 + 19 / 20 * exp(-20 / 19 * branch_length), 1 / 20 - 1 / 20 * exp(-20 / 19 * branch_length)


def simulate_sequence_jc(branch_length: float, sequence_length: Union[int, str, None] = 1,
                         variant: int = 1, sequence: str = '') -> Tuple[float, str, str]:
    sequence = sequence if sequence else get_random_sequence(sequence_length, None, variant)
    new_sequence = ''
    p_same, p_change = get_jukes_cantor_probabilities(branch_length)
    for i in sequence:
        if rnd.random() >= p_same:
            i = rnd.choice([x for x in CHARACTERS[variant] if x != i])
        new_sequence += i

    return get_sequences_difference(sequence, new_sequence), sequence, new_sequence


def simulate_dna_gillespie(branch_length: float, dna_length: Union[int, str, None] = 4) -> Tuple[float, str, str]:
    dna = ''
    new_dna = ''
    lambda_param = 1
    for _ in range(dna_length):
        current_time = 0
        i = j = get_random_sequence()
        while True:
            current_time += rnd.exponential(lambda_param)
            if current_time <= branch_length:
                j = rnd.choice([x for x in DNA if x != j])
            else:
                break
        dna += i
        new_dna += j

    return get_sequences_difference(dna, new_dna), dna, new_dna


def simulate_dna_gillespie_efficient(branch_length: float,
                                     dna_length: Union[int, str, None] = 4) -> Tuple[float, str, str]:
    dna = get_random_sequence(dna_length)
    new_dna = dna
    lambda_param = 1 / dna_length
    current_time = 0
    while True:
        current_time += rnd.exponential(lambda_param)
        if current_time <= branch_length:
            index = rnd.randint(0, dna_length)
            i = rnd.choice([x for x in DNA if x != new_dna[index]])
            new_dna = new_dna[:index] + i + new_dna[index + 1:]
        else:
            break

    return get_sequences_difference(dna, new_dna), dna, new_dna


def generate_sequences(repetition_count: int, branch_length: float, method: int = 1,
                       dna_length: Union[int, str, None] = 4) -> List[Tuple[float, str, str]]:
    """
    argument method
    1 - Jukes and Cantor simulation
    2 - Gillespie algorithm simulation
    3 - Gillespie algorithm simulation efficient
    """
    func = {1: simulate_sequence_jc, 2: simulate_dna_gillespie, 3: simulate_dna_gillespie_efficient}

    result_list = []
    for i in range(repetition_count):
        new_row = func[method](branch_length, dna_length)
        result_list.append(new_row)

    return result_list


def simulate_insertion_event(dna: str, low: int = 0, is_lower: bool = True) -> Tuple[str, int, int, int, int, int]:
    insertion_length = zipf(1.2)
    high = len(dna) + 1
    position = rnd.randint(low, high)
    insertion_start_position = position if position >= 0 else 0
    insertion_sequence = (
        get_random_sequence(insertion_length).lower() if is_lower else get_random_sequence(insertion_length))
    dna_result = f'{dna[:insertion_start_position]}{insertion_sequence}{dna[insertion_start_position:]}'

    return dna_result, insertion_start_position, insertion_start_position + insertion_length, low, high - 1, len(
        dna_result)


def simulate_deletion_event(dna: str, low: int = 0) -> Tuple[str, int, int, int, int, int]:
    deletion_length = zipf(1.2)
    high = len(dna)
    position = rnd.randint(low, high)
    deletion_start_position = position if position >= 0 else 0
    deletion_end_position = 0 if low < position + deletion_length < 0 else position + deletion_length
    dna_result = f'{dna[:deletion_start_position]}{dna[deletion_end_position:]}' if deletion_end_position >= 0 else dna

    return dna_result, deletion_start_position, deletion_end_position, low, high, len(dna_result)


def format_sequence(seq1: List[str], seq2: List[str], sep: str, issite: bool, seq_dict: Dict[str, str]) -> str:
    return sep.join([(seq_dict[i] if issite else i) if i in seq1 else '-' for i in seq2])


def simulate_indel_events(dna_length: int = 4, events_count: int = 1, low: int = 0
                          ) -> Dict[str, Union[str, float, int]]:
    super_sequence = current_sequence = start_sequence = list(map(str, range(dna_length)))
    dna = get_random_sequence(dna_length)
    seq_dict = {super_sequence[i]: dna[i] for i in range(dna_length)}
    counter = dna_length
    for _ in range(events_count):
        event_length = zipf(1.2)
        current_sequence_length = len(current_sequence)
        if rnd.choice([0, 1]):
            position = rnd.randint(current_sequence_length + 1)
            insertion = list(map(str, range(counter, counter + event_length)))
            dna = get_random_sequence(event_length)
            seq_dict.update({insertion[i]: dna[i] for i in range(event_length)})

            current_sequence = current_sequence[:position] + insertion + current_sequence[position:]
            super_position = super_sequence.index(current_sequence[position - 1]) if position else position
            super_sequence = super_sequence[:super_position] + insertion + super_sequence[super_position:]
            counter = counter + event_length
        else:
            position = rnd.randint(low, current_sequence_length)
            deletion_start_position = 0 if position < 0 else position
            deletion_end_position = 0 if low < position + event_length < 0 else position + event_length
            current_sequence = current_sequence[:deletion_start_position] + current_sequence[deletion_end_position:]

    return {'start_sequence': format_sequence(start_sequence, super_sequence, ' ', False, seq_dict),
            'end_sequence': format_sequence(current_sequence, super_sequence, ' ', False, seq_dict),
            'super_sequence': format_sequence(super_sequence, super_sequence, ' ', False, seq_dict),
            'super_sequence_(nucleotides)': format_sequence(super_sequence, super_sequence, '', True, seq_dict),
            'start_sequence_(nucleotides)': format_sequence(start_sequence, super_sequence, '', True, seq_dict),
            'end_sequence_(nucleotides)': format_sequence(current_sequence, super_sequence, '', True, seq_dict)}


def get_replacement(current_time: float, branch_length: float, qmatrix: np.ndarray, amino_acids_frequencies: np.ndarray,
                    amino_acid: str, aa_index: int) -> Tuple[float, int, str, str]:
    j = m = amino_acid
    lambda_param = 1 / -qmatrix[aa_index][aa_index]
    counter = 0
    while True:
        current_time += rnd.exponential(lambda_param)
        if current_time <= branch_length:
            j = get_random_amino_acid_sequence(1, amino_acids_frequencies, )
            counter += 1 if j != m else 0
        else:
            break

    return get_sequences_difference(m, j), counter, m, j


def get_replacement_statistic(replacement_statistic: List[Tuple[float, int, str, str]], name: str = ''
                              ) -> Dict[str, Union[str, float, int]]:
    name = f'_({name})' if name else ''
    replacement_mean = sum([x[1] for x in replacement_statistic]) / len(replacement_statistic)
    replacement_probabilities = sum([x[0] for x in replacement_statistic]) / len(replacement_statistic)
    no_change_probabilities = sum([1 - x[0] for x in replacement_statistic]) / len(replacement_statistic)

    return {f'replacement_mean{name}': f'{replacement_mean:.5f}',
            f'replacement_probabilities{name}': f'{replacement_probabilities:.5f}',
            f'no_change_probabilities{name}': f'{no_change_probabilities:.5f}'}


def __simulate_amino_acid_replacements_along_tree(newick_tree: Tree, probabilities: Tuple[np.ndarray, ...], aa_length:
                                                  int = 1, starting_amino_acid: Optional[str] = None) -> Tuple[Union[
                                                   str, float, int], ...]:
    final_sequence: str = ''
    qmatrix_nn, qmatrix, amino_acids_frequencies, replacement_frequencies = probabilities

    def get_replacements_along_tree(node: Node, amino_acid_sequence: Optional[str] = None) -> None:
        nonlocal final_sequence, starting_amino_acid
        if node.father:
            aa_index = AMINO_ACIDS[0].index(amino_acid_sequence)
            replacement = get_replacement(0, node.distance_to_father, qmatrix_nn, replacement_frequencies[aa_index],
                                          amino_acid_sequence, aa_index)
            amino_acid_sequence = replacement[3]
            if not node.children:
                final_sequence += amino_acid_sequence
        else:
            amino_acid_sequence = starting_amino_acid = amino_acid_sequence if amino_acid_sequence else (
                get_random_amino_acid_sequence(aa_length, amino_acids_frequencies))
        for child in node.children:
            get_replacements_along_tree(child, amino_acid_sequence)

    get_replacements_along_tree(newick_tree.root, starting_amino_acid)

    return (1 - get_sequences_difference(final_sequence, starting_amino_acid * len(final_sequence)), final_sequence,
            starting_amino_acid)


def __simulate_amino_acid_replacements_by_lg(probabilities: Tuple[np.ndarray, ...],
                                             branch_length: float, simulations_count: int = 100000, aa_length: int = 1,
                                             name: str = '', starting_amino_acid: Optional[Union[str, int]] =
                                             None) -> Dict[str, Union[str, float, int]]:
    qmatrix_nn, qmatrix, amino_acids_frequencies, replacement_frequencies = probabilities
    replacement_statistic = []
    amino_acid_sequence = ''

    for _ in range(simulations_count):
        if starting_amino_acid is None or (isinstance(starting_amino_acid, str) and not starting_amino_acid):
            amino_acid_sequence = get_random_amino_acid_sequence(aa_length, amino_acids_frequencies)
        elif isinstance(starting_amino_acid, str) and starting_amino_acid:
            amino_acid_sequence = AMINO_ACIDS[0][AMINO_ACIDS[1].index(starting_amino_acid)] * aa_length
        elif isinstance(starting_amino_acid, int):
            amino_acid_sequence = AMINO_ACIDS[0][starting_amino_acid] * aa_length
        for i in range(aa_length):
            aa_index = AMINO_ACIDS[0].index(amino_acid_sequence[i])
            replacement_statistic.append(get_replacement(0, branch_length, qmatrix_nn,
                                                         replacement_frequencies[aa_index], amino_acid_sequence[i],
                                                         aa_index))

    return get_replacement_statistic(replacement_statistic, name)


def calculate_pij(state_frequency: Tuple[Optional[float], ...], parameters_p: Tuple[float, ...]
                  ) -> Dict[str, Union[str, float, int]]:
    start_time = time()
    qmatrix = af.get_one_parameter_qmatrix(*state_frequency)
    pij = dict()
    for parameter in enumerate(parameters_p):
        ij = (parameter[0] // 2, parameter[0] % 2)
        pij.update({f'P<sub>{"".join(map(str, ij))}</sub>(<span class="text-success">{parameter[0]}</span>)':
                    af.get_pij(qmatrix, parameter[1], ij)})
    result = {'execution_time': convert_seconds(time() - start_time)}
    result.update(**pij)

    return result


def simulate_sites_along_branch_with_one_parameter_matrix(branch_length: float, state_frequency: Tuple[Optional[float],
                                                          ...], aa_length: int, simulations_count: int = 10000
                                                          ) -> Dict[str, Union[str, float, int]]:
    start_time = time()
    qmatrix = af.get_one_parameter_qmatrix(*state_frequency)
    difference = 0

    for _ in range(simulations_count):
        aa = ''
        new_aa = ''
        lambda_param = 1
        for _ in range(aa_length):
            current_time = 0
            i = j = '0'
            while True:
                current_time += rnd.exponential(lambda_param / -qmatrix[int(j)][int(j)])
                if current_time <= branch_length:
                    j = BINARY[int(j) - 1]
                else:
                    break
            aa += i
            new_aa += j
        difference += round(get_sequences_difference(aa, new_aa) / simulations_count, 12)

    return {'execution_time': convert_seconds(time() - start_time), 'different': difference, 'same': 1 - difference}


def change_amino_acid(sequence: Union[str, List[str]], sep: str = ' ') -> Union[str, List[str]]:
    final_sequence = [AMINO_ACIDS[1][AMINO_ACIDS[0].index(i)] for i in sequence]

    return sep.join(final_sequence) if isinstance(sequence, str) else final_sequence


def simulate_amino_acid_replacements_along_tree(probabilities: Tuple[np.ndarray, ...], newick_text: str,
                                                simulations_count: int = 100000, aa_length: int = 10
                                                ) -> Dict[str, Union[str, float, int]]:
    start_time = time()
    newick_tree = Tree(newick_text)
    simulations_result = []
    for i in range(simulations_count):
        simulations_result.append(__simulate_amino_acid_replacements_along_tree(newick_tree, probabilities, aa_length))

    return {'execution_time': convert_seconds(time() - start_time), 'the_expected_number_of_constant_site':
            sum([x[0] for x in simulations_result]) / len(simulations_result), 'EXAMPLES': ' ->  ->  -> ',
            '1)_starting amino acid(-1)': f'{change_amino_acid(simulations_result[-1][2])}',
            'final_sequence(-1)': f'{change_amino_acid(simulations_result[-1][1])};',
            '2)_starting amino acid(-2)': f'{change_amino_acid(simulations_result[-2][2])}',
            'final_sequence(-2)': f'{change_amino_acid(simulations_result[-2][1])};',
            '3)_starting amino acid(-3)': f'{change_amino_acid(simulations_result[-3][2])}',
            'final_sequence(-3)': f'{change_amino_acid(simulations_result[-3][1])};',
            '4)_starting amino acid(-4)': f'{change_amino_acid(simulations_result[-4][2])}',
            'final_sequence(-4)': f'{change_amino_acid(simulations_result[-4][1])};',
            '5)_starting amino acid(-5)': f'{change_amino_acid(simulations_result[-5][2])}',
            'final_sequence(-5)': f'{change_amino_acid(simulations_result[-5][1])};'}


def __simulate_with_binary_jc(newick_text: str, sequence_length: int = 1) -> str:
    tree = Tree(newick_text)
    sequence = ''
    char = ''

    def get_sequence(tree_node: Node) -> None:
        nonlocal sequence, char
        if tree_node.father:
            char = simulate_sequence_jc(tree_node.distance_to_father, 1, 3, char)[2]
        else:
            char = get_random_sequence(sequence_length, None, 3)

        if tree_node.children:
            for child in tree_node.children:
                get_sequence(child)
        else:
            sequence += char

    get_sequence(tree.root)
    return sequence


def simulate_with_binary_jc(newick_text: str, variant: int = 1, final_sequence: Optional[str] = None,
                            simulations_count: Optional[int] = None, sequence_length: int = 1
                            ) -> Dict[str, Union[str, int, float]]:
    start_time = time()
    simulations_result = dict()
    if variant == 1:
        sequence = __simulate_with_binary_jc(newick_text, sequence_length)
        simulations_result.update({'final_sequence': sequence})
    else:
        if not simulations_count:
            simulations_result.update({'simulations_count': 'number of simulations was entered incorrectly'})
        if not final_sequence:
            simulations_result.update({'final_sequence': 'final sequence was entered incorrectly'})
        if final_sequence and simulations_count and variant == 2:
            count = 0
            for _ in range(simulations_count):
                sequence = __simulate_with_binary_jc(newick_text, sequence_length)
                count += 1 if sequence == final_sequence else 0
            likelihood = count / simulations_count
            simulations_result.update({'sequence_of_interest': final_sequence})
            simulations_result.update({'simulations_count': simulations_count})
            simulations_result.update({'number_of_matches': count})
            simulations_result.update({'likelihood_of_the_data': likelihood})

    result = {'execution_time': convert_seconds(time() - start_time)}
    result.update(simulations_result)
    return result


def find_dict_in_iterable(iterable: Union[List[Dict[str, Union[float, bool, str, List[float], Tuple[int, ...]]]], Tuple[
                         Dict[str, Union[float, bool, str, List[float], Tuple[int, ...]]]]], key: str, value: Optional[
                         Union[float, bool, str, List[float]]] = None) -> Dict[str, Union[float, bool, str, List[float],
                                                                               List[int], Tuple[int, ...]]]:
    for index, dictionary in enumerate(iterable):
        if key in dictionary and (True if value is None else dictionary[key] == value):
            return dictionary


def get_nodes_starting_values(list_nodes_info: List[Dict[str, Union[float, bool, str, List[float]]]], characters:
                              List[str], answer: str) -> List[Dict[str, Tuple[int, ...]]]:
    list_result = []
    for number, i in enumerate(list(answer)):
        states = []
        for j in characters:
            states.append(int(i == j))
        list_result.append({list_nodes_info[number].get('node'): tuple(states)})

    return list_result


def get_states(node_states: List[Dict[str, Tuple[int, ...]]], node_starting_values: Dict[str, Union[float, bool, str,
               list[float], list[int], Tuple[int, ...], Tuple[float, ...]]], node_info: Dict[str,
               Union[float, bool, str, List[float], Tuple[int, ...]]]) -> Tuple[int, ...]:
    starting_value = node_starting_values.get(node_info.get('node'))
    father_name = node_info.get('father_name')
    father_starting_value = find_dict_in_iterable(node_states, father_name).get(father_name)
    char_1 = [BINARY.index(BINARY[i]) for i, x in enumerate(father_starting_value) if float(x) > 0][0]
    char_2 = [BINARY.index(BINARY[i]) for i, x in enumerate(starting_value) if float(x) > 0][0]

    return char_1, char_2


def get_node_likelihood(nodes_starting_values: List[Dict[str, Tuple[int, ...]]], node_starting_values: Dict[str,
                        Tuple[int, ...]], node_info: Dict[str, Union[float, bool, str, List[float]]], qmatrix:
                        np.ndarray, repeat: Union[float, int], repeat_f1: str, repeat_f2: str) -> Tuple[
                        Union[float, int], str, str]:
    states = get_states(nodes_starting_values, node_starting_values, node_info)
    pij = af.get_pij(qmatrix, node_info.get('distance'), states)
    repeat *= pij
    repeat_f1 = f'{repeat_f1} * {pij:.4f}'
    repeat_f2 = f'{repeat_f2} P<sub>{states[0]}{states[1]}</sub>({node_info.get("distance"):.3f})'

    return repeat, repeat_f1, repeat_f2


def __compute_likelihood_with_binary_jc(newick_text: str, final_sequence: Optional[str] = None) -> Tuple[float, str,
                                                                                                         str]:
    newick_tree = Tree(newick_text)
    nodes_info = newick_tree.get_list_nodes_info(False, True, 'pre-order', {'node_type': ['node', 'root']})
    alphabet_size = len(BINARY)
    cartesian_product = list(product(list(range(alphabet_size)), repeat=len(nodes_info)))

    leaves_info = newick_tree.get_list_nodes_info(False, True, 'pre-order', {'node_type': ['leaf']})
    leaves_vectors = get_nodes_starting_values(leaves_info, BINARY, final_sequence)

    frequency = 1 / alphabet_size
    state_frequency = (frequency, None)
    qmatrix = af.get_one_parameter_qmatrix(*state_frequency)
    likelihood, formula, solution = 0, f'', f''
    for iterable in cartesian_product:
        repeat, repeat_f1, repeat_f2 = frequency, f'{frequency:.3f}', f'P({iterable[0]})'
        answer = ''.join([BINARY[x] for x in iterable])
        nodes_starting_values = get_nodes_starting_values(nodes_info, BINARY, answer)
        for number, node_starting_values in enumerate(nodes_starting_values):
            node_info = nodes_info[number]
            if find_dict_in_iterable(nodes_info, 'node', node_info.get('father_name')):
                repeat, repeat_f1, repeat_f2 = get_node_likelihood(nodes_starting_values, node_starting_values,
                                                                   node_info, qmatrix, repeat, repeat_f1, repeat_f2)

        for number, leaf_starting_values in enumerate(leaves_vectors):
            repeat, repeat_f1, repeat_f2 = get_node_likelihood(nodes_starting_values, leaf_starting_values,
                                                               leaves_info[number], qmatrix, repeat, repeat_f1,
                                                               repeat_f2)

        likelihood += repeat
        solution = f'{solution}<br>&emsp;&emsp;&emsp;&emsp;&nbsp; + {repeat_f1}' if solution else f'{repeat_f1}'
        formula = f'{formula}<br>&emsp;&emsp;&emsp;&emsp;&nbsp; + {repeat_f2}' if formula else f'{repeat_f2}'

    return likelihood, formula, solution


def compute_likelihood_with_binary_jc(newick_text: str, final_sequence: Optional[str] = None) -> Dict[str, Union[str,
                                                                                                      float, int]]:
    start_time = time()
    simulations_result = dict()
    likelihood = __compute_likelihood_with_binary_jc(newick_text, final_sequence)
    simulations_result.update({'formula': likelihood[1]})
    simulations_result.update({'solution': likelihood[2]})
    simulations_result.update({'likelihood_of_the_data': likelihood[0]})

    result = {'execution_time': convert_seconds(time() - start_time)}
    result.update(simulations_result)

    return result


def get_ancestors_of_leaves_only(newick_tree: Tree, exception_set: Optional[Set] = None) -> (Tuple[List[Dict[str, Union[
                                 float, bool, str, List[float]]]], Set[str]]):
    ancestors_list, ancestors_set = [], set()
    exception_set = set() if exception_set is None else exception_set
    leaves_info = newick_tree.get_list_nodes_info(False, True, 'pre-order', {'node_type': ['leaf']})
    nodes_info = newick_tree.get_list_nodes_info(False, True, 'pre-order', {'node_type': ['node']})
    for dict_node in nodes_info:
        exception_set.add(dict_node.get('father_name'))

    for dict_leaf in leaves_info:
        father_name = dict_leaf.get('father_name')
        if (sum([True for i in leaves_info if i.get('father_name') == father_name]) > 1
                and father_name not in exception_set and father_name not in ancestors_set):
            ancestors_set.add(father_name)
            ancestors_list.append(newick_tree.get_list_nodes_info(False, True, None, {'node': [father_name]})[0])
    return ancestors_list, ancestors_set


def calculate_felsensteins_likelihood_for_amino_acids(newick_node: Node, leaves_dict: Dict[str, Tuple[int, ...]],
                                                      alphabet: Tuple[str, ...]) -> Union[Tuple[Union[Tuple[np.ndarray,
                                                                                          ...], Tuple[float, ...]],
                                                                                          float], float]:
    alphabet_size = len(alphabet)
    if not newick_node.children:
        return leaves_dict.get(newick_node.name), newick_node.distance_to_father

    l_vect, l_dist = calculate_felsensteins_likelihood_for_amino_acids(newick_node.children[0], leaves_dict, alphabet)
    r_vect, r_dist = calculate_felsensteins_likelihood_for_amino_acids(newick_node.children[1], leaves_dict, alphabet)

    l_qmatrix = af.get_jukes_cantor_probabilities_amino_acids_matrix(l_dist, alphabet_size)
    r_qmatrix = af.get_jukes_cantor_probabilities_amino_acids_matrix(r_dist, alphabet_size)

    vector = []
    for j in range(alphabet_size):
        freq_l = freq_r = 0
        for i in range(alphabet_size):
            freq_l += l_qmatrix[i, j] * l_vect[i]
            freq_r += r_qmatrix[i, j] * r_vect[i]
        vector.append(freq_l * freq_r)
    vector = tuple(vector)

    if newick_node.father:
        return vector, newick_node.distance_to_father
    else:
        return np.sum([1 / alphabet_size * i for i in vector])


def calculate_felsensteins_likelihood(newick_node: Node, matrix: np.ndarray, leaves_dict: Dict[str, Tuple[int, int]]
                                      ) -> Union[Tuple[Union[Tuple[np.ndarray, ...], Tuple[float, ...]], float],
                                                 np.ndarray]:
    if not newick_node.children:
        return leaves_dict.get(newick_node.name), newick_node.distance_to_father

    l_vect, l_dist = calculate_felsensteins_likelihood(newick_node.children[0], matrix, leaves_dict)
    r_vect, r_dist = calculate_felsensteins_likelihood(newick_node.children[1], matrix, leaves_dict)

    freq_0 = ((af.get_pij(matrix, l_dist, (0, 0)) * l_vect[0] + af.get_pij(matrix, l_dist, (0, 1)) * l_vect[1]) *
              (af.get_pij(matrix, r_dist, (0, 0)) * r_vect[0] + af.get_pij(matrix, r_dist, (0, 1)) * r_vect[1]))
    freq_1 = ((af.get_pij(matrix, l_dist, (1, 0)) * l_vect[0] + af.get_pij(matrix, l_dist, (1, 1)) * l_vect[1]) *
              (af.get_pij(matrix, r_dist, (1, 0)) * r_vect[0] + af.get_pij(matrix, r_dist, (1, 1)) * r_vect[1]))

    if newick_node.father:
        return (freq_0, freq_1), newick_node.distance_to_father
    else:
        return 1 / len(BINARY) * freq_0 + 1 / len(BINARY) * freq_1


def __compute_felsensteins_likelihood_with_binary_jc(newick_text: str, final_sequence: Optional[str] = None
                                                     ) -> np.ndarray:
    newick_tree = Tree(newick_text)
    alphabet_size = len(BINARY)

    leaves_info = newick_tree.get_list_nodes_info(False, True, 'pre-order', {'node_type': ['leaf']})

    leaves_dict = dict()
    for i in range(len(leaves_info)):
        leaves_dict.update({leaves_info[i].get('node'): (1 - BINARY.index(final_sequence[i]), 0 +
                                                         BINARY.index(final_sequence[i]))})

    frequency = 1 / alphabet_size
    state_frequency = (frequency, None)
    qmatrix = af.get_one_parameter_qmatrix(*state_frequency)
    likelihood = calculate_felsensteins_likelihood(newick_tree.root, qmatrix, leaves_dict)

    return likelihood


def get_amino_acid_replacement_frequencies(amino_acid1: str, amino_acid2: str, replacement_frequencies):
    return replacement_frequencies[AMINO_ACIDS[0].index(amino_acid1)][AMINO_ACIDS[0].index(amino_acid2)]


def __compute_amino_acids_likelihood(newick_text: str, final_sequence: Optional[str] = None, alphabet_number: int = 2
                                     ) -> Tuple[List[float], float, float]:
    alphabet = CHARACTERS[alphabet_number]
    # qmatrix_nn, qmatrix, amino_acids_frequencies, replacement_frequencies = probabilities
    newick_tree = Tree(newick_text)
    alphabet_size = len(alphabet)

    leaves_info = newick_tree.get_list_nodes_info(False, True, 'pre-order', {'node_type': ['leaf']})

    sequence_list = final_sequence.split()
    sequence_dict = dict()
    for j in range(len(sequence_list) // 2):
        sequence_dict.update({sequence_list[j + j][1::]: sequence_list[j + j - 1]})

    len_seq = len(list(sequence_dict.values())[0])
    likelihood = 1
    log_likelihood = 0
    log_likelihood_list = []
    for i_char in range(len_seq):
        # leaves_dict = dict()
        # for i in range(len(leaves_info)):
        #     node_name = leaves_info[i].get('node')
        #     sequence = sequence_dict.get(node_name)[i_char]
        #     leaves_dict.update({leaves_info[i].get('node'): (1 - BINARY.index(sequence), 0 +
        #                                                      BINARY.index(sequence))})
        # qmatrix = af.get_one_parameter_qmatrix(0.5, None)
        # char_likelihood = calculate_felsensteins_likelihood(newick_tree.root, qmatrix, leaves_dict)
        leaves_dict = dict()
        for i in range(len(leaves_info)):
            node_name = leaves_info[i].get('node')
            sequence = sequence_dict.get(node_name)[i_char]
            frequency = [0] * alphabet_size
            frequency[alphabet.index(sequence)] = 1
            leaves_dict.update({node_name: tuple(frequency)})
        char_likelihood = calculate_felsensteins_likelihood_for_amino_acids(newick_tree.root, leaves_dict, alphabet)
        likelihood *= char_likelihood
        log_likelihood += log(char_likelihood)
        log_likelihood_list.append(log(char_likelihood))

    return log_likelihood_list, log_likelihood, likelihood


def compute_amino_acids_likelihood(newick_text: str, final_sequence: Optional[str] = None, alphabet_number: int = 1
                                   ) -> Dict[str, Union[str, float, int]]:
    start_time = time()
    log_likelihood_list, log_likelihood, likelihood = __compute_amino_acids_likelihood(newick_text, final_sequence,
                                                                                       alphabet_number)

    result = {'execution_time': convert_seconds(time() - start_time)}
    result.update({'likelihood_of_the_tree': likelihood})
    result.update({'log_likelihood_of_the_tree': log_likelihood})
    result.update({'log_likelihood_list': log_likelihood_list})

    return result


def compute_felsensteins_likelihood_with_binary_jc(newick_text: str, final_sequence: Optional[str] = None) -> Dict[
                                                   str, Union[str, float, int]]:
    start_time = time()
    likelihood = __compute_felsensteins_likelihood_with_binary_jc(newick_text, final_sequence)

    result = {'execution_time': convert_seconds(time() - start_time)}
    result.update({'likelihood_of_the_data': likelihood})

    return result


def simulate_amino_acid_replacements_by_lg(probabilities: Tuple[np.ndarray, ...], branch_length: float,
                                           simulations_count: int = 100000, aa_length: int = 10,
                                           starting_amino_acid: Optional[Union[str, int]] = None
                                           ) -> Dict[str, Union[str, float, int]]:
    start_time = time()
    simulations_result = __simulate_amino_acid_replacements_by_lg(probabilities, branch_length, simulations_count,
                                                                  aa_length, '', starting_amino_acid)
    result = {'execution_time': convert_seconds(time() - start_time)}
    result.update(simulations_result)

    return result


def simulate_pairwise_alignment(dna_length: int = 4, events_count: int = 1, low: int = 0
                                ) -> Dict[str, Union[str, float, int]]:
    start_time = time()
    events_result = simulate_indel_events(dna_length, events_count, low)
    result = {'execution_time': convert_seconds(time() - start_time)}
    result.update(events_result)

    return result


def calculate_change_dna_length_statistics(repetition_count: int, low: int, dna_length: int = 4,
                                           events_count: int = 4) -> Dict[str, Union[str, float, int]]:
    start_time = time()
    event_list = []
    event_result = ('', 0, 0, 0, 0, 0)
    for _ in range(repetition_count):
        dna = get_random_sequence(dna_length)
        for i in range(events_count):
            event_result = simulate_deletion_event(dna, low) if rnd.choice([0, 1]) else simulate_insertion_event(
                                                                                        dna, 0, False)
            dna = event_result[0]
        event_list.append(event_result[5])
    event_array = np.array(event_list)
    mean = np.mean(event_array)
    std = np.std(event_array)

    return {'execution_time': convert_seconds(time() - start_time),
            'average_deviation ': mean,
            'standard_deviation': std}


def calculate_event_simulation_statistics(repetition_count: int, low: int, dna_length: Union[int, str, None] = 4,
                                          method: int = 1) -> Dict[str, Union[str, float, int]]:
    start_time = time()
    dna_length = int(dna_length)
    if method == 1:
        sequences_list = [simulate_insertion_event(get_random_sequence(dna_length), 0, False)
                          for _ in range(repetition_count)]
        event_name = 'inserted'
    else:
        sequences_list = [simulate_deletion_event(get_random_sequence(dna_length), low) for _ in
                          range(repetition_count)]
        event_name = 'deleted'

    first_position_quantity = sum(map(lambda x: 1 if x[1] <= 0 < x[2] else 0, sequences_list))
    middle_position_quantity = sum(map(lambda x: 1 if x[1] <= dna_length / 2 - 1 < x[2] else 0, sequences_list))
    last_position_quantity = sum(map(lambda x: 1 if x[1] <= dna_length - 1 < x[2] else 0, sequences_list))

    return {'execution_time': convert_seconds(time() - start_time),
            f'number_of_times_first_position_was_{event_name}': first_position_quantity,
            f'number_of_times_middle_position_was_{event_name}': middle_position_quantity,
            f'number_of_times_last_position_was_{event_name}': last_position_quantity}


def change_dna_length(method: int = 1, dna_length: Union[int, str, None] = 4):
    start_time = time()
    first_dna_sequence = get_random_sequence(int(dna_length)) if dna_length.isnumeric() else dna_length
    if method == 1:
        event_result = simulate_insertion_event(first_dna_sequence, 0)
        event = 'insertion'
    else:
        event_result = simulate_deletion_event(first_dna_sequence, -49)
        event = 'deletion'

    return {'execution_time': convert_seconds(time() - start_time),
            'first_dna_sequence': first_dna_sequence,
            'last_dna_sequence': event_result[0],
            'start_position': event_result[1],
            'end_position': event_result[2],
            f'length_of_{event}': event_result[2] - event_result[1],
            'minimum_index_for_the_event': event_result[3],
            f'length_of_the_sequence_before_{event}': event_result[4],
            f'length_of_the_sequence_after_{event}': event_result[5]}


def calculate_simulations_statistics(repetition_count: int, branch_length: float, method: int = 1,
                                     dna_length: Union[int, str, None] = 4) -> Dict[str, Union[str, float, int]]:
    start_time = time()
    sequences_list = generate_sequences(repetition_count, branch_length, method, dna_length)
    average_distance = f'{sum([x[0] for x in sequences_list]) / repetition_count:.3f}'
    execution_time = convert_seconds(time() - start_time)
    url = url_for("result_table",
                  file_name=get_html_table_file_name(sequences_list, ("number", "count", "probability")))

    return {'execution_time': execution_time,
            'including_file_generation': convert_seconds(time() - start_time),
            'average_distance': average_distance,
            'first_dna_sequence': sequences_list[0][1],
            'last_dna_sequence': sequences_list[0][2],
            'url': f'<a href=\"{url}\">{url}</a>'}


def zipf(alpha: float) -> int:
    while True:
        n = rnd.zipf(alpha)
        if 0 < n <= 50:
            return n


def get_zipf(alpha: float, simulations_count: int = 100000,
             result_rows_number: int = 5) -> Dict[str, Union[str, float, int, List[Dict[str, Union[str, float, int]]]]]:
    start_time = time()
    probabilities_list = [{'number': i, 'count': 0} for i in range(1, 51)]
    for i in range(simulations_count):
        n = zipf(alpha)
        probabilities_list[n - 1]['count'] += 1
    probabilities_list = [(i['number'], i['count'], i['count'] / simulations_count) for i in probabilities_list]
    execution_time = convert_seconds(time() - start_time)
    url = url_for("result_table",
                  file_name=get_html_table_file_name(probabilities_list, ("#", "dna / new dna", "%"), 2))
    probabilities_result = [{'number': f'{i[1][0]}', 'probability': f'{i[1][2]:.5f}'} for i in
                            enumerate(probabilities_list) if i[0] <= result_rows_number - 1]
    probabilities_result.append({'url': f'<a href=\"{url}\">{url}</a>'})

    return {'execution_time': execution_time, 'including_file_generation': convert_seconds(time() - start_time),
            'result': probabilities_result}


def get_html_table_head(head: Tuple[str, str, str]) -> str:
    str_result = '<table id="sequencesResultTable" class="m-3 w-75 table-bordered table-danger bg-light">\n'
    style = "'text-center bg-dark bg-opacity-10'"
    str_result += f'<tr>{"".join([f"<th class={style}>{i}</th>" for i in head])}</tr>\n'

    return str_result


def get_html_table(data_array: List[Tuple[Union[float, int, str], ...]], str_table: str,
                   variant: int = 1) -> str:
    if variant == 1:
        for row in enumerate(data_array):
            str_table += (f'<tr><td rowspan="2" class="w-10 text-center bg-opacity-25">{row[0] + 1}</td>'
                          f'<td class="w-80 text-start bg-info bg-opacity-25">{row[1][1]}</td>\n'
                          f'<td rowspan="2" class="w-10 m-3 text-end bg-opacity-25">{row[1][0]}</td></tr>\n'
                          f'<tr><td class="w-80 m-3 text-start bg-primary bg-opacity-25">{row[1][2]}</td></tr>\n')
    elif variant == 2:
        for row in data_array:
            str_table += (f'<tr><td class="w-35 m-3 text-center bg-opacity-25">{row[0]}</td>\n'
                          f'<td class="w-35 m-3 text-start bg-info bg-opacity-25">{row[1]}</td>\n'
                          f'<td class="w-30 m-3 text-start bg-primary bg-opacity-25">{row[2]}</td></tr>\n')
    str_table += '</table>\n'

    return str_table


def get_html_table_file_name(data_array: List[Tuple[Union[float, int, str], ...]],
                             head: Tuple[str, str, str], variant: int = 1) -> str:
    if not os.path.exists(RESULT_DATA_PATH):
        os.makedirs(RESULT_DATA_PATH)

    list_dir = os.listdir(RESULT_DATA_PATH)
    while True:
        name = f'{random.randrange(1000000 * variant, 1000000 * variant + 999999)}.txt'
        if name not in list_dir:
            break

    file_name = f'{RESULT_DATA_PATH}/{name}'
    str_table = get_html_table(data_array, get_html_table_head(head), variant)
    with open(file_name, 'w') as f:
        f.write(str_table)

    return name
