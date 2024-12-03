from math import exp, log
from typing import List, Union, Tuple, Optional, Dict
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

from main import RESULT_DATA_PATH as RESULT_DATA_PATH
from main import AMINO_ACIDS as AMINO_ACIDS
from main import DNA as DNA
from main import BINARY as BINARY

CHARACTERS = {1: DNA, 2: AMINO_ACIDS[0], 3: BINARY}


def func_for_ex7_task1(x: List[float]) -> float:
    return (x[0] - 3) ** 2 - 4


def func_for_ex7_task2(x: List[float]) -> float:
    return 5 - (x[0] - 2) ** 2


def get_minus_func_for_ex7_task3(x):
    return -func_for_ex7_task2(x)


def get_minus_func(x, *args, **kwargs):
    return -__get_sequences_log_likelihood_for_optimization(x, *args, **kwargs)


def __get_minimized(parameter_x, func, *args, **kwargs):
    if kwargs.get('bounds'):
        bounds = kwargs.pop('bounds')
        kwargs.update({'bounds': Bounds(bounds[0], bounds[1])})
    return mz(func, parameter_x, *args, **kwargs)


def __get_sequences_log_likelihood_for_optimization(branch_length: float, dna1: str, dna2: str) -> float:
    dna_len = len(dna1)
    different_char = dna_len * get_sequences_difference(dna1, dna2)
    same_char = dna_len - different_char
    p_same, p_change = get_jukes_cantor_probabilities(branch_length)

    return (same_char * log(p_same, 10)) + (different_char * log(p_change, 10)) + (dna_len * log(0.25, 10))


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


def get_sequences_log_likelihood(branch_length: float, dna1: str, dna2: str, variant: int
                                 ) -> Dict[str, Union[str, float, int]]:
    start_time = time()
    if variant == 1:
        log_likelihood = __get_sequences_log_likelihood(branch_length, dna1, dna2)
        result = {f'execution_time': convert_seconds(time() - start_time),
                  f'The_log_likelihood_of_same_characters_is': log_likelihood[0][0],
                  f'The_log_likelihood_of_different_characters_is': log_likelihood[1][0],
                  f'The_log_likelihood_of_characters_is': log_likelihood[2][0],
                  f'The_log_likelihood_of_these_sequences_is': log_likelihood[3][0]}
    else:
        maximized_function_results = __get_minimized([branch_length], get_minus_func,
                                                        (dna1, dna2), bounds=(0.01, 10), method='Powell')
        minimized_function_results = __get_minimized([branch_length], __get_sequences_log_likelihood_for_optimization,
                                                        (dna1, dna2), bounds=(0.01, 100), method='Powell')
        result = {f'execution_time': convert_seconds(time() - start_time),
                  f'status_(max)': 'maximized',
                  f'x_(max)': float(maximized_function_results.x),
                  f'fun_(max)': float(maximized_function_results.fun),
                  f'nit_(max)': int(maximized_function_results.nit),
                  f'nfev_(max)': int(maximized_function_results.nfev),
                  f'message_(max)': str(maximized_function_results.message),
                  f'status_(min)': 'minimized',
                  f'x_(min)': float(minimized_function_results.x),
                  f'fun_(min)': float(minimized_function_results.fun),
                  f'nit_(min)': int(minimized_function_results.nit),
                  f'nfev_(min)': int(minimized_function_results.nfev),
                  f'message_(min)': str(minimized_function_results.message),
                  f'<i>f(x)</i>': get_distance(dna1, dna2)}

    return result


def get_maximized(parameter_x, limits_x: Optional[Tuple[float, float]] = None) -> Dict[str, Union[str, float, int]]:
    start_time = time()
    maximized_function_results = __get_minimized(parameter_x, get_minus_func_for_ex7_task3, bounds=limits_x)
    maximized_function_results.x[0] = round(maximized_function_results.x[0], 7)
    dict_results = {f'<i>f(x)</i>': '5 - (x - 2)<sup>2</sup>',
                    f'The_function_obtains_its_maximun_in_X': maximized_function_results.x,
                    f'The_value_of_the_function_in_this_maximun_is': maximized_function_results.fun}
    result = {'execution_time': convert_seconds(time() - start_time)}
    result.update(dict_results)

    return result


def get_minimized(parameter_x) -> Dict[str, Union[str, float, int]]:
    start_time = time()
    minimized_function_results = __get_minimized(parameter_x, func_for_ex7_task1)
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


def get_random_sequence(sequence_length: Union[int, str, None] = 1, exclusion_index: Optional[int] = None,
                        variant: int = 1) -> str:
    sequence = CHARACTERS[variant] if exclusion_index is None else (CHARACTERS[variant][:exclusion_index] +
                                                                    CHARACTERS[variant][:exclusion_index + 1])

    return ''.join(rnd.choice(sequence, int(sequence_length)))


def get_random_amino_acid_sequence(aa_length: Union[int, str, None] = 1, aa_frequencies: Optional[np.ndarray] = None) -> str:
    return ''.join(rnd.choice(AMINO_ACIDS[0], int(aa_length), True, aa_frequencies))


def get_jukes_cantor_probabilities(branch_length) -> Tuple[float, float]:
    return (1 / 4) + (3 / 4) * exp((-4 / 3) * branch_length), (1 / 4) - (1 / 4) * exp((-4 / 3) * branch_length)


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
    # print(branch_length)
    # print(-qmatrix[aa_index][aa_index])
    lambda_param = 1 / -qmatrix[aa_index][aa_index]
    # print(lambda_param)
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


def __simulate_amino_acid_replacements_along_tree(newick_tree: Tree, probabilities: Tuple[np.ndarray, ...],
                                                  aa_length: int = 1, starting_amino_acid: Optional[str] = None,
                                                  name: str = '') -> Tuple[Union[str, float, int], ...]:
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
                get_random_amino_acid_sequence(aa_length,amino_acids_frequencies))
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
            replacement_statistic.append(get_replacement(0, branch_length, qmatrix_nn, replacement_frequencies[aa_index],
                                                         amino_acid_sequence[i], aa_index))

    return get_replacement_statistic(replacement_statistic, name)


def calculate_pij(gl_coefficient: Tuple[Optional[float], None], parameters_p: Tuple[float, ...]
                  ) -> Dict[str, Union[str, float, int]]:
    start_time = time()
    qmatrix = af.get_one_parameter_qmatrix(*gl_coefficient)
    pij = dict()
    for parameter in enumerate(parameters_p):
        ij = (parameter[0] // 2, parameter[0] % 2)
        pij.update({f'P<sub>{"".join(map(str, ij))}</sub>(<span class="text-success">{parameter[0]}</span>)':
                    af.get_pij(qmatrix, parameter[1], ij)})
    result = {'execution_time': convert_seconds(time() - start_time)}
    result.update(pij)

    return result


def simulate_sites_along_branch_with_one_parameter_matrix(branch_length: float, gl_coefficient: Tuple[Optional[float],
                                                          None], aa_length: int, simulations_count: int = 10000
                                                          ) -> Dict[str, Union[str, float, int]]:
    start_time = time()
    qmatrix = af.get_one_parameter_qmatrix(*gl_coefficient)
    difference = 0

    for _ in range(simulations_count):
        letters = '01'
        aa = ''
        new_aa = ''
        lambda_param = 1
        for _ in range(aa_length):
            current_time = 0
            i = j = '0'
            while True:
                # indices should be of the exact current state (letter), and not just starting one, so it's j and not i
                current_time += rnd.exponential(lambda_param / -qmatrix[int(j)][int(j)])
                if current_time <= branch_length:
                    # the exercise clearly states that "If a change has occurred, it is always to the other state"
                    # so if there was 1 then it should change to zero and if there was zero it should change to 1
                    # you can't choose again the same letter as you had before
                    j = '1' if j == '0' else '0'
                    # j = rnd.choice([x for x in letters])
                else:
                    break
            aa += i
            new_aa += j
        difference += round(get_sequences_difference(aa, new_aa) / simulations_count, 12)

    return {'execution_time': convert_seconds(time() - start_time), 'different': difference, 'same': 1 - difference}


def change_amino_acid(sequence: Union[str, List[str]], sep: str = ' ') -> Union[str, List[str]]:
    final_sequence = [AMINO_ACIDS[1][AMINO_ACIDS[0].index(i)] for i in sequence]

    return sep.join(final_sequence) if isinstance(sequence, str) else final_sequence


def simulate_amino_acid_replacements_along_tree(lg_text: str, newick_text: str, simulations_count: int = 100000,
                                                aa_length: int = 10) -> Dict[str, Union[str, float, int]]:
    start_time = time()
    probabilities = af.lq_to_qmatrix(lg_text)
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
            char = simulate_sequence_jc(tree_node.distance_to_father,1, 3, char)[2]
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


def __compute_likelihood_with_binary_jc(newick_text: str, final_sequence: Optional[str] = None, sequence_length: int = 1
                                        ) -> float:
    sequence_list = [final_sequence[i:i+sequence_length] for i in range(0, len(final_sequence), sequence_length)]
    tree = Tree(newick_text)
    sequence = ''
    char = ''
    list_node_names = tree.get_list_node_names(False, True)
    print(tree.get_leaf_count())
    print(tree.get_list_node_names(False, True))
    print(tree.get_node_listt())

    # def get_sequence(tree_node: Node) -> None:
    #     nonlocal sequence, char
    #     if tree_node.father:
    #         char = simulate_sequence_jc(tree_node.distance_to_father, sequence_length, 3, char)[2]
    #     else:
    #         char = get_random_sequence(sequence_length, None, 3)
    #
    #     if tree_node.children:
    #         for child in tree_node.children:
    #             get_sequence(child)
    #     else:
    #         sequence += char
    #
    # get_sequence(tree.root)

    # dna_len = len(dna1)
    # different_char = dna_len * get_sequences_difference(dna1, dna2)
    # same_char = dna_len - different_char
    # p_same, p_change = get_jukes_cantor_probabilities(branch_length)
    #
    # return (same_char * log(p_same, 10)) + (different_char * log(p_change, 10)) + (dna_len * log(0.25, 10))
    return 0.0


def compute_likelihood_with_binary_jc(newick_text: str, final_sequence: Optional[str] = None, sequence_length: int = 1
                                      ) -> Dict[str, Union[str, float, int]]:
    start_time = time()
    simulations_result = dict()
    if final_sequence:
        likelihood = __compute_likelihood_with_binary_jc(newick_text, final_sequence, sequence_length)
        simulations_result.update({'likelihood_of_the_data': likelihood})
    else:
        simulations_result.update({'final_sequence': 'final sequence was entered incorrectly'})

    result = {'execution_time': convert_seconds(time() - start_time)}
    result.update(simulations_result)

    return result


def simulate_amino_acid_replacements_by_lg(probabilities: Tuple[np.ndarray, ...], branch_length:
                                           float, simulations_count: int = 100000, aa_length: int = 10,
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
    # url = url_for("result_table", file_name=get_html_table_file_name(sequences_list, ("number", "count", "probability")))

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
    # url = url_for("result_table", file_name=get_html_table_file_name(sequences_list, ("number", "count", "probability")))

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
