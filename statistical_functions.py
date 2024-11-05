from math import exp
from typing import List, Union, Tuple, Optional, Dict
from numpy import random as rnd
from time import time
from datetime import timedelta
from flask import url_for
from tree import Tree

import numpy as np
import os
import random
import array_functions as af

from main import RESULT_DATA_PATH as RESULT_DATA_PATH
from main import AMINO_ACIDS as AMINO_ACIDS
from main import DNA as DNA

AMINO_ACIDS = AMINO_ACIDS[0]


def convert_seconds(seconds: float) -> str:
    return str(timedelta(seconds=seconds))


def get_sequences_differentce(dna1: str, dna2: str) -> float:
    return sum(a != b for a, b in zip(dna1, dna2)) / len(dna1)


def get_random_sequence(dna_length: Union[int, str, None] = 1) -> str:
    return ''.join(rnd.choice(DNA, int(dna_length)))


def get_random_amino_acid_structure(aa_length: Union[int, str, None] = 1,
                                    aa_frequencies: Optional[np.ndarray] = None) -> str:
    return ''.join(rnd.choice(AMINO_ACIDS, int(aa_length), True, aa_frequencies))


def get_jukes_cantor_probabilities(branch_length) -> Tuple[float, float]:
    return (1 / 4) + (3 / 4) * exp((-4 / 3) * branch_length), (1 / 4) - (1 / 4) * exp((-4 / 3) * branch_length)


def simulate_sequence_jc(branch_length: float, dna_length: Union[int, str, None] = 4) -> Tuple[float, str, str]:
    dna = get_random_sequence(dna_length)
    new_dna = ''
    p_same, p_change = get_jukes_cantor_probabilities(branch_length)
    for i in dna:
        if rnd.random() >= p_same:
            i = rnd.choice([x for x in DNA if x != i])
        new_dna += i
    return get_sequences_differentce(dna, new_dna), dna, new_dna


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
    return get_sequences_differentce(dna, new_dna), dna, new_dna


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
    return get_sequences_differentce(dna, new_dna), dna, new_dna


def generate_sequences(repetition_count: int, branch_length: float, method: int = 1,
                       dna_length: Union[int, str, None] = 4) -> List[Tuple[float, str, str]]:
    '''
    argument method
    1 - Jukes and Cantor simulation
    2 - Gillespie algorithm simulation
    3 - Gillespie algorithm simulation efficient
    '''
    result_list = []
    for i in range(repetition_count):
        if method == 1:
            new_row = simulate_sequence_jc(branch_length, dna_length)
        elif method == 2:
            new_row = simulate_dna_gillespie(branch_length, dna_length)
        elif method == 3:
            new_row = simulate_dna_gillespie_efficient(branch_length, dna_length)
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


def simulate_indel_events(dna_length: int = 4, events_count: int = 1, low: int = 0) -> Dict[str, str]:
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
                    acid: str, aa_index: int, is_replacement: bool) -> Tuple[str, str, int]:
    j = m = acid
    lambda_param = -qmatrix[aa_index][aa_index]
    counter = 0
    while True:
        current_time += rnd.exponential(lambda_param)
        if current_time <= branch_length:
            j = get_random_amino_acid_structure(1, amino_acids_frequencies)
            counter += 1 if j != m else 0
        else:
            break
    return m, j, counter if is_replacement else int(j == m)


def __simulate_amino_acid_replacements_by_lg(probabilities: Tuple[np.ndarray, np.ndarray], branch_length: float,
                                             simulations_count: int = 100000, aa_length: int = 1, node_name: str =
                                             '') -> Dict[str, str]:
    qmatrix, amino_acids_frequencies = probabilities
    replacement_statistic_1 = []
    replacement_statistic_2 = []
    node_name = f'_{node_name}' if node_name else ''

    for _ in range(simulations_count):
        amino_acid = get_random_amino_acid_structure(aa_length, amino_acids_frequencies)
        for i in range(aa_length):
            aa_index = AMINO_ACIDS.index(amino_acid[i])
            replacement_frequencies = np.array([np.float32(0.0000001) if x == aa_index else
                                                np.round(- qmatrix[aa_index][x] * 1 / qmatrix[aa_index][aa_index], 7)
                                                for x in range(20)])
            replacement_statistic_1.append(get_replacement(0, branch_length, qmatrix, replacement_frequencies,
                                                           amino_acid[i], aa_index, True))

        replacement_statistic_2.append(
            get_replacement(0, branch_length, qmatrix, amino_acids_frequencies, AMINO_ACIDS, 0, False))

    replacement_mean = sum([x[2] for x in replacement_statistic_1]) / len(replacement_statistic_1)
    replacement_probabilities = sum([x[2] for x in replacement_statistic_2]) / len(replacement_statistic_2)

    return {f'replacement_mean{node_name}': f'{replacement_mean:.5f}',
            f'replacement_probabilities{node_name}': f'{replacement_probabilities:.5f}'}


def simulate_single_site_along_branch_with_one_parameter_matrix(branch_length: float, gl_coefficient: float,
                                                                aa_length: int, simulations_count: int = 10000
                                                                ) -> Dict[str, Union[str, float, int]]:
    start_time = time()
    qmatrix = af.get_one_parameter_qmatrix(None, gl_coefficient)
    differentce = 0
    counts = simulations_count * aa_length

    for _ in range(simulations_count):
        letters = '01'
        aa = ''
        new_aa = ''
        lambda_param = 1
        for _ in range(aa_length):
            current_time = 0
            i = j = '0'
            while True:
                current_time += rnd.exponential(lambda_param / -qmatrix[int(i)][int(i)])
                if current_time <= branch_length:
                    j = rnd.choice([x for x in letters])
                else:
                    break
            aa += i
            new_aa += j
        differentce += round(get_sequences_differentce(aa, new_aa) / counts, 12)
    return {'execution_time': convert_seconds(time() - start_time), 'different': differentce, 'same': 1 - differentce}


def simulate_amino_acid_replacements_along_tree(lg_text: str, newick_text: str, simulations_count: int = 100000,
                                                aa_length: int = 10) -> Dict[str, str]:
    start_time = time()
    probabilities = af.lq_to_qmatrix(lg_text)
    newick_tree = Tree(newick_text)
    simulations_result = dict()
    nodes = newick_tree.list_node_names(newick_tree.root, False, True)
    for newick_node in nodes:
        distance = newick_node.get('distance')
        if distance:
            simulations_result.update(__simulate_amino_acid_replacements_by_lg(probabilities, distance,
                                                                               simulations_count, aa_length,
                                                                               newick_node.get('node')))
    result = {'execution_time': convert_seconds(time() - start_time)}
    result.update(simulations_result)

    return result


def simulate_amino_acid_replacements_by_lg(probabilities: Tuple[np.ndarray, np.ndarray], branch_length: float,
                                           simulations_count: int = 100000, aa_length: int = 10) -> Dict[str, str]:
    start_time = time()

    simulations_result = __simulate_amino_acid_replacements_by_lg(probabilities, branch_length, simulations_count,
                                                                  aa_length)
    result = {'execution_time': convert_seconds(time() - start_time)}
    result.update(simulations_result)

    return result


def simulate_pairwise_alignment(dna_length: int = 4, events_count: int = 1, low: int = 0) -> Dict[str, str]:
    start_time = time()
    events_result = simulate_indel_events(dna_length, events_count, low)
    result = {'execution_time': convert_seconds(time() - start_time)}
    result.update(events_result)

    return result


def calculate_change_dna_length_statistics(repetition_count: int, low: int, dna_length: int = 4,
                                           events_count: int = 4) -> Dict[str, str]:
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
                                          method: int = 1) -> Dict[str, str]:
    start_time = time()
    dna_length = int(dna_length)
    if method == 1:
        sequences_list = [simulate_insertion_event(get_random_sequence(dna_length), 0, False)
                          for _ in range(repetition_count)]
        event_name = 'inserted'
    else:
        sequences_list = [simulate_deletion_event(get_random_sequence(dna_length), low) for _ in range(repetition_count)]
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
                                     dna_length: Union[int, str, None] = 4) -> Dict[str, str]:
    start_time = time()
    sequences_list = generate_sequences(repetition_count, branch_length, method, dna_length)
    average_distance = f'{sum([x[0] for x in sequences_list]) / repetition_count:.3f}'
    execution_time = convert_seconds(time() - start_time)
    first_dna_sequence = sequences_list[0][1]
    last_dna_sequence = sequences_list[0][2]
    url = url_for("result_table",
                  file_name=get_html_table_file_name(sequences_list, ("number", "count", "probability")))

    return {'execution_time': execution_time,
            'including_file_generation': convert_seconds(time() - start_time),
            'average_distance': average_distance,
            'first_dna_sequence': first_dna_sequence,
            'last_dna_sequence': last_dna_sequence,
            'url': f'<a href=\"{url}\">{url}</a>'}


def zipf(alpha: float) -> int:
    while True:
        n = rnd.zipf(alpha)
        if 0 < n <= 50:
            return n


def get_zipf(alpha: float, simulations_count: int = 100000,
             result_rows_number: int = 5) -> Dict[str, List[Dict[str, str]]]:
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


def get_html_table_heade(heade: Tuple[str, str, str]) -> str:
    str_result = '<table id="sequencesResultTable" class="m-3 w-75 table-bordered table-danger bg-light">\n'
    style = "'text-center bg-dark bg-opacity-10'"
    str_result += f'<tr>{"".join([f"<th class={style}>{i}</th>" for i in heade])}</tr>\n'

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
                             heade: Tuple[str, str, str], variant: int = 1) -> str:
    str_table = get_html_table(data_array, get_html_table_heade(heade), variant)
    while True:
        name = f'{random.randrange(1000000 * variant, 1000000 * variant + 999999)}'
        file_name = f'{RESULT_DATA_PATH}/{name}.txt'
        if not os.path.exists(file_name):
            break
    with open(file_name, 'w') as f:
        f.write(str_table)

    return name
