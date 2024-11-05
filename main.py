from flask import Flask, request, render_template, url_for, flash, jsonify
# from typing import List, Any, Union, Tuple, Optional
from tree import Tree
import array_functions as af
import statistical_functions as sf
import design_functions as df
import os

app = Flask(__name__)
app.config.update(MAX_CONTENT_LENGTH=16 * 1024 * 1024,
                  SECRET_KEY=os.getenv('SECRET_KEY'),
                  DEBUG=True)

MENU = ({'name': 'Home page', 'url': 'index',
         'submenu': ()
         },
        {'name': 'Exercise #1', 'url': '',
         'submenu': ({'name': 'Task #1', 'url': 'exercise1task1'}, {'name': 'Task #2', 'url': 'exercise1task2'})
         },
        {'name': 'Exercise #2', 'url': '',
         'submenu': ({'name': 'Task #1', 'url': 'exercise2task1'}, {'name': 'Task #2', 'url': 'exercise2task2'},
                     {'name': 'Task #3', 'url': 'exercise2task3'})
         },
        {'name': 'Exercise #3', 'url': '',
         'submenu': ({'name': 'Task #1', 'url': 'exercise3task1'}, {'name': 'Task #2, #5', 'url': 'exercise3task2'},
                     {'name': 'Task #3, #5', 'url': 'exercise3task3'}, {'name': 'Task #4, #5', 'url': 'exercise3task4'})
         },
        {'name': 'Exercise #4', 'url': '',
         'submenu': ({'name': 'Task #1', 'url': 'exercise4task1'}, {'name': 'Task #2, #3', 'url': 'exercise4task2'},
                     {'name': 'Task #4', 'url': 'exercise4task4'}, {'name': 'Task #5', 'url': 'exercise4task5'},
                     {'name': 'Task #6', 'url': 'exercise4task6'})
         },
        {'name': 'Exercise #5', 'url': '',
         'submenu': ({'name': 'Task #1', 'url': 'exercise5task1'}, {'name': 'Task #2', 'url': 'exercise5task2'},
                     {'name': 'Task #3', 'url': 'exercise5task3'})
         },
        {'name': 'Exercise #6', 'url': '',
         'submenu': ({'name': 'Task #1', 'url': 'exercise6task1'}, {'name': 'Task #2', 'url': 'exercise6task2'}
                     # ,
                     # {'name': 'Task #3', 'url': 'exercise6task3'}
                     )
         }
        )

CONTENT_TABLE = [[0, 17, 21, 31, 23], [17, 0, 30, 34, 21], [21, 30, 0, 28, 39], [31, 34, 28, 0, 43],
                 [23, 21, 39, 43, 0]]

CONTENT_TEXTAREA = ('((e:11.0,(a:8.5,b:8.5):2.5):5.5,(c:14.0,d:14.0):2.5);',
                    '((t:11.0,(a:8.5,b:8.5):2.5):5.5,(c:14.0,d:14.0):2.5);',
                    ' \n'
                    '0.425093\n' 
                    '0.276818 0.751878\n' 
                    '0.395144 0.123954 5.076149\n' 
                    '2.489084 0.534551 0.528768 0.062556\n' 
                    '0.969894 2.807908 1.695752 0.523386 0.084808\n' 
                    '1.038545 0.363970 0.541712 5.243870 0.003499 4.128591\n' 
                    '2.066040 0.390192 1.437645 0.844926 0.569265 0.267959 0.348847\n' 
                    '0.358858 2.426601 4.509238 0.927114 0.640543 4.813505 0.423881 0.311484\n' 
                    '0.149830 0.126991 0.191503 0.010690 0.320627 0.072854 0.044265 0.008705 0.108882\n' 
                    '0.395337 0.301848 0.068427 0.015076 0.594007 0.582457 0.069673 0.044261 0.366317 4.145067\n' 
                    '0.536518 6.326067 2.145078 0.282959 0.013266 3.234294 1.807177 0.296636 0.697264 0.159069 '
                    '0.137500\n' 
                    '1.124035 0.484133 0.371004 0.025548 0.893680 1.672569 0.173735 0.139538 0.442472 4.273607 '
                    '6.312358 0.656604\n' 
                    '0.253701 0.052722 0.089525 0.017416 1.105251 0.035855 0.018811 0.089586 0.682139 1.112727 '
                    '2.592692 0.023918 1.798853\n' 
                    '1.177651 0.332533 0.161787 0.394456 0.075382 0.624294 0.419409 0.196961 0.508851 0.078281 '
                    '0.249060 0.390322 0.099849 0.094464\n' 
                    '4.727182 0.858151 4.008358 1.240275 2.784478 1.223828 0.611973 1.739990 0.990012 0.064105 '
                    '0.182287 0.748683 0.346960 0.361819 1.338132\n' 
                    '2.139501 0.578987 2.000679 0.425860 1.143480 1.080136 0.604545 0.129836 0.584262 1.033739 '
                    '0.302936 1.136863 2.020366 0.165001 0.571468 6.472279\n' 
                    '0.180717 0.593607 0.045376 0.029890 0.670128 0.236199 0.077852 0.268491 0.597054 0.111660 '
                    '0.619632 0.049906 0.696175 2.457121 0.095131 0.248862 0.140825\n' 
                    '0.218959 0.314440 0.612025 0.135107 1.165532 0.257336 0.120037 0.054679 5.306834 0.232523 '
                    '0.299648 0.131932 0.481306 7.803902 0.089613 0.400547 0.245841 3.151815\n' 
                    '2.547870 0.170887 0.083688 0.037967 1.959291 0.210332 0.245034 0.076701 0.119013 10.649107 '
                    '1.702745 0.185202 1.898718 0.654683 0.296501 0.098369 2.188158 0.189510 0.249313\n' 
                    '\n'
                    '0.079066 0.055941 0.041977 0.053052 0.012937 0.040767 0.071586 0.057337 0.022355 0.062157 '
                    '0.099081 0.064600 0.022951 0.042302 0.044040 0.061197 0.053287 0.012066 0.034155 0.069147',
                    '(s1:0.2,s2:0.3,s3:0.4);')

DNA_LENGTH = (100, 'AGCTC', 1000, 11, 1)
AA_LENGTH = (1, 11)
DISTANCE_TO_FATHER = BRANCH_LENGTH = 0.1
GL_COEFFICIENT = 0.2
REPETITION_COUNT = 1000
ARGUMENT_ZIPF_ALPHA = 1.2
SIMULATIONS_COUNT = (100000, 10000)
RESULT_ROWS_NUMBER = 3
RESULT_DATA_PATH = 'data_files'
EVENTS_COUNT = (20, 4)

AMINO_ACIDS = (('A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'),
               ('Ala', 'Arg', 'Asn', 'Asp', 'Cys', 'Gln', 'Glu', 'Gly', 'His', 'Ile', 'Leu', 'Lys', 'Met', 'Phe', 'Pro',
                'Ser', 'Thr', 'Trp', 'Tyr', 'Val'))
DNA = ('A', 'C', 'G', 'T')


@app.route('/')
@app.route('/index', methods=['GET'])
def index():
    return render_template('index.html', title=(':', '  Home page'), menu=MENU)


@app.route('/exercise1task1', methods=['GET'])
def exercise1task1():
    return render_template('exercise1task1.html',
                           title=(' - Exercise #1 in building a web-server:',
                                  ' finding the letter A or a in a name (Task #1)'), menu=MENU)


@app.route('/exercise1task2', methods=['GET'])
def exercise1task2():
    return render_template('exercise1task2.html', content_textarea=CONTENT_TEXTAREA[0],
                           title=(' - Exercise #1 in building a web-server:', ' finding a node by name (Task #2)'),
                           menu=MENU)


@app.route('/exercise2task1', methods=['GET'])
def exercise2task1():
    return render_template('exercise2task1.html', content_table=af.get_html_table(af.set_names_to_array(CONTENT_TABLE)),
                           title=(' - Exercise #2 in building and visualizing phylogenetic trees:',
                                  ' newick generator based on the UPGMA algorithm (Task #1)'), menu=MENU)


@app.route('/exercise2task2', methods=['GET'])
def exercise2task2():
    return render_template('exercise2task2.html', content_textarea=CONTENT_TEXTAREA[0],
                           distance_to_father=DISTANCE_TO_FATHER,
                           title=(' - Exercise #2 in building and visualizing phylogenetic trees:',
                                  ' adding / removing units to branches (Task #2)'), menu=MENU)


@app.route('/exercise2task3', methods=['GET'])
def exercise2task3():
    return render_template('exercise2task3.html', content_textarea=CONTENT_TEXTAREA,
                           title=(' - Exercise #2 in building and visualizing phylogenetic trees:',
                                  ' Robinson Folds distance calculation (Task #3)'), menu=MENU)


@app.route('/exercise3task1', methods=['GET'])
def exercise3task1():
    return render_template('exercise3task1.html', dna_length=DNA_LENGTH[0],
                           title=(' - Exercise #3 in building and visualizing phylogenetic trees (simulations):',
                                  ' generates a random DNA sequence (Task #1)'), menu=MENU)


@app.route('/exercise3task2', methods=['GET'])
def exercise3task2():
    return render_template('exercise3task2.html', dna_length=DNA_LENGTH[0],
                           branch_length=BRANCH_LENGTH, repetition_count=REPETITION_COUNT,
                           title=(' - Exercise #3 in building and visualizing phylogenetic trees (simulations):',
                                  ' DNA sequence generator based on the JC model (Task #2, #5)'), menu=MENU)


@app.route('/exercise3task3', methods=['GET'])
def exercise3task3():
    return render_template('exercise3task3.html', dna_length=DNA_LENGTH[0],
                           branch_length=BRANCH_LENGTH, repetition_count=REPETITION_COUNT,
                           title=(' - Exercise #3 in building and visualizing phylogenetic trees (simulations):',
                                  ' DNA sequence generator based on the Gillespie algorithm (Task #3, #5)'), menu=MENU)


@app.route('/exercise3task4', methods=['GET'])
def exercise3task4():
    return render_template('exercise3task4.html', dna_length=DNA_LENGTH[0],
                           branch_length=BRANCH_LENGTH, repetition_count=REPETITION_COUNT,
                           title=(' - Exercise #3 in building and visualizing phylogenetic trees (simulations):',
                                  ' DNA sequence generator based on the more efficient Gillespie algorithm '
                                  '(Task #4, #5)'), menu=MENU)


@app.route('/exercise4task1', methods=['GET'])
def exercise4task1():
    return render_template('exercise4task1.html', simulations_count=SIMULATIONS_COUNT[0],
                           argument_zipf_alpha=ARGUMENT_ZIPF_ALPHA, result_rows_number=RESULT_ROWS_NUMBER,
                           title=(' - Exercise #4 in building and visualizing phylogenetic trees (simulations):',
                                  ' generator a truncated Zipfian length distribution (Task #1)'), menu=MENU)


@app.route('/exercise4task2', methods=['GET'])
def exercise4task2():
    return render_template('exercise4task2.html', dna_length=DNA_LENGTH[1],
                           title=(' - Exercise #4 in building and visualizing phylogenetic trees (simulations):',
                                  ' simulator of insertion / deletion of random sequence into / from DNA sequence '
                                  '(Task #2, #3)'), menu=MENU)


@app.route('/exercise4task4', methods=['GET'])
def exercise4task4():
    return render_template('exercise4task4.html', dna_length=DNA_LENGTH[0], simulations_count=SIMULATIONS_COUNT[0],
                           title=(' - Exercise #4 in building and visualizing phylogenetic trees (simulations):',
                                  ' simulator of deletion of random sequence from DNA sequence (Task #4)'), menu=MENU)


@app.route('/exercise4task5', methods=['GET'])
def exercise4task5():
    return render_template('exercise4task5.html', dna_length=DNA_LENGTH[2], simulations_count=SIMULATIONS_COUNT[0],
                           events_count=EVENTS_COUNT[0],
                           title=(' - Exercise #4 in building and visualizing phylogenetic trees (simulations):',
                                  ' simulator of insertion / deletion of random sequence into / from DNA sequence '
                                  '(Task #5)'), menu=MENU)


@app.route('/exercise4task6', methods=['GET'])
def exercise4task6():
    return render_template('exercise4task6.html', dna_length=DNA_LENGTH[3], events_count=EVENTS_COUNT[1],
                           title=(' - Exercise #4 in building and visualizing phylogenetic trees (simulations):',
                                  ' simulator of the pairwise alignment between the original sequence and the sequence '
                                  'after experiencing indel events (Task #6)'), menu=MENU)


@app.route('/exercise5task1', methods=['GET'])
def exercise5task1():
    return render_template('exercise5task1.html', content_textarea=CONTENT_TEXTAREA[2],
                           title=(' - Exercise #5 simulating with the LG matrix:', ' reads the file LG.txt into a Q '
                                  'matrix (Task #1)'), menu=MENU)


@app.route('/exercise5task2', methods=['GET'])
def exercise5task2():
    return render_template('exercise5task2.html', content_textarea=CONTENT_TEXTAREA[2],
                           simulations_count=SIMULATIONS_COUNT[1], branch_length=BRANCH_LENGTH, aa_length=AA_LENGTH[0],
                           title=(' - Exercise #5 simulating with the LG matrix:', ' simulator of the amino acid '
                                  'replacements using the LG matrix (Task #2)'), menu=MENU)


@app.route('/exercise5task3', methods=['GET'])
def exercise5task3():
    return render_template('exercise5task3.html', content_textarea=CONTENT_TEXTAREA,
                           simulations_count=SIMULATIONS_COUNT[0], aa_length=AA_LENGTH[0],
                           title=(' - Exercise #5 simulating with the LG matrix:', ' simulator of the amino acid '
                                  'replacements using the LG matrix, along the tree (Task #3)'), menu=MENU)


@app.route('/exercise6task1', methods=['GET'])
def exercise6task1():
    return render_template('exercise6task1.html', gl_coefficient=GL_COEFFICIENT,
                           title=(' - Exercise #6 processing data with two-state continuous time Markov models:',
                                  ' generator a one parameter gain-loss Q matrix (Task #1)'), menu=MENU)


@app.route('/exercise6task2', methods=['GET'])
def exercise6task2():
    return render_template('exercise6task2.html', gl_coefficient=GL_COEFFICIENT,
                           branch_length=BRANCH_LENGTH, simulations_count=SIMULATIONS_COUNT[1],
                           title=(' - Exercise #6 processing data with two-state continuous time Markov models:',
                                  ' generator a one parameter gain-loss Q matrix (Task #2)'), menu=MENU)


@app.route('/check_name', methods=['POST'])
def check_name():
    if request.method == 'POST':
        text_name = request.form.get('textName')

        if text_name:
            result = text_name.find('A') > -1 or text_name.find('a') > -1

            return jsonify(message=result)


@app.route('/clustering', methods=['GET'])
def clustering():
    if request.method == 'GET':
        result = af.clustering(CONTENT_TABLE)

        return jsonify(message=str(result))


@app.route('/result_table', methods=['GET'])
def result_table():
    if request.method == 'GET':
        file_name = request.args["file_name"]
        full_file_name = f'{RESULT_DATA_PATH}/{file_name}.txt'
        with open(full_file_name, 'r') as f:
            model_table = f.read()

        return render_template('table.html', title=(f' - file number ', f'  ({file_name})'),
                               model_table=model_table, menu=MENU)


@app.route('/get_robinson_foulds_distance', methods=['POST'])
def get_robinson_foulds_distance():
    if request.method == 'POST':
        newick_text1 = request.form.get('newickText1')
        newick_text2 = request.form.get('newickText2')
        result = Tree.get_robinson_foulds_distance(newick_text1, newick_text2)

        return jsonify(message=result)


@app.route('/generate_dna_sequence', methods=['POST'])
def generate_dna_sequence():
    if request.method == 'POST':
        dna_length = int(request.form.get('dnaLength'))
        result = sf.get_random_sequence(dna_length)

        return jsonify(message=result)


@app.route('/change_dna_length', methods=['POST'])
def change_dna_length():
    if request.method == 'POST':
        dna_length = request.form.get('dnaLength')
        method = int(request.form.get('method'))

        sequences = sf.change_dna_length(method, dna_length)
        message = ''
        for key, value in sequences.items():
            if key.find('_dna_sequence') != -1:
                if 'length_of_insertion' in sequences.keys() and key == 'last_dna_sequence':
                    value = df.dna_design(value, (sequences['start_position'], sequences['end_position']), (14, 11))
                else:
                    value = df.dna_design(value)
            message += f'{df.key_design(key)}{df.value_design(value)}<br>'
        result = (sequences['last_dna_sequence'], f'<b>{message}</b>')

        return jsonify(message=result)



@app.route('/simulate_single_site_along_branch_with_one_parameter_matrix', methods=['POST'])
def simulate_single_site_along_branch():
    if request.method == 'POST':
        branch_length = float(request.form.get('branchLength'))
        gl_coefficient = float(request.form.get('glCoefficient'))
        simulations_count = int(request.form.get('simulationsCount'))

        statistics = sf.simulate_single_site_along_branch_with_one_parameter_matrix(branch_length, gl_coefficient,
                                                                                    1, simulations_count)
        result = df.result_design(statistics)

        return jsonify(message=result)



@app.route('/simulate_amino_acid_replacements_along_tree', methods=['POST'])
def simulate_amino_acid_replacements_along_tree():
    if request.method == 'POST':
        lg_text = request.form.get('textArea')
        newick_text = request.form.get('newickText')
        aa_length = int(request.form.get('aaLength'))
        simulations_count = int(request.form.get('simulationsCount'))

        statistics = sf.simulate_amino_acid_replacements_along_tree(lg_text, newick_text, simulations_count, aa_length)
        result = df.result_design(statistics)

        return jsonify(message=result)


@app.route('/simulate_amino_acid_replacements_by_lg', methods=['POST'])
def simulate_amino_acid_replacements_by_lg():
    if request.method == 'POST':
        lg_text = request.form.get('textArea')
        aa_length = int(request.form.get('aaLength'))
        branch_length = float(request.form.get('branchLength'))
        simulations_count = int(request.form.get('simulationsCount'))

        statistics = sf.simulate_amino_acid_replacements_by_lg(af.lq_to_qmatrix(lg_text), branch_length,
                                                               simulations_count, aa_length)
        result = df.result_design(statistics)

        return jsonify(message=result)


@app.route('/get_one_parameter_qmatrix', methods=['POST'])
def get_one_parameter_qmatrix():
    if request.method == 'POST':
        gl_coefficient = float(request.form.get('glCoefficient'))

        result = af.get_html_table(af.set_names_to_array(af.get_one_parameter_qmatrix(gl_coefficient, None).tolist(),
                                                         ('0', '1')))

        return jsonify(message=result)


@app.route('/lq_to_qmatrix', methods=['POST'])
def lq_to_qmatrix():
    if request.method == 'POST':
        lg_text = request.form.get('textArea')

        result = af.get_html_table(af.set_names_to_array(af.lq_to_qmatrix(lg_text)[0].tolist(), AMINO_ACIDS[1]))

        return jsonify(message=result)


@app.route('/simulate_pairwise_alignment', methods=['POST'])
def simulate_pairwise_alignment():
    if request.method == 'POST':
        dna_length = int(request.form.get('dnaLength'))
        events_count = int(request.form.get('eventsCount'))

        statistics = sf.simulate_pairwise_alignment(dna_length, events_count, -49)

        result = df.result_design(statistics)

        return jsonify(message=result)


@app.route('/calculate_change_dna_length_statistics', methods=['POST'])
def calculate_change_dna_length_statistics():
    if request.method == 'POST':
        dna_length = int(request.form.get('dnaLength'))
        simulations_count = int(request.form.get('simulationsCount'))
        events_count = int(request.form.get('eventsCount'))

        statistics = sf.calculate_change_dna_length_statistics(simulations_count, - 49, dna_length, events_count)
        result = df.result_design(statistics)

        return jsonify(message=result)


@app.route('/calculate_event_simulation_statistics', methods=['POST'])
def calculate_event_simulation_statistics():
    if request.method == 'POST':
        dna_length = int(request.form.get('dnaLength'))
        simulations_count = int(request.form.get('simulationsCount'))
        method = int(request.form.get('method'))

        sequences = sf.calculate_event_simulation_statistics(simulations_count, - 49, dna_length, method)
        result = df.result_design(sequences)

        return jsonify(message=result)


@app.route('/compute_average_distance', methods=['POST'])
def compute_average_distance():
    if request.method == 'POST':
        dna_length = int(request.form.get('dnaLength'))
        branch_length = float(request.form.get('branchLength'))
        repetition_count = int(request.form.get('repetitionCount'))
        method = int(request.form.get('method'))

        sequences = sf.calculate_simulations_statistics(repetition_count, branch_length, method, dna_length)
        result = df.result_design(sequences)

        return jsonify(message=result)


@app.route('/get_zipfian_distribution', methods=['POST'])
def get_zipfian_distribution():
    if request.method == 'POST':
        zipf_alpha = float(request.form.get('argumentZipfAlpha'))
        simulations_count = int(request.form.get('simulationsCount'))
        result_rows_number = int(request.form.get('resultRowsNumber'))

        simulations_result = sf.get_zipf(zipf_alpha, simulations_count, result_rows_number)
        result = f'{df.key_design("execution time")}{df.value_design(simulations_result.get("execution_time"))}<br>'
        for row in simulations_result['result']:
            for key, value in row.items():
                result += f'{df.key_design(key)}{df.value_design(value)}<br>'
        result = f'<b>{result}</b>'

        return jsonify(message=result)


@app.route('/upload_file', methods=['POST'])
def upload_file():
    '''
    Handle a POST request containing a file and return its contents.

    This function processes a POST request with JSON data that includes a file upload under the
    key `newickFile`. It reads the contents of this file and returns a response where the body
    contains the file content in a JSON object with the key `'message'`.

    Request: The POST request object containing JSON data with a file upload.
                           The file should be provided with the key 'newickFile'.

    Response: A response object where the body contains a JSON object with a single key
                  'message' containing the content of the uploaded file as a string.
    '''
    if request.method == 'POST':
        newick_file = request.files.get('newickFile')

        try:
            if newick_file:
                return jsonify(message=newick_file.read().decode('utf-8'))

        except Exception as error:
            return jsonify(message=f'Error: {error}'), 400


@app.route('/find_node', methods=['POST'])
def find_node():
    '''
    Process a POST request containing JSON data to search for a node in a tree.

    This function handles a POST request with JSON payload that includes a Newick formatted
    string and a node name. It parses the Newick string to construct a tree and then searches
    for the specified node name within that tree. The function returns a response where the body
    contains a JSON object with the search result as a boolean value.

    Request: The POST request object containing JSON data with the following structure:
                           - 'newickText': A string in Newick format representing the tree structure.
                           - 'nodeName': The name of the node to search for in the tree.

    Response: A response object where the body contains a JSON object with a single key
                  'message' indicating whether the node was found in the tree. The value is `True`
                  if the node is present, `False` otherwise.
    '''
    if request.method == 'POST':
        newick_text = request.form.get('newickText')
        node_name = request.form.get('nodeName')

        try:
            if newick_text:
                tree = Tree(newick_text)
                return jsonify(message=tree.find_node_by_name(node_name))

        except Exception as error:
            return jsonify(message=f'Error: {error}'), 400


@app.route('/print_tree', methods=['POST'])
def print_tree():
    if request.method == 'POST':
        newick_text = request.form.get('newickText')
        status = request.form.get('status')
        try:
            html_result = ''
            if newick_text:
                tree = Tree(newick_text)
                html_result = tree.get_html_tree(('tree-padding tree-vertical-lines tree-horizontal-lines '
                                                  'tree-summaries tree-markers tree-buttons'), status)
            return jsonify(message=html_result)
        except Exception as error:
            return jsonify(message=f'Error: {error}'), 400


@app.route('/change_distance_to_father', methods=['POST'])
def change_distance_to_father():
    if request.method == 'POST':
        newick_text = request.form.get('newickText')
        distance_to_father = float(request.form.get('distanceToFather'))
        if newick_text:
            try:
                tree = Tree(newick_text)
                tree.add_distance_to_father(distance_to_father)
                newick_text = tree.get_newick()
                return jsonify(message=newick_text)
            except Exception as error:
                return jsonify(message=f'Error: {error}'), 400


if __name__ == '__main__':
    app.run(port=4000, debug=True)
