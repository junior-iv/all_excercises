// document.getElementById('textName').onkeypress = function(e) {
//   if (e.keyCode === 13) {
//     document.getElementById('checkButton').click();
//   }
// };
//
// document.getElementById('nodeName').onkeypress = function(e) {
//   if (e.keyCode === 13) {
//     document.getElementById('findButton').click();
//   }
// };
//
// document.getElementById('distanceToFather').onkeypress = function(e) {
//   if (e.keyCode === 13) {
//     document.getElementById('theButton').click();
//   }
// };
// if (document.querySelector('#carousel')) {
// document.addEventListener('DOMContentLoaded', function () {
//     let elems = document.querySelector('.carousel');
//     let instances = M.Carousel.init(elems, {
//         duration: 1000,
//         dist: -50,
//         shift: 20,
//         numVisible: 3,
//         fullWidth: true
//     });
// });
// }

if (document.querySelector('#glCoefficient')) {
    getOneParameterQMatrix();
}

document.getElementById('newickForm').addEventListener('submit', function (e) {
    e.preventDefault();
    document.getElementById('theButton').click();
});

function printTree(status) {
    const newickText = document.getElementById('textArea');
    const formData = new FormData();
    formData.append('newickText', newickText.value);
    formData.append('status', status);

    fetch('/print_tree', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            document.getElementById('result').innerHTML = data.message;
        })
        .catch(error => {
            console.error('Error:', error);
            // showMessage(3, error.message)
        });
}

function changeTree() {
    const newickText = document.getElementById('textArea');
    const distanceToFather = document.getElementById('distanceToFather');
    const formData = new FormData();
    formData.append('newickText', newickText.value);
    formData.append('distanceToFather', distanceToFather.value);

    if (distanceToFather.value === '') { return }

    fetch('/change_distance_to_father', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            document.getElementById('textArea').value = data.message;
            printTree('open');
        })
        .catch(error => {
            console.error('Error:', error);
            showMessage(3, error.message)
        });
}

function uploadFile() {
    const flexSwitchProcessFileOnServer = document.getElementById('flexSwitchProcessFileOnServer')
    let newickFile = document.getElementById('newickFile').files[0];
    let textArea = document.getElementById('textArea');
    if (flexSwitchProcessFileOnServer.checked) {
        const formData = new FormData();
        formData.append('newickFile', newickFile);

        fetch('/upload_file', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                textArea.value = data.message;
            })
            .catch(error => {
                console.error('Error:', error);
                showMessage(3, error.message)
            });
    } else {
        let file = newickFile;

        if (file) {
            let reader = new FileReader();

            reader.onload = function(event) {
                textArea.value = event.target.result;
            };
            reader.readAsText(file);
        }
    }
}

function findSelectTxt(text) {
    const newickText = document.getElementById('textArea');
    newickText.selectionStart = newickText.value.indexOf(text + ':');
    newickText.selectionEnd = newickText.selectionStart + text.length;
    newickText.focus();
}

function findNode() {
    const newickText = document.getElementById('textArea');
    const nodeName = document.getElementById('nodeName');
    const flexSwitchSelectFound = document.getElementById('flexSwitchSelectFound');
    const formData = new FormData();
    formData.append('newickText', newickText.value);
    formData.append('nodeName', nodeName.value);

    fetch('/find_node', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            if (data.message) {
                showMessage(1);
                if (flexSwitchSelectFound.checked) {findSelectTxt(nodeName.value)}
            } else {
                showMessage(2);
            }
        })
        .catch(error => {
            showMessage(3, error.message)
        });
}

function checkName() {
    const textName = document.getElementById('textName');
    const formData = new FormData();
    formData.append('textName', textName.value);

    fetch('/check_name', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            if (data.message) {
                showMessage(1);
            } else {
                showMessage(2);
            }
        })
        .catch(error => {
            showMessage(3, error.message)
        });
}

function clustering() {

    fetch('/clustering', {
        method: 'GET',
    })
        .then(response => response.json())
        .then(data => {
            showMessage(1, data.message)
        })
        .catch(error => {
            console.error('Error:', error);
            showMessage(3, error.message)
        });
}

function getRobinsonFouldsDistance() {
    const newickText1 = document.getElementById('newickText1');
    const newickText2 = document.getElementById('newickText2');
    const formData = new FormData();
    formData.append('newickText1', newickText1.value);
    formData.append('newickText2', newickText2.value);

    fetch('/get_robinson_foulds_distance', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            showMessage(1, data.message)
        })
        .catch(error => {
            console.error('Error:', error);
            showMessage(3, error.message)
        });
}

function getOneParameterQMatrix() {
    const parameterName = document.getElementById('parameterName');
    const glCoefficient = document.getElementById('glCoefficient');
    const formData = new FormData();
    formData.append('parameterName', parameterName.value);
    formData.append('glCoefficient', glCoefficient.value);

    fetch('/get_one_parameter_qmatrix', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            document.getElementById('resultTable').innerHTML = data.message;
        })
        .catch(error => {
            console.error('Error:', error);
            showMessage(3, error.message)
        });
}

function lgToQMatrix() {
    const textArea = document.getElementById('textArea');
    const formData = new FormData();
    formData.append('textArea', textArea.value);

    fetch('/lq_to_qmatrix', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            document.getElementById('result').innerHTML = data.message;
        })
        .catch(error => {
            console.error('Error:', error);
            showMessage(3, error.message)
        });
}

function calculatePij() {
    const parameterName = document.getElementById('parameterName');
    const glCoefficient = document.getElementById('glCoefficient');
    const parametersP = [document.getElementById('P00').value, document.getElementById('P01').value,
        document.getElementById('P10').value, document.getElementById('P11').value];
    const formData = new FormData();
    formData.append('parameterName', parameterName.value);
    formData.append('glCoefficient', glCoefficient.value);
    formData.append('parametersP', parametersP);

    const loaderID = getLoader();
    setVisibilityLoader(true, loaderID);

    fetch('/calculate_pij', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            setVisibilityLoader(false, loaderID);
            showMessage(1, data.message)
        })
        .catch(error => {
            setVisibilityLoader(false, loaderID);
            console.error('Error:', error);
            showMessage(3, error.message)
        });
}

function simulateSitesAlongBranchWithOneParameterMatrix() {
    const aaLength = document.getElementById('aaLength');
    const branchLength = document.getElementById('branchLength');
    const parameterName = document.getElementById('parameterName');
    const glCoefficient = document.getElementById('glCoefficient');
    const simulationsCount = document.getElementById('simulationsCount');
    const formData = new FormData();
    formData.append('aaLength', aaLength.value);
    formData.append('branchLength', branchLength.value);
    formData.append('parameterName', parameterName.value);
    formData.append('glCoefficient', glCoefficient.value);
    formData.append('simulationsCount', simulationsCount.value);

    const loaderID = getLoader();
    setVisibilityLoader(true, loaderID);

    fetch('/simulate_sites_along_branch_with_one_parameter_matrix', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            setVisibilityLoader(false, loaderID);
            showMessage(1, data.message)
        })
        .catch(error => {
            setVisibilityLoader(false, loaderID);
            console.error('Error:', error);
            showMessage(3, error.message)
        });
}

function simulateAminoAcidReplacementsAlongTree() {
    const textArea = document.getElementById('textArea');
    const aaLength = document.getElementById('aaLength');
    const newickText = document.getElementById('newickText');
    const simulationsCount = document.getElementById('simulationsCount');
    const formData = new FormData();
    formData.append('textArea', textArea.value);
    formData.append('aaLength', aaLength.value);
    formData.append('newickText', newickText.value);
    formData.append('simulationsCount', simulationsCount.value);

    const loaderID = getLoader();
    setVisibilityLoader(true, loaderID);

    fetch('/simulate_amino_acid_replacements_along_tree', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            setVisibilityLoader(false, loaderID);
            showMessage(1, data.message)
        })
        .catch(error => {
            setVisibilityLoader(false, loaderID);
            console.error('Error:', error);
            showMessage(3, error.message)
        });
}

function simulateAminoAcidReplacementsByLG() {
    const textArea = document.getElementById('textArea');
    const aaLength = document.getElementById('aaLength');
    const aminoAcid = document.getElementById('aminoAcid');
    const branchLength = document.getElementById('branchLength');
    const simulationsCount = document.getElementById('simulationsCount');
    const formData = new FormData();
    formData.append('textArea', textArea.value);
    formData.append('aaLength', aaLength.value);
    formData.append('aminoAcid', aminoAcid.value);
    formData.append('branchLength', branchLength.value);
    formData.append('simulationsCount', simulationsCount.value);

    const loaderID = getLoader();
    setVisibilityLoader(true, loaderID);

    fetch('/simulate_amino_acid_replacements_by_lg', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            setVisibilityLoader(false, loaderID);
            showMessage(1, data.message)
        })
        .catch(error => {
            setVisibilityLoader(false, loaderID);
            console.error('Error:', error);
            showMessage(3, error.message)
        });
}
function simulateWithBinaryJC(variant = 1) {
    const textArea = document.getElementById('textArea');
    const finalSequence = document.getElementById('finalSequence');
    const simulationsCount = document.getElementById('simulationsCount');
    const formData = new FormData();
    formData.append('textArea', textArea.value);
    formData.append('finalSequence', finalSequence.value);
    formData.append('simulationsCount', simulationsCount.value);
    formData.append('variant', variant);

    const loaderID = getLoader();
    setVisibilityLoader(true, loaderID);

    fetch('/simulate_with_binary_jc', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            setVisibilityLoader(false, loaderID);
            showMessage(1, data.message)
        })
        .catch(error => {
            setVisibilityLoader(false, loaderID);
            console.error('Error:', error);
            showMessage(3, error.message)
        });
}


function computeLikelihoodWithBinaryJC() {
    const textArea = document.getElementById('textArea');
    const finalSequence = document.getElementById('finalSequence');
    const formData = new FormData();
    formData.append('textArea', textArea.value);
    formData.append('finalSequence', finalSequence.value);

    const loaderID = getLoader();
    setVisibilityLoader(true, loaderID);

    fetch('/compute_likelihood_with_binary_jc', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            setVisibilityLoader(false, loaderID);
            showMessage(1, data.message)
        })
        .catch(error => {
            setVisibilityLoader(false, loaderID);
            console.error('Error:', error);
            showMessage(3, error.message)
        });
}


function getMinimized() {
    const loaderID = getLoader();
    setVisibilityLoader(true, loaderID);

    fetch('/get_minimized', {
        method: 'GET',
    })
        .then(response => response.json())
        .then(data => {
            setVisibilityLoader(false, loaderID);
            showMessage(1, data.message)
        })
        .catch(error => {
            setVisibilityLoader(false, loaderID);
            console.error('Error:', error);
            showMessage(3, error.message)
        });
}


function getLogLikelihood(variant = 1) {
    const formData = new FormData();
    const branchLength = document.getElementById('branchLength');
    const dna = [document.getElementById('dna1').value, document.getElementById('dna2').value];
    formData.append('branchLength', branchLength.value);
    formData.append('dna', dna);
    formData.append('variant', variant);

    const loaderID = getLoader();
    setVisibilityLoader(true, loaderID);

    fetch('/get_log_likelihood', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            setVisibilityLoader(false, loaderID);
            showMessage(1, data.message)
        })
        .catch(error => {
            setVisibilityLoader(false, loaderID);
            console.error('Error:', error);
            showMessage(3, error.message)
        });
}


function getMaximized() {
    const formData = new FormData();
    if (document.querySelector('#minX')) {
        const limitsX = [document.getElementById('minX').value, document.getElementById('maxX').value];
        formData.append('limitsX', limitsX);
    }

    const loaderID = getLoader();
    setVisibilityLoader(true, loaderID);

    fetch('/get_maximized', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            setVisibilityLoader(false, loaderID);
            showMessage(1, data.message)
        })
        .catch(error => {
            setVisibilityLoader(false, loaderID);
            console.error('Error:', error);
            showMessage(3, error.message)
        });
}


function generateDNASequence() {
    const dnaLength = document.getElementById('dnaLength');
    const formData = new FormData();
    formData.append('dnaLength', dnaLength.value);

    const loaderID = getLoader();
    setVisibilityLoader(true, loaderID);

    fetch('/generate_dna_sequence', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            setVisibilityLoader(false, loaderID);
            showMessage(1, data.message)
        })
        .catch(error => {
            setVisibilityLoader(false, loaderID);
            console.error('Error:', error);
            showMessage(3, error.message)
        });
}

function getLoader() {
    const rnd = Math.floor(Math.random() * 3);
    if (rnd === 0) {return 'loaderCube'}
    else if (rnd === 1) {return 'loaderSpinner'}
    else if (rnd === 2) {return 'loaderGrow'}
}

function setVisibilityLoader(visible = true, loaderID) {
    if (visible) {
        document.getElementById(loaderID).classList.remove('invisible');

        document.getElementById('divWarning').style.visibility = 'hidden';
        document.getElementById('divDanger').style.visibility = 'hidden';
        document.getElementById('divInfo').style.visibility = 'hidden';
        document.getElementById('divSuccess').style.visibility = 'hidden';
        document.getElementById('divSecondary').style.visibility = 'hidden';
    } else {
        document.getElementById(loaderID).classList.add('invisible');
    }
}

function changeDNALength(method) {
    const dnaLength = document.getElementById('dnaLength');
    const flexSwitchUseResultingDNA = document.getElementById('flexSwitchUseResultingDNA');
    const formData = new FormData();
    formData.append('dnaLength', dnaLength.value);
    formData.append('method', method);

    fetch('/change_dna_length', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            if (flexSwitchUseResultingDNA.checked) {dnaLength.value = data.message[0]}
            showMessage(1, data.message[1]);
        })
        .catch(error => {
            console.error('Error:', error);
            showMessage(3, error.message);
        });
}

function simulatePairwiseAlignment() {
    const dnaLength = document.getElementById('dnaLength');
    const eventsCount = document.getElementById('eventsCount');
    const formData = new FormData();
    formData.append('dnaLength', dnaLength.value);
    formData.append('eventsCount', eventsCount.value);

    const loaderID = getLoader();
    setVisibilityLoader(true, loaderID);

    fetch('/simulate_pairwise_alignment', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            setVisibilityLoader(false, loaderID);
            showMessage(1, data.message);
        })
        .catch(error => {
            setVisibilityLoader(false, loaderID);
            console.error('Error:', error);
            showMessage(3, error.message);
        });

}

function calculateChangeDNALengthStatistics() {
    const dnaLength = document.getElementById('dnaLength');
    const repetitionCount = document.getElementById('simulationsCount');
    const eventsCount = document.getElementById('eventsCount');
    const formData = new FormData();
    formData.append('dnaLength', dnaLength.value);
    formData.append('simulationsCount', repetitionCount.value);
    formData.append('eventsCount', eventsCount.value);

    const loaderID = getLoader();
    setVisibilityLoader(true, loaderID);

    fetch('/calculate_change_dna_length_statistics', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            setVisibilityLoader(false, loaderID);
            showMessage(1, data.message);
        })
        .catch(error => {
            setVisibilityLoader(false, loaderID);
            console.error('Error:', error);
            showMessage(3, error.message);
        });
}

function calculateEventSimulationStatistics(method) {
    const dnaLength = document.getElementById('dnaLength');
    const repetitionCount = document.getElementById('simulationsCount');
    const formData = new FormData();
    formData.append('dnaLength', dnaLength.value);
    formData.append('simulationsCount', repetitionCount.value);
    formData.append('method', method);


    const loaderID = getLoader();
    setVisibilityLoader(true, loaderID);

    fetch('/calculate_event_simulation_statistics', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            setVisibilityLoader(false, loaderID);
            showMessage(1, data.message);
        })
        .catch(error => {
            setVisibilityLoader(false, loaderID);
            console.error('Error:', error);
            showMessage(3, error.message);
        });
}


function computeAverageDistance(method) {
    const dnaLength = document.getElementById('dnaLength');
    const branchLength = document.getElementById('branchLength');
    const repetitionCount = document.getElementById('repetitionCount');
    const formData = new FormData();
    formData.append('dnaLength', dnaLength.value);
    formData.append('branchLength', branchLength.value);
    formData.append('repetitionCount', repetitionCount.value);
    formData.append('method', method);

    const loaderID = getLoader();
    setVisibilityLoader(true, loaderID);

    fetch('/compute_average_distance', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            setVisibilityLoader(false, loaderID);
            showMessage(1, data.message);
        })
        .catch(error => {
            setVisibilityLoader(false, loaderID);
            console.error('Error:', error);
            showMessage(3, error.message);
        });
}

function getZipfianDistribution() {
    const argumentZipfAlpha = document.getElementById('argumentZipfAlpha');
    const simulationsCount = document.getElementById('simulationsCount');
    const resultRowsNumber = document.getElementById('resultRowsNumber');
    const formData = new FormData();
    formData.append('argumentZipfAlpha', argumentZipfAlpha.value);
    formData.append('simulationsCount', simulationsCount.value);
    formData.append('resultRowsNumber', resultRowsNumber.value);

    const loaderID = getLoader();
    setVisibilityLoader(true, loaderID);

    fetch('/get_zipfian_distribution', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            setVisibilityLoader(false, loaderID);
            showMessage(1, data.message)
        })
        .catch(error => {
            setVisibilityLoader(false, loaderID);
            console.error('Error:', error);
            showMessage(3, error.message)
        });
}

function showMessage(variant = 1, message = null) {
    let mode = [];
    if (variant === 1) {
        if (message === null) {message = 'YES'}
        mode = ['hidden', 'hidden', 'visible', 'hidden', 'hidden', message];
    } else if (variant === 2) {
        if (message === null) {message = 'NO'}
        mode = ['hidden', 'visible', 'hidden', 'hidden', 'hidden', message];
    } else if (variant === 3) {
        mode = ['visible', 'hidden', 'hidden', 'hidden', 'hidden', message];
    } else if (variant === 4) {
        mode = ['hidden', 'hidden', 'hidden', 'visible', 'hidden', message];
    } else if (variant === 5) {
        mode = ['hidden', 'hidden', 'hidden', 'hidden', 'visible', message];
    }
    document.getElementById('divWarning').style.visibility = mode[0];
    document.getElementById('divWarning').innerHTML = mode[5];
    document.getElementById('divDanger').style.visibility = mode[1];
    document.getElementById('divDanger').innerHTML = mode[5];
    document.getElementById('divInfo').style.visibility = mode[2];
    document.getElementById('divInfo').innerHTML = mode[5];
    document.getElementById('divSuccess').style.visibility = mode[3];
    document.getElementById('divSuccess').innerHTML = mode[5];
    document.getElementById('divSecondary').style.visibility = mode[4];
    document.getElementById('divSecondary').innerHTML = mode[5];
}