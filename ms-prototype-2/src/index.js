import "./styles.scss";
import "babel-polyfill";

import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import {
    internalActivations, ClassActivationMaps
} from "./main.js"

/** INDEX.JS **/

$("#image-selector").change(function () {
    let reader = new FileReader();
    reader.onload = function () {
        let dataURL = reader.result;
        $("#selected-image").attr("src", dataURL);
    }
    let file = $("#image-selector").prop("files")[0];
    reader.readAsDataURL(file);
});

const modelSelect = document.querySelector('#model-selector');
$("#model-selector").change(function () {
    loadModel($("#model-selector").val());
});

let progressBar = document.querySelector('.progress-bar');
let model;
async function loadModel(name) {
    console.log(name)
    progressBar.classList.remove("hide");
    model = undefined;
    model = await tf.loadLayersModel(`./tfjs-models/${name}/model.json`);
    progressBar.classList.add("hide");
}

let tensor, tensor1;
$('#predict-button').click(async function () {
    let image = $('#selected-image').get(0);

    tensor = tf.browser.fromPixels(image);
    tensor1 = tensor.resizeNearestNeighbor([100, 100]);
    tensor = tensor.resizeNearestNeighbor([224, 224]).toFloat();
    //.expandDims();
    let processedTensor = getProcessedTensor(tensor);
    let predictions = await model.predict(processedTensor).data();

    //testVis(tensor1);
    showPredictions(predictions);

    //ENABLE BUTTONS

    //tensor.dispose();
    //processedTensor.dispose();
});

function getProcessedTensor(tensor) {
    // More pre-processing
    let meanImageNetRGB = {
        red: 123.68,
        green: 116.779,
        blue: 103.939
    };

    let indices = [
        tf.tensor1d([0], "int32"),
        tf.tensor1d([1], "int32"),
        tf.tensor1d([2], "int32")
    ];

    // Centering the RGB values
    let centeredRGB = {
        red: tf.gather(tensor, indices[0], 2)
            .sub(tf.scalar(meanImageNetRGB.red))
            .reshape([50176]),
        green: tf.gather(tensor, indices[1], 2)
            .sub(tf.scalar(meanImageNetRGB.green))
            .reshape([50176]),
        blue: tf.gather(tensor, indices[2], 2)
            .sub(tf.scalar(meanImageNetRGB.blue))
            .reshape([50176])
    };
    // Stacking, reversing, and reshaping
    let processedTensor = tf.stack([
            centeredRGB.red, centeredRGB.green, centeredRGB.blue
        ], 1)
        .reshape([224, 224, 3])
        .reverse(2)
        .expandDims();

    return processedTensor;
}

function showPredictions(predictions) {
    let top5 = Array.from(predictions)
        .map(function (p, i) {
            return {
                probability: p,
                className: IMAGENET_CLASSES[i]
            };
        }).sort(function (a, b) {
            return b.probability - a.probability;
        }).slice(0, 5);

    var el = document.querySelector('#prediction-list');

    var children = el.children,
        number_of_children = children.length;

    for (var i = 0; i < number_of_children; i++) {
        const text = children[i].children[1].children;
        text[0].innerHTML = top5[i].className;
        text[1].innerHTML = top5[i].probability.toFixed(6);
    }
}

// Generate Internal Activations
function getActivations() {
    console.log('ACTIVATIONS')
    const activationsDiv = document.querySelector('#activations');
    activationsDiv.innerHTML = '';
    if (model && tensor)
        internalActivations(model, tensor, activationsDiv);
}

// Generate Activation map on input image
function getActivationMaps() {
    console.log('MAPS')
    const camDiv = document.querySelector('#cam');
    camDiv.innerHTML = '';
    if (model && tensor)
        ClassActivationMaps(model, tensor, camDiv);
}


function setupListeners() {
    console.log('CALLED')
    document.querySelector('#show-metrics')
        .addEventListener('click', showModel);

    document.querySelector('#activation-btn')
        .addEventListener('click', getActivations);

    document.querySelector('#heatmap-btn')
        .addEventListener('click', getActivationMaps);
}

function run() {}

async function testVis() {
    // Get a surface
    const surface = tfvis.visor().surface({
        name: 'Surface',
        tab: 'Image from Tensor'
    });
    const drawArea = surface.drawArea;

    const canvas = document.createElement('canvas');
    canvas.getContext('2d');
    canvas.width = origTensor[0];
    canvas.height = origTensor[1];
    canvas.style = 'margin: 4px;';
    await tf.browser.toPixels(origTensor, canvas);
    drawArea.appendChild(canvas);
}

async function showModel() {

    const visorInstance = tfvis.visor();
    //console.log(visorInstance)
    if (!visorInstance.isOpen()) {
        visorInstance.toggle();
    }

    const surface = {
        name: 'Model Summary',
        tab: 'Model'
    };
    tfvis.show.modelSummary(surface, model);
}

// EVENT HANDLERS
document
    .addEventListener('DOMContentLoaded', () => {
        setupListeners();
        run();
    });

modelSelect.addEventListener('change', run);