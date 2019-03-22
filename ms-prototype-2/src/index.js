import "./styles.scss";
import "babel-polyfill";

import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

console.log(tf.version);

$("#image-selector").change(function () {
    let reader = new FileReader();
    reader.onload = function () {
        let dataURL = reader.result;
        $("#selected-image").attr("src", dataURL);
        $("#prediction-list").empty();
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

$('#predict-button').click(async function() {
    let image = $('#selected-image').get(0);
    let modelName = $("#model-selector").val();

    let origTensor = tf.browser.fromPixels(image);
    origTensor = origTensor.resizeNearestNeighbor([100, 100]);
    let tensor = origTensor.resizeNearestNeighbor([224, 224]).toFloat();
    //.expandDims();

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

    let predictions = await model.predict(processedTensor).data();
    let top5 = Array.from(predictions)
        .map(function (p, i) {
            return {
                probability: p,
                className: IMAGENET_CLASSES[i]
            };
        }).sort(function (a, b) {
            return b.probability - a.probability;
        }).slice(0, 5);
    
    $('#prediction-list').empty();
    top5.forEach((p) => {
        $('#prediction-list').append(`<li>${p.className}: ${p.probability.toFixed(6)}</li>`);
    });

    // Get a surface
    const surface = tfvis.visor().surface({ name: 'Surface', tab: 'Image from Tensor' });
    const drawArea = surface.drawArea;

    const canvas = document.createElement('canvas');
    canvas.getContext('2d');
    canvas.width = tensor[0];
    canvas.height = tensor[1];
    canvas.style = 'margin: 4px;';
    await tf.browser.toPixels(origTensor, canvas);
    drawArea.appendChild(canvas);

    tensor.dispose();
    processedTensor.dispose();
})

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


document.querySelector('#show-metrics')
    .addEventListener('click', showModel);

function setupListeners() {
    
}




function setup () {
    setupListeners();
}

function run() {
}

// EVENT HANDLERS
document
.addEventListener('DOMContentLoaded', () => {
    setup();
    run();
});

modelSelect.addEventListener('change', run);