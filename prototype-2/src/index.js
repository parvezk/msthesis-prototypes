import "./styles.scss";
import "babel-polyfill";

import * as tf from '@tensorflow/tfjs-node';
import * as tfvis from '@tensorflow/tfjs-vis';
import * as imageNetClasses from "./imagenet_classes";
import { internalActivations, ClassActivationMaps } from "./main.js"
import WebCam from './webcam';

/** INDEX.JS **/


const imageElem = document.querySelector('#image-container');
const videoElem = document.querySelector('.video-option');
const webcamBtn = document.querySelector('input[id="webcam-btn"]');
const webcamElement = document.getElementById('video');


const progressBar1 = document.querySelector('#progress-bar-1');
const progressBar2 = document.querySelector('#progress-bar-2');
const loaderBox = document.querySelector('.loader-box');

let model, tensor, mediaTensor;

function getProcessedTensor(media) {
    tensor = tf.browser.fromPixels(media);
    tensor = tensor.resizeNearestNeighbor([224, 224]).toFloat();

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

let top5, IMAGENET_CLASSES;
IMAGENET_CLASSES = imageNetClasses.IMAGENET_CLASSES

function showPredictions(predictions) {
    top5 = Array.from(predictions)
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
        len = children.length;

    for (var i = 0; i < len; i++) {
        const text = children[i].children[1].children;
        const prob = top5[i].probability.toFixed(6);
        children[i].children[0].setAttribute('style', 'width: ' + (Math.round(prob*100)).toString() + '%');
        text[0].innerHTML = top5[i].className.split(',')[0]
        text[1].innerHTML = Math.floor(prob * 100) / 100;
    }
}

// Generate Internal Activations
async function getActivations() {
    console.log('Loading activations..')
    const activationsDiv = document.querySelector('#activations');
    progressBar2.classList.remove("hide");
    activationsDiv.innerHTML = '';
    if (model && tensor) {
        //await internalActivations(model, tensor, activationsDiv);
        //progressBar2.classList.add("hide");
    }
}

// Generate Activation map on input image
async function getActivationMaps() {
    console.log('Loading heatmap..');
    
    const camDiv = document.querySelector('#cam');
    
    const tensorData = mediaTensor || tensor;
    console.log(tensorData.shape);
    camDiv.innerHTML = '';
    if (model && tensorData) {
        await ClassActivationMaps(model, tensorData, top5, camDiv);
        loaderBox.classList.add("hide");
    }
    if (mediaTensor) mediaTensor.dispose();
    tensor.dispose();
    tensorData.dispose();
}

// EVENT HANDLERS
function setupListeners() {
    document.querySelector('#show-metrics')
        .addEventListener('click', showModel);

    document.querySelector('#activation-btn')
        .addEventListener('click', getActivations);

    document.querySelector('#heatmap-btn')
        .addEventListener('click', async function(){
            loaderBox.classList.remove("hide");
            setTimeout(() => {
                getActivationMaps();
            }, 500)
        });

    webcamBtn.addEventListener('click', videoOption);
}

async function run() {

}

document.querySelector('#image-selector')
.addEventListener('change', (e) => {

    let reader = new FileReader();
    reader.onload = function () {
        let dataURL = reader.result;
        imageElem.setAttribute("src", dataURL);
    }
    let file = $("#image-selector").prop("files")[0];
    reader.readAsDataURL(file);
});


document.querySelector('#model-selector')
.addEventListener('change', () => {
    loadModel($("#model-selector").val());
});

async function loadModel(name) {
    console.log(name)
    progressBar1.classList.remove("hide");
    model = undefined;
    model = await tf.loadLayersModel(`./tfjs-models/${name}/model.json`);
    progressBar1.classList.add("hide");
}

$('#predict-button').click(async function () {
    progressBar1.classList.remove("hide");
    let image = $('#selected-image').get(0);
    //.expandDims();
    let processedTensor = getProcessedTensor(image);
    let predictions = await model.predict(processedTensor).data();

    //testVis(tensor1);
    showPredictions(predictions);
    progressBar1.classList.add("hide");

    //ENABLE BUTTONS
    //tensor.dispose();
    processedTensor.dispose();
});

async function videoOption() {
    imageElem.classList.add('hide');
    videoElem.classList.remove('hide');

    if (this.checked)
    {
        const webcam = new WebCam();
        await webcam.setupWebcam(webcamElement);

        while (true) {
            let mediaTensor = getProcessedTensor(webcamElement);
            let predictions = await model.predict(mediaTensor).data();
            showPredictions(predictions);
            await tf.nextFrame();
        }
    } else {
        videoElem.classList.add('hide');
        imageElem.classList.remove('hide');
    }
}

document.getElementById('capture')
.addEventListener('click', () => {
    const canvas = document.getElementById('webcam-frame');
    const context = canvas.getContext('2d');
    // Draw the video frame to the canvas.
    context.drawImage(webcamElement, 0, 0, canvas.width, canvas.height);
  });


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


