import "./styles.scss";
import "babel-polyfill";
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

import { MnistData } from './data';
import { createConvModel } from './model';
import { plotLoss, plotAccuracy } from './vis';
import {setupListeners } from './events';

async function train(model, onIteration) {
    
    // display metrics
    const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
    const container = {
        name: 'Model Training',
        styles: {
            height: '1000px'
        }
    };
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

    const LEARNING_RATE = 0.15;
    const optimizer = tf.train.sgd(LEARNING_RATE);

    model.compile({
        optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });

    const batchSize = 320;
    const validationSplit = 0.15;
    const trainEpochs = 1;
    let trainBatchCount = 0;

    const trainData = data.getTrainData();
    const testData = data.getTestData();

    const totalNumBatches = 
    Math.ceil(trainData.xs.shape[0] * (1 - validationSplit) / batchSize) * trainEpochs;

    let valAcc;
    let saveResults;
    await model.fit(trainData.xs, trainData.labels, {
        batchSize,
        validationSplit,
        epochs: trainEpochs,
        //callbacks: fitCallbacks
        callbacks: {

            onTrainBegin: async () => {
                console.log('START')
            },

            onBatchEnd: async (batch, logs) => {
                console.log(batch, logs)
                trainBatchCount++;
                plotLoss(trainBatchCount, logs.loss, 'train');
                plotAccuracy(trainBatchCount, logs.acc, 'train');
                if (onIteration && batch % 10 === 0) {
                    onIteration('onBatchEnd', batch, logs);
                }
                await tf.nextFrame();
            },

            onEpochEnd: async (epoch, logs) => {
                valAcc = logs.val_acc;
                plotLoss(trainBatchCount, logs.val_loss, 'validation');
                plotAccuracy(trainBatchCount, logs.val_acc, 'validation');
                if (onIteration) {
                    onIteration('onEpochEnd', epoch, logs);
                }
                await tf.nextFrame();
            },

            onTrainEnd: async () => {
                console.log('THE END')
                saveResults = await model.save('localstorage://my-model-1');
                const testResult = model.evaluate(testData.xs, testData.labels);
                const testAccPercent = testResult[1].dataSync()[0] * 100;
                const finalValAccPercent = valAcc * 100;
                console.log('Final validation accuracy', finalValAccPercent.toFixed(1))
                console.log('Final test accuracy', testAccPercent.toFixed(1));
            }
        }
    });
}


let model;

function createModel() {
    model = createConvModel();
    return model;
}

let data;
async function load() {
    data = new MnistData();
    await data.load();
}

async function run() {
    setupListeners(
        async (trainButton) => {
                console.log('Loading MNIST data...');
                await load();
                console.log('success data');
                trainButton.removeAttribute("disabled");
                //miniExamples = data.nextTestBatch(10);
                window.miniExamples = data.getTestData(10);
            },
            async () => {
                console.log('Creating model...');
                const model = createModel();
                model.summary();

                console.log('Starting model training...');
                await train(model, () => showPredictions(model));
            }
    );
}

document.addEventListener('DOMContentLoaded', () => {
    run();
});

let examples;
async function showPredictions(model) {
    const testExamples = 100;
    examples = data.getTestData(testExamples);
    tf.tidy(() => {
        const output = model.predict(examples.xs);
        const axis = 1;
        const labels = Array.from(examples.labels.argMax(axis).dataSync());
        const predictions = Array.from(output.argMax(axis).dataSync());

        showTestResults(examples, predictions, labels);
    });
}

const statusElement = document.getElementById('status');
const messageElement = document.getElementById('message');
const imagesElement = document.getElementById('images');

function logStatus(message) {
    statusElement.innerText = message;
}

function trainingLog(message) {
    messageElement.innerText = `${message}\n`;
    console.log(message);
}

function showTestResults(batch, predictions, labels) {
    const testExamples = batch.xs.shape[0];
    imagesElement.innerHTML = '';

    for (let i = 0; i < testExamples; i++) {
        const image = batch.xs.slice([i, 0], [1, batch.xs.shape[1]]);

        const div = document.createElement('div');
        div.className = 'pred-container';

        const canvas = document.createElement('canvas');
        canvas.className = 'prediction-canvas';
        draw(image.flatten(), canvas);

        const pred = document.createElement('div');

        const prediction = predictions[i];
        const label = labels[i];
        const correct = prediction === label;

        pred.className = `pred ${(correct ? 'pred-correct' : 'pred-incorrect')}`;
        pred.innerText = `pred: ${prediction}`;

        div.appendChild(pred);
        div.appendChild(canvas);

        imagesElement.appendChild(div);
    }
}

function draw(image, canvas) {
    const [width, height] = [28, 28];
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const imageData = new ImageData(width, height);
    const data = image.dataSync();
    for (let i = 0; i < height * width; ++i) {
        const j = i * 4;
        imageData.data[j + 0] = data[i] * 255;
        imageData.data[j + 1] = data[i] * 255;
        imageData.data[j + 2] = data[i] * 255;
        imageData.data[j + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
}