import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

import {
    getActivationTable,
    renderImageTable
} from './vis';

export function setupListeners(callback1, callback2) {

    const showMetrics = document.querySelector('#metrics');
    const trainButton = document.getElementById('train');
    const loadButton = document.getElementById('load');

    trainButton.setAttribute('disabled', true);
    loadButton.addEventListener('click', callback1(trainButton))

    trainButton.addEventListener('click', callback2);

    showMetrics.addEventListener('click', () => {
        const visorInstance = tfvis.visor();
        if (!visorInstance.isOpen()) {
            visorInstance.toggle();
        }
    });

    document.querySelector('#load-model').addEventListener('click', async () => {
        // Load saved model
        window.loadedModel = await tf.loadModel('localstorage://my-model-1');
        console.log('Loaded Model....')
        loadedModel.summary();
    });

    // CONV2D1 IMAGE
    const conv1 = document.querySelector('#conv1');
    conv1.addEventListener('click', async () => {

        const surface = tfvis.visor().surface({
            name: 'Conv2D1 Activations',
            tab: 'Activation',
            styles: {
                height: 650,
            },
        });

        const drawArea = surface.drawArea;
        console.log(loadedModel)
        const {
            filters,
            filterActivations
        } = getActivationTable(loadedModel, miniExamples, 'conv2d_Conv2D1');

        renderImageTable(drawArea, filters, filterActivations);
    });

    // CONV2D1 DETAIL
    const conv2 = document.querySelector('#conv2');
    conv2.addEventListener('click', async () => {

        const surface = tfvis.visor().surface({
            name: 'Conv2D2 Activations',
            tab: 'Activation',
            styles: {
                width: 1000,
                height: 650,
            },
        });

        const drawArea = surface.drawArea;
        const {
            filters,
            filterActivations
        } = getActivationTable(loadedModel, miniExamples, 'conv2d_Conv2D2');

        renderImageTable(drawArea, filters, filterActivations);
    });

}