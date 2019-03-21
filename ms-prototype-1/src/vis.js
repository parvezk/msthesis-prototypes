import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

const lossLabelElement = document.getElementById('loss-label');
const accuracyLabelElement = document.getElementById('accuracy-label');
const lossValues = [
    [],
    []
];

export function plotLoss(batch, loss, set) {
    // Render to visor
    const container = {
        name: 'Loss',
        tab: 'Model Training'
    };
    const surface = tfvis.visor().surface(container);

    const series = set === 'train' ? 0 : 1;
    lossValues[series].push({
        x: batch,
        y: loss
    });

    tfvis.render.linechart({
        values: lossValues,
        series: ['train', 'validation']
    }, surface, {
        xLabel: 'Training Step',
        yLabel: 'Loss',
        width: 500,
        height: 180,
    });
    lossLabelElement.innerText = `last loss: ${loss.toFixed(3)}`;
}

const accuracyValues = [
    [],
    []
];

export function plotAccuracy(batch, accuracy, set) {
    // Render to visor
    const container = {
        name: 'Accuracy',
        tab: 'Model Training'
    };
    const surface = tfvis.visor().surface(container);

    const series = set === 'train' ? 0 : 1;
    accuracyValues[series].push({
        x: batch,
        y: accuracy
    });
    tfvis.render.linechart({
        values: accuracyValues,
        series: ['train', 'validation']
    }, surface, {
        xLabel: 'Batch #',
        yLabel: 'Accuracy',
        width: 500,
        height: 180,
    });
    accuracyLabelElement.innerText =
        `last accuracy: ${(accuracy * 100).toFixed(1)}%`;
}

// An 'activation' is the output of any of the internal layers of the
// network.
function getActivation(input, model, layer) {
    const activationModel = tf.model({
        inputs: model.input,
        outputs: layer.output,
    });

    return activationModel.predict(input);
}

// Render a tensor as an image on canvas object. We will render the
// activations from the convolutional layers.
async function renderImage(container, tensor, imageOpts) {
    const resized = tf.tidy(() =>
        tf.image.resizeNearestNeighbor(tensor,
            [imageOpts.height, imageOpts.width]).clipByValue(0.0, 1.0)
    );

    const canvas = container.querySelector('canvas') || document.createElement('canvas');
    canvas.width = imageOpts.width;
    canvas.height = imageOpts.height;
    canvas.style = `margin: 4px; width:${imageOpts.width}px; height:${imageOpts.height}px`;
    container.appendChild(canvas);
    await tf.toPixels(resized, canvas);
    resized.dispose();
}

// Render a table of images, we will show the output for each filter
// in the convolution.
export function renderImageTable(container, headerData, data) {
    let table = d3.select(container).select('table');
    if (table.size() === 0) {
        table = d3.select(container).append('table');
        table.append('thead').append('tr');
        table.append('tbody');
    }

    const headers = table.select('thead').select('tr').selectAll('th').data(headerData);
    const headersEnter = headers.enter().append('th')
    headers.merge(headersEnter).each((d, i, group) => {
        const node = group[i];
        if (typeof d == 'string') {
            node.innerHTML = d;
        } else {
            renderImage(node, d, {
                width: 25,
                height: 25
            });
        }
    });

    const rows = table.select('tbody').selectAll('tr').data(data);
    const rowsEnter = rows.enter().append('tr');

    const cells = rows.merge(rowsEnter).selectAll('td').data(d => d);
    const cellsEnter = cells.enter().append('td');
    cells.merge(cellsEnter).each((d, i, group) => {
        const node = group[i];
        renderImage(node, d, {
            width: 40,
            height: 40
        });
    })

    cells.exit().remove();
    rows.exit().remove();
}

export function getActivationTable(model, examples, layerName) {
    const exampleImageSize = 28;

    const layer = model.getLayer(layerName);

    // Get the filters
    let filters = tf.tidy(() => layer.kernel.val.transpose([3, 0, 1, 2]).unstack());
    // It is hard to draw high dimensional filters so we just use a string
    if (filters[0].shape[2] > 3) {
        filters = filters.map((d, i) => `Filter ${i + 1}`);
    }
    filters.unshift('Input');

    // Get the inputs
    const numExamples = examples.xs.shape[0];
    const xs = examples.xs.reshape([numExamples, exampleImageSize, exampleImageSize, 1]);

    // Get the activations
    const activations = tf.tidy(() => {
        return getActivation(xs, model, layer).unstack();
    });
    const activationImageSize = activations[0].shape[0]; // e.g. 24
    const numFilters = activations[0].shape[2]; // e.g. 8


    const filterActivations = activations.map((activation, i) => {
        // activation has shape [activationImageSize, activationImageSize, i];
        const unpackedActivations = Array(numFilters).fill(0).map((_, i) =>
            activation.slice([0, 0, i], [activationImageSize, activationImageSize, 1])
        );

        // prepend the input image
        const inputExample = tf.tidy(() =>
            xs.slice([i], [1]).reshape([exampleImageSize, exampleImageSize, 1]));

        unpackedActivations.unshift(inputExample);
        return unpackedActivations;
    });

    return {
        filters,
        filterActivations,
    };
}


// INFER FUNCTIONS
/*
// CONV2D1 DETAIL
async function showConv2d1() {
    const conv1Surface = {
        name: 'Conv2D1 Details',
        tab: 'Model'
    };
    tfvis.show.layer(conv1Surface, loadedModel.getLayer('conv2d_Conv2D1'));
}
document.querySelector('#show-conv1').addEventListener('click', showConv2d1);
*/

/*
// CONV2D2 DETAIL
async function showConv2d2() {
    const conv1Surface = { name: 'Conv2D2 Details', tab: 'Model' };
    tfvis.show.layer(conv1Surface, loadedModel.getLayer('conv2d_Conv2D2'));
}
document.querySelector('#show-conv2').addEventListener('click', showConv2d2);
*/