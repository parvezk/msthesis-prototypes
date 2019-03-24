/**
 * Algorithms for analyzing and visualizing the convolutional filters
 * internal to a convnet.
 * 
 * 1. Retrieving internal activations of a convnet.
 * See function `writeInternalActivationAndGetOutput()`.
 **/

import * as path from "path";
import * as tf from "@tensorflow/tfjs";
import { readImageTensorFromFile, writeImageTensorToFile} from "./utils";

export async function writeInternalActivationAndGetOutput(
    model, layerNames, inputImage, numFilters, outputDir, activationsDiv) {
    const layerName2FilePaths = {};
    const layerName2ImageDims = {};
    const layerOutputs = layerNames.map(layerName => model.getLayer(layerName).output);

    // Construct a model that returns all the desired internal activations,
    // in addition to the final output of the original model.
    const compositeModel = tf.model({
        inputs: model.input,
        outputs: layerOutputs.concat(model.outputs[0])
    });
    
    // `outputs` is an array of `tf.Tensor`s consisting of the internal-activation
    // values and the final output value.
    const outputs = compositeModel.predict(inputImage);

    for (let i = 0; i < outputs.length - 1; ++i) {
        const layerName = layerNames[i];
        // Split the activation of the convolutional layer by filter.

        var row = document.createElement("div");
        row.classList.add('activations-row');
        activationsDiv.appendChild(row);
    
        //console.log('SPLIT');
        const activationTensors = tf.split(outputs[i], outputs[i].shape[outputs[i].shape.length - 1], -1);
        //console.log(activationTensors.length, Array.isArray(activationTensors))
        //console.log(activationTensors[0])
    
        const actualNumFilters = numFilters <= activationTensors.length ? numFilters : activationTensors.length;
        const filePaths = [];
        let imageTensorShape;
    
        for (let j = 0; j < actualNumFilters; ++j) {
          // Format activation tensors and write them to disk.
          const imageTensor = tf.tidy(() => deprocessImage(tf.tile(activationTensors[j], [1, 1, 1, 3])));
          const outputFilePath = path.join(outputDir, `${layerName}_${j + 1}.png`);
          filePaths.push(outputFilePath);
          writeImageTensorToFile(imageTensor, outputFilePath, row);
          imageTensorShape = imageTensor.shape;

          
        }
        layerName2FilePaths[layerName] = filePaths;
        layerName2ImageDims[layerName] = imageTensorShape.slice(1, 3);
        tf.dispose(activationTensors);
      }
      tf.dispose(outputs.slice(0, outputs.length - 1));

      return {
        modelOutput: outputs[outputs.length - 1],
        layerName2FilePaths,
        layerName2ImageDims
      };
}

/** Center and scale input image so the pixel values fall into [0, 255]. */
function deprocessImage(x) {
    return tf.tidy(() => {
      const {
        mean,
        variance
      } = tf.moments(x);
      x = x.sub(mean);
      // Add a small positive number (EPSILON) to the denominator to prevent
      // division-by-zero.
      x = x.div(tf.sqrt(variance).add(tf.ENV.get('EPSILON')));
      // Clip to [0, 1].
      x = x.add(0.5);
      x = tf.clipByValue(x, 0, 1);
      x = x.mul(255);
      return tf.clipByValue(x, 0, 255).asType('int32');
    });
  }