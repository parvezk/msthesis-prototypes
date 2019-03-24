/**
 * Based on
 * https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/5.4-visualizing-what-convnets-learn.ipynb
 */

 import * as tf from "@tensorflow/tfjs";
 import {writeInternalActivationAndGetOutput} from "./filters";
 import * as utils from "./utils";

 import * as cam from "./cam";
 import * as imagenetClasses from "./imagenet_classes";
 
 /**
 * Calcuate and save the maximally-activating input images for a covn2d layer.
 */

 const CONFIG = {
     layerNames: "block1_conv1,block2_conv1,block3_conv2,block4_conv2,block5_conv3",
     filters: 8,
     inputImage: "",
     outputDir: "activation"
 }

export async function internalActivations(model, inputTensor, activationsDiv) {

    const imageHeight = model.inputs[0].shape[1];
    const imageWidth = model.inputs[0].shape[2];
    const { filters, outputDir } = CONFIG;

    let x = inputTensor.resizeNearestNeighbor([imageHeight, imageWidth]);
    x = x.reshape([1, x.shape[0], x.shape[1], x.shape[2]]);
    //x = x.as4D([1, x.shape.length[0], x.shape.length[1], x.shape.length[2]]);
    const layerNames = CONFIG.layerNames.split(',');

    await writeInternalActivationAndGetOutput(
        model, layerNames, x, filters, outputDir, activationsDiv);
 }

 export async function ClassActivationMaps(model, inputTensor, camDiv) {

    // Compute the internal activations of the conv layers' outputs.
    const imageHeight = model.inputs[0].shape[1];
    const imageWidth = model.inputs[0].shape[2];
    let x = inputTensor.resizeNearestNeighbor([imageHeight, imageWidth]);
    x = x.reshape([1, x.shape[0], x.shape[1], x.shape[2]]);
    const layerNames = CONFIG.layerNames.split(',');

    const layerOutputs =
      layerNames.map(layerName => model.getLayer(layerName).output);

      // Construct a model that returns all the desired internal activations,
  // in addition to the final output of the original model.
  const compositeModel = tf.model(
    {inputs: model.input, outputs: layerOutputs.concat(model.outputs[0])});

    // `outputs` is an array of `tf.Tensor`s consisting of the internal-activation
    // values and the final output value.
    const outputs = compositeModel.predict(x);
    tf.dispose(outputs.slice(0, outputs.length - 1));
    const modelOutput = outputs[outputs.length - 1];

    // Calculate internal activations and final output of the model.
    const topNum = 10;
    const {values: topKVals, indices: topKIndices} =
        tf.topk(modelOutput, topNum);
  
    // Predictions
    const probScores = Array.from(await topKVals.data());
    const indices = Array.from(await topKIndices.data());
    const classNames =
        indices.map(index => imagenetClasses.IMAGENET_CLASSES[index]);

    console.log(`Top-${topNum} classes:`);
    for (let i = 0; i < topNum; ++i) {
      console.log(
          `  ${classNames[i]} (index=${indices[i]}): ` +
          `${probScores[i].toFixed(4)}`);
    }

    var container = document.createElement("div");
        container.classList.add('class-activation-map');
        camDiv.appendChild(container);

    // Calculate Grad-CAM heatmap.
    const xWithCAMOverlay = cam.gradClassActivationMap(model, indices[0], x);
    const camImagePath = 'dist/cam/cam.png';
    await utils.writeImageTensorToFile(xWithCAMOverlay, camImagePath, container);
    console.log(`Written CAM-overlaid image to: ${camImagePath}`);
 }