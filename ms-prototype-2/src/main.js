/**
 * Based on
 * https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/5.4-visualizing-what-convnets-learn.ipynb
 */

 import * as argparse from "argparse";
 import * as fs from "fs";
 import * as path from "path";
 import * as shelljs from "shelljs"
 import * as tf from "@tensorflow/tfjs";


 /**
 * Calcuate and save the maximally-activating input images for a covn2d layer.
 */

 const CONFIG = {
     layerNames: "block1_conv1,block2_conv1,block3_conv2,block4_conv2,block5_conv3",
     filters: 8,
     inputImage: "",
     outputDir: "dist/activation"
 }

 async function processing(model, inputTensor) {

    const x = inputTensor;
    const { layerNames, filters, outputDir } = CONFIG;

    const layerNames = args.convLayerNames.split(',');

    await filters.writeInternalActivationAndGetOutput(
        model, layerNames, x, filters, outputDir);
 }

 module.exports = {
    processing
  };
