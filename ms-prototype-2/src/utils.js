/**
 * Tensorflow.js Examples Utility functions for the visualize-convnet
 */

//import * as fs from "fs";
//var jimp = require('jimp');

import * as jimp from 'jimp';
import * as tf from '@tensorflow/tfjs';

var fs = require('fs');
const Jimp = require('jimp/browser/lib/jimp');


BrowserFS.install(window);
  // Configures BrowserFS to use the LocalStorage file system.
  BrowserFS.configure({
    fs: "LocalStorage"
  }, function(e) {
    if (e) {
      // An error happened!
      throw e;
    }
    // Otherwise, BrowserFS is ready-to-use!
  });

/**
 * Read an image file as a TensorFlow.js tensor.
 * 
 */
export async function readImageTensorFromFile(filePath, height, width) {
    return new Promise((resolve, reject) => {
      jimp.read(filePath, (err, image) => {
        if (err) {
          reject(err);
        } else {
          const h = image.bitmap.height;
          const w = image.bitmap.width;
          const buffer = tf.buffer([1, h, w, 3], 'float32');
          image.scan(0, 0, w, h, function(x, y, index) {
            buffer.set(image.bitmap.data[index], 0, y, x, 0);
            buffer.set(image.bitmap.data[index + 1], 0, y, x, 1);
            buffer.set(image.bitmap.data[index + 2], 0, y, x, 2);
          });
          resolve(tf.tidy(
              () => tf.image.resizeBilinear(buffer.toTensor(), [height, width])));
        }
      });
    });
  }

  /**
 * Write an image tensor to a image file.
 */

export async function writeImageTensorToFile(imageTensor, filePath, row) {
  //console.log(filePath)
    const imageH = imageTensor.shape[1];
    const imageW = imageTensor.shape[2];
    const imageData = imageTensor.dataSync();
  
    const bufferLen = imageH * imageW * 4;
    const buffer = new Uint8Array(bufferLen);
    let index = 0;
    for (let i = 0; i < imageH; ++i) {
      for (let j = 0; j < imageW; ++j) {
        const inIndex = 3 * (i * imageW + j);
        buffer.set([Math.floor(imageData[inIndex])], index++);
        buffer.set([Math.floor(imageData[inIndex + 1])], index++);
        buffer.set([Math.floor(imageData[inIndex + 2])], index++);
        buffer.set([255], index++);
      }
    }

    return new Promise((resolve, reject) => {
  
      new jimp(
          {data: new Buffer(buffer, 'base64'), width: imageW, height: imageH},
          (err, img) => {

            if (err) {
              reject(err);

            } else {
              
              
              img.getBase64(Jimp.AUTO, function (err, src) {

                
                var img = document.createElement("img");
                img.setAttribute("src", src);
                row.appendChild(img);

                //fs.writeFile(filePath, src, function(err){ console.log(err)})
              });
              

              //img.write(filePath);
              resolve();
            }
          });
    });
  }