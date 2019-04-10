import * as tf from '@tensorflow/tfjs';

const webcamBtn = document.querySelector('input[id="webcam"]');
const webcamElement = document.getElementById('webcam');

webcamBtn.addEventListener('click', async function(e) {
    if (this.checked)
        await setupWebcam();
    
});

async function setupWebcam() {
    return new Promise((resolve, reject) => {
      const navigatorAny = navigator;
      navigator.getUserMedia = navigator.getUserMedia ||
          navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia ||
          navigatorAny.msGetUserMedia;
      if (navigator.getUserMedia) {
        navigator.getUserMedia({video: true},
          stream => {
            webcamElement.srcObject = stream;
            webcamElement.addEventListener('loadeddata',  () => resolve(), false);
          },
          error => reject());
      } else {
        reject();
      }
    });
  }