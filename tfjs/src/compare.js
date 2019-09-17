const test = require('../test.json');
const test2 = require('../../model/output.json');

const tf = require('@tensorflow/tfjs-node');

function tensorsEqual(t1, t2) {
  return v1.shape.reduce((a,b) => a *= b) === v1.equal(v2).sum().dataSync()[0];
}

v1 = tf.tensor(test.h);
v2 = tf.tensor(test2.h);

console.log(v1.shape);
console.log(v2.shape);

const v1arr = v1.arraySync()[0];
const v2arr = v2.arraySync()[0];
for (let i = 0 ; i < v1arr.length; i++) {
  if (Math.abs(v1arr[i] - v2arr[i]) > 0.001) {
    console.log(i, v1arr[i], v2arr[i]);
  }
}
console.log(tensorsEqual(v1, v2));
