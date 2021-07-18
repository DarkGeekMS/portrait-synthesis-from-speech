// TensorFlow.js for Node,js
const tf = require('@tensorflow/tfjs-node');
import * as tf from '@tensorflow/tfjs';


// train and validation data
const trainDataUrl = '';
const testDataUrl = '';

// hyper parameters
const batchSize = 16;
const epochsNumber = 5;

// data preprocessing


// ctc loss function
export function ctcLoss(
  /** 
   * batch_size / max_label_seq_length 
   */
  labels: tf.Tensor2D,
  /**
   * batch_size / frames / num_labels
   */
  logits: tf.Tensor3D,
  options: {} = {} ): tf.Tensor1D { return tf.tidy(() => {
                                                            const vec = ctcLoss_vec(labels, logits, options);
                                                            return vec.mean().neg() as tf.Tensor1D;
                                                          })
                                  }

export function ctcLoss_vec(
  labels: tf.Tensor2D,
  logits: tf.Tensor3D,
  options: {} = {}): tf.Tensor2D { function p(name: string, t?: tf.Tensor) { return; if (t)  printTensor(name, t); else console.log(name); }
  return tf.tidy(() => {
      const SUPER_SMALL = -1e9; //-1e38;

      const logits_normalized = logits.sub(logits.logSumExp(2, true));

      p('labels', labels);
      p('logits', logits);
      p('logits_normalized', logits_normalized);

      const [batch_size, num_time_steps] = logits.shape;
      if (labels.shape[0] != batch_size) ASC.dieError('1694042736');
      const max_label_seq_length = labels.shape[1];
      const y0_size = max_label_seq_length * 2 + 1;
      const y_size = y0_size + 1;

      p('labels', labels);
      p('logits', logits);
      p('logits_normalized', logits_normalized);
      p('max_label_seq_length=' + max_label_seq_length);
      p('y0_size=' + y0_size);

      const labels_buff = labels.bufferSync();
      const y_loc_buff = tf.buffer([batch_size, y0_size, 2], 'int32');
      const res_loc_buff = tf.buffer([batch_size, 2], 'int32');

      for (let b = 0; b < batch_size; b++) {
          for (let y = 0; y < y0_size; y++)
              y_loc_buff.set(b, b, y, 0);
          let len = max_label_seq_length;
          for (let t = 0; t < max_label_seq_length; t++) {
              const l = labels_buff.get(b, t);
              if (l == 0) {
                  len = t;
                  break;
              }
              const y = t * 2 + 1;
              y_loc_buff.set(l, b, y, 1);
          }

          res_loc_buff.set(b, b, 0);
          res_loc_buff.set(2 * len, b, 1);
      }

      const incoming_loc_buff = tf.buffer([batch_size, y_size, 3, 2], 'int32');
      for (let b = 0; b < batch_size; b++) {
          for (let y = 0; y <= y0_size; y++) {
              for (let i = 0; i < 3; i++)
                  incoming_loc_buff.set(b, b, y, i, 0);
          }
          for (let y = 0; y < y0_size; y++) {
              incoming_loc_buff.set(y, b, y, 0, 1);
              incoming_loc_buff.set(y > 0 ? y - 1 : y0_size, b, y, 1, 1);
              let moze_double = false;
              if (y > 2) {
                  if (y_loc_buff.get(b, y, 1) != y_loc_buff.get(b, y - 2, 1))
                      moze_double = true;
              }
              incoming_loc_buff.set(moze_double ? y - 2 : y0_size, b, y, 2, 1);
          }
          incoming_loc_buff.set(y0_size, b, y0_size, 0, 1);
          incoming_loc_buff.set(y0_size, b, y0_size, 1, 1);
          incoming_loc_buff.set(y0_size, b, y0_size, 2, 1);
      }


      const y_loc = y_loc_buff.toTensor();
      const res_loc = res_loc_buff.toTensor();
      const incoming_loc = incoming_loc_buff.toTensor();
      p('y_loc', y_loc);
      p('res_loc', res_loc);
      p('incoming_loc', incoming_loc);

      const y0 = gatherND(
          logits_normalized.transpose([0, 2, 1]),
          y_loc,
      )
          .transpose([0, 2, 1]);

      const y = y0.pad([[0, 0], [0, 0], [0, 1]], SUPER_SMALL);

      p('y', y);

      let log_alpha =
          tf.scalar(0).reshape([1, 1]).tile([batch_size, 1])
              .concat(
                  tf.scalar(SUPER_SMALL).reshape([1, 1]).tile([batch_size, y0_size]),
                  1
              );

      function shift(t: tf.Tensor) {
          return t.pad([[0, 0], [1, 0]], SUPER_SMALL).slice([0, 0], t.shape);
      }
      function logSumExp(a: tf.Tensor, b: tf.Tensor) {
          return a.expandDims(2).concat(b.expandDims(2), 2).logSumExp(2);
      }

      const t2y = y.unstack(1);
      for (let t = 0; t < num_time_steps; t++) {
          p("Time: " + t);

          const ty = t2y[t];
          p('log_alpha', log_alpha);
          p('ty', ty);

          const incoming = gatherND(log_alpha, incoming_loc);
          p('incoming', incoming);

          const incoming_plus_ty = incoming.add(ty.expandDims(2));
          p('incoming_plus_ty', incoming_plus_ty);

          const new_log_alpha2 = incoming_plus_ty.logSumExp(2);
          p('new_log_alpha2', new_log_alpha2);

          log_alpha = new_log_alpha2;
      }

      const log_alpha_final = logSumExp(log_alpha, shift(log_alpha));
      p('log_alpha_final', log_alpha_final);

      const vec = gatherND(log_alpha_final, res_loc);
      //printTensor('vec', vec);

      return vec as tf.Tensor2D;
  })
}

function gatherND(x: tf.Tensor, indices: tf.Tensor): tf.Tensor {
  const grad = (dy: tf.Tensor, saved: tf.Tensor[]) => {
      return { x: () => tf.scatterND(saved[0], dy, x.shape) }
  }
  return ENGINE.runKernel(
      (backend, save) => {
          save([indices]);
          return backend.gatherND(x, indices);
      },
      { x },
      grad
  ) as
      tf.Tensor;
}


// model creation
const buildModel = function (inputDim) {
    const model = tf.sequential();

    model.add(tf.input({shape: inputDim}));

    model.add(tf.layers.gru({
        units: 250, 
        returnSequences: true,
        activation: 'relu'
    }));
 
    model.add(tf.layers.gru({
        units: 200, 
        returnSequences: true,
        activation: 'relu'
    }));

    model.add(tf.layers.batchNormalization({}));

    model.add(tf.layers.timeDistributed({
        layer: tf.layers.dense({units: 29}),
    }));

    model.add(tf.layers.activation({activation: 'softmax'}));
    // compile the model
    model.compile({
      optimizer: tf.train.adam(0.001),          // large learning rate
      loss: ctc_loss(),
      metrics: ['accuracy']
    });
  
    return model;
};

// train function
const trainModel = async function (model, trainingData, epochs=epochsNumber) {
    const options = {
      epochs: epochs,
      verbose: 1,
      callbacks: {
        onEpochBegin: async (epoch, logs) => {
          console.log(`Epoch ${epoch + 1} of ${epochs} ...`)
        },
        onEpochEnd: async (epoch, logs) => {
          console.log(`  train-set loss: ${logs.loss.toFixed(4)}`)
          console.log(`  train-set accuracy: ${logs.acc.toFixed(4)}`)
        }
      }
    };
  
    return await model.fitDataset(trainingData, options);
  };
  
// evaluation function
const evaluateModel = async function (model, validationData) {

    const result = await model.evaluateDataset(validationData);
    const validLoss = result[0].dataSync()[0];
    const validAcc = result[1].dataSync()[0];

    console.log(`  test-set loss: ${validLoss.toFixed(4)}`);
    console.log(`  test-set accuracy: ${validAcc.toFixed(4)}`);
};



// run
const run = async function () {
    const trainData = loadData(trainDataUrl);
    const valiDation = loadData(testDataUrl);
  
    // Full path to the directory to save the model in
    const saveModelPath = 'file://./home/monda/Documents/ForthYear/gp/tfjs_speech/tfjs_model/tfjs_model_saved';
    // build the model
    const model = buildModel();
    model.summary();
  
    // train the model
    const info = await trainModel(model, trainData);
    // print the training loss and accuracy
    console.log('\r\n', info);
    // evaluating the model
    console.log('\r\nEvaluating model...');
    await evaluateModel(model, valiDation);
    console.log('\r\nSaving model...');
    // save the model
    await model.save(saveModelPath);
  };
  
  run();