import React, {useEffect, useState} from 'react';
import logo from './logo.svg';
import './App.css';
// import dependancies 
import * as tf from "@tensorflow/tfjs";
// used to load tensorflow model to tfjs one

const App = () => {

  // ceate model and action states
  const [model, setModel] = useState(null)
  const [action, setAction] = useState(null)
  const [labels, setLabels] = useState(null) 

  // create recognizer

  const loadModel = async () => {
    //const jsonUpload = document.getElementById('/home/monda/Documents/ForthYear/gp/tfjs_speech/tf_speech_model/weights/model/model.json');
    //const weightsUpload = document.getElementById('/home/monda/Documents/ForthYear/gp/tfjs_speech/tf_speech_model/weights/model/weights.bin');
    // here we load the model , model must be saved like the below format
    const model = await tf.loadLayersModel('file://temp/tfjs_model');
    console.log('Model Loaded')
    // to ensure the model was loaded
    //await model.ensureModelLoaded();
    
    // set the model and labels to use them
    //setModel(model)
  }
  // to trigger the function
  useEffect(()=>{loadModel()}, []);

  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Edit <code>src/App.js</code> and save to reload.
        </p>
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>
      </header>
    </div>
  );
}

export default App;
