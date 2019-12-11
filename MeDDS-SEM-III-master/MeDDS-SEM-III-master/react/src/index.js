import React from 'react';
import ReactDOM from 'react-dom';
import "bootstrap-css-only/css/bootstrap.min.css";
import "mdbreact/dist/css/mdb.css";
import './styles/index.scss';

import App from './scripts/App';

import * as serviceWorker from "./serviceWorker";

ReactDOM.render(<App />, document.getElementById('root'));
serviceWorker.unregister();