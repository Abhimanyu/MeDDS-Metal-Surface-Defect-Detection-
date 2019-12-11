
import React, { Component } from "react";
import Nav from "./Nav"
// import Upper from "./Upper";
// import Employee from "./login/Employee";
import Test from "./Test/Test";
// import Output from "./Test/Output";


class App extends Component {
  render() {
    return (
      <div className="App">
        <Nav/>
        {/* <Upper /> */}
        {/* <Employee /> */}
        <Test />
        {/* <Output /> */}
      </div>
    );
  }
}

export default App;

