import React, { Component } from "react";
import "../styles/Nav.scss";
import logo from "../assets/logo.png";
class Nav extends Component {
  state = {};
  render() {
    return (
      <div className="outerNav">
        <div className="row onPc">
          <div className="col-sm-12 col-md-9">
            <span className="logoNav">
              <img className="navLogo" src={logo} alt="" />
            </span>
          </div>
          <div className="col-sm-12 col-md-3 navButton">
            <div className="row">
              <div className="col-sm-12 col-md-6">
                <span className="ButtonNav">HOME</span>
              </div>
              <div className="col-sm-12 col-md-6">
                <span className="ButtonNav">ABOUT US</span>
              </div>
            </div>
          </div>
        </div>

<div className="row onPhone">
  <div className="col-sm-12 col-md-12" >
    <span className="logoNav">
      <img className="navLogo" src={logo} alt="" />
    </span>
  </div>
  
</div>
</div>
    );
  }
}

export default Nav;
