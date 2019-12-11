import React, { Component } from 'react';
import "../styles/Upper.scss";
import upperImage from "../assets/Upper.png";
class Upper extends Component {
    state = {  }
    render() { 
        return ( 
            <div className="outerUpper">
                    <div className="row onPc">
                        <div className="col-sm-12 col-md-5 textUpper">
                                    <span className="upperText">
                                        WELCOME <br />TO <span className="upperLogo">MEDDS</span>
                                    </span>
                                    <div className="row butttons">
                                        <div className="col-sm-12 col-md-3 buttonUpper">
                                            <button className="btn-sm btn button">
                                                <span className="textButton">
                                                <span className="bold1">EMP</span> LOGIN
                                                </span>
                                            </button>
                                        </div>
                                        <div className="col-sm-12 col-md-3 buttonUpper">
                                            <button className="btn-sm btn button">
                                                <span className="textButton">
                                                <span className="bold1">ADMIN</span> LOGIN
                                                </span>
                                            </button>

                                        </div>
                                    </div>
                        </div>
                        <div className="col-sm-12 col-md-7 imageUpper">
                            <img className="image" src={upperImage} alt=""/>
                        </div>
                    </div>
                    <div className="row onPhone ">
                        <div className="col-sm-12 col-md-5 textUpper UpperPhone">
                                  <span className="upperText">
                                        WELCOME <br />TO <span className="upperLogo">MEDDS</span>
                                    </span>
                                    <div className="row buttons">
                                        <div className="col-sm-6 col-md-6 buttonUpper">
                                            <button className="btn-sm btn button">
                                                <span className="textButton">
                                                <span className="bold1">EMP</span> LOGIN
                                                </span>
                                            </button>
                                        </div>
                                        <div className="col-sm-6 col-md-6 buttonUpper">
                                            <button className="btn-sm btn button">
                                                <span className="textButton">
                                                <span className="bold1">ADMIN</span> LOGIN
                                                </span>
                                            </button>

                                        </div>
                                    </div>
                        </div>
                        <div className="col-sm-12 col-md-12 imageUpper UpperPhone">
                            <img className="image" src={upperImage} alt=""/>
                        </div>
                    </div>
            </div>
         );
    }
}
 
export default Upper;