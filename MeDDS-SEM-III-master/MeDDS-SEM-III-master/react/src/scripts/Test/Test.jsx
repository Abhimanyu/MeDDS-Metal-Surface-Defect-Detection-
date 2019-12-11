import React, { Component } from 'react'
import '../../styles/Test/Test.scss'
import "../../styles/Test/Output.scss";
import delay from 'delay'
import img1 from '../../assets/1.svg'
import bsCustomFileInput from 'bs-custom-file-input'
// import Output from "./Output";
// import axios from 'axios'

class Test extends Component {
	constructor(props) {
		super(props)
		this.handleClick = this.handleClick.bind(this)
		this.state = { l: [] }
	}
	async handleClick() {
		let fileInput = document.getElementById('inputGroupFile01')
		let fileName = fileInput.files[0].name
		let path_out = 'http://localhost:5000/'
		// this.setState({})
		// await delay(500)
		let l = Array.from(this.state.l)
		l.push(path_out + '1/' + fileName)
		l.push(path_out + '2/' + fileName)
		l.push(path_out + '3/' + fileName)
		l.push(path_out + '4/' + fileName)
		// let data = await axios.get('http://localhost:8080')
		// l.push(path_out + 'colour/' + fileName)

		this.setState({ l: l })

		console.log(l)
	}

	componentDidMount() {
		bsCustomFileInput.init()
	}
	state = {}
	render() {
		return (
			<React.Fragment>
				<div className="container">
					<div className="div1 z-depth-3">
						<div className="row">
							<div className="col-sm-12 col-md-6 leftTest">
								<div className="text">
									<span className="titleTest">
										Submit Images
									</span>
									<div className="mainTest">
										<span className="text">
											Select Images to be uploaded for
											processing, 1600 x 256 B/W metal
											strips
										</span>
									</div>
									<div className="input-group upload">
										<div className="custom-file">
											<input
												type="file"
												className="custom-file-input"
												id="inputGroupFile01"
												aria-describedby="inputGroupFileAddon01"
											/>
											<label
												className="custom-file-label custom-file"
												htmlFor="inputGroupFile01"
											>
												Choose File
											</label>
										</div>
									</div>
								</div>
								<br />
								<div className="buttontest ">
									<a href="#1">

									<button
										onClick={this.handleClick}
										className="btn btnTest z-depth-2"
										>

										Submit
									</button>
										</a>
								</div>
							</div>
							<div className="col-sm-12 col-md-6 rightTest">
								<img className="image" src={img1} alt="" />
							</div>
						</div>
					</div>
				</div>
				{/* <div className="k">
					<img className="imag" src={this.state.l[0]} alt="" /><br/>
					<img className="imag" src={this.state.l[1]} alt="" /><br/>
					<img className="imag" src={this.state.l[2]} alt="" /><br/>
					<img className="imag" src={this.state.l[3]} alt="" /><br/>
					<img className="imag" src={this.state.l[4]} alt="" />
				</div> */}
				<React.Fragment >
					<div  className="container">
						<div className="div10 z-depth-3">
							<span className="heading " >Output</span>
							<br />
							<img className="imageOutput" src={this.state.l[0]} alt="" /><br/><br />
							<img className="imageOutput" src={this.state.l[1]} alt="" /><br/><br />
							<img className="imageOutput" src={this.state.l[2]} alt="" /><br/><br />
							<img className="imageOutput" id ="1" src={this.state.l[3]} alt="" />
							{/* <img className="imageOutput" src={this.state.l[0]} alt="" /> */}
						</div>
					</div>
				</React.Fragment>

				{/* <Output /> */}
			</React.Fragment>
		)
	}
}

export default Test
