import React from "react";
import { Button, Form, FormGroup, Input, Label } from "reactstrap";

import axios from "axios";

import { API_URL } from "../constants";

class NewDocumentForm extends React.Component {
  state = {
    pk: 0,
    name: "",
    document_type: "",
    image: ""
  };

  // should store an image id here rather than an image
  componentDidMount() {
    if (this.props.document) {
      const { pk, name, document_type, image } = this.props.document;
      this.setState({ pk, name, document_type, image });
    }
  }

  onChange = e => {
    this.setState({ [e.target.name]: e.target.value });
  };

  onChangeImage = e => {
    // Create a new FileReader object
    const reader = new FileReader();
    // Set the onload function to read the file and set the state
    this.setState({[e.target.name]: reader.result});
    // Read the file
    reader.readAsArrayBuffer( e.target.files[0])
  }

  createStudent = e => {
    e.preventDefault();
    axios.post(API_URL, this.state).then(() => {
      this.props.resetState();
      this.props.toggle();
    });
  };

  editStudent = e => {
    e.preventDefault();
    axios.put(API_URL + this.state.pk, this.state).then(() => {
      this.props.resetState();
      this.props.toggle();
    });
  };

  defaultIfEmpty = value => {
    return value === "" ? "" : value;
  };

  render() {
    return (
      <Form onSubmit={this.props.student ? this.editStudent : this.createStudent}>
        <FormGroup>
          <Label for="name">Name:</Label>
          <Input
            type="text"
            name="name"
            onChange={this.onChange}
            value={this.defaultIfEmpty(this.state.name)}
          />
        </FormGroup>
        <FormGroup>
          <Label for="document_type">Document Type:</Label>
          <Input
            type="text"
            name="document_type"
            onChange={this.onChange}
            value={this.defaultIfEmpty(this.state.document_type)}
          />
        </FormGroup>
        <FormGroup>
          <Label for="image">Document Type:</Label>
          <Input
            type="file"
            name="image"
            onChange={this.onChangeImage}
            value={this.defaultIfEmpty(this.state.image)}
          />
        </FormGroup>
    <Button>Send</Button>
      </Form>
    );
  }
}

export default NewDocumentForm;