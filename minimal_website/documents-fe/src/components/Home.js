import React, { Component } from "react";
import { Col, Container, Row } from "reactstrap";
import DocumentList from "./DocumentList";
import NewDocumentModal from "./NewDocumentModal";
import ChatWindow from "./ChatWindow.tsx";

import axios from "axios";

import { API_URL } from "../constants";

class Home extends Component {
  state = {
    documents: []
  };

  componentDidMount() {
    this.resetState();
  }

  getStudents = () => {
    axios.get(API_URL).then(res => this.setState({ documents: res.data }));
  };

  resetState = () => {
    this.getStudents();
  };

  render() {
    return (
      <Container style={{ marginTop: "20px" }}>
        <Row>
          <Col>
            <DocumentList
              documents={this.state.documents}
              resetState={this.resetState}
            />
          </Col>
        </Row>
        <Row>
          <Col>
            <NewDocumentModal create={true} resetState={this.resetState} />
          </Col>
        </Row>
        <ChatWindow />
      </Container>
    );
  }
}

export default Home;