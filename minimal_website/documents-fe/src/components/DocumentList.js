import React, { Component } from "react";
import { Table } from "reactstrap";
import NewDocumentModal from "./NewDocumentModal";

import ConfirmRemovalModal from "./ConfirmRemovalModal";

class DocumentList extends Component {
  render() {
    const documents = this.props.documents;
    return (
      <Table dark>
        <thead>
          <tr>
            <th>Name</th>
            <th>Document Type</th>
            <th>Image</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {!documents || documents.length <= 0 ? (
            <tr>
              <td colSpan="6" align="center">
                <b>Ops, no one here yet</b>
              </td>
            </tr>
          ) : (
            documents.map(document => (
              <tr key={document.pk}>
                <td>{document.name}</td>
                <td>{document.document_type}</td>
                {/*<td>{document.image}</td>*/}
                <td>{document.registrationDate}</td>
                <td align="center">
                  <NewDocumentModal
                    create={false}
                    student={document}
                    resetState={this.props.resetState}
                  />
                  &nbsp;&nbsp;
                  <ConfirmRemovalModal
                    pk={document.pk}
                    resetState={this.props.resetState}
                  />
                </td>
              </tr>
            ))
          )}
        </tbody>
      </Table>
    );
  }
}

export default DocumentList;
