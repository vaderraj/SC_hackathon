import React, { Fragment } from "react";

const NewsItem = (props) => {
  return (
    <Fragment>
      <div
        className="container card my-3"
        style={{
          border: "thin solid orange",
          backgroundColor: props.isPositive ? "#e6ffee90" : "#ffece690",
        }}
      >
        <h6 className="mx-1 my-3">{props.source}</h6>
        <h4 className="card-title">{props.headline}</h4>
        <hr />
        <div className="card-body">{props.description}</div>
        <p>{props.time}</p>
      </div>
    </Fragment>
  );
};

export default NewsItem;
