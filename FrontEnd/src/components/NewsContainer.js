import React, { Fragment, useEffect, useState } from "react";
import axios from "axios";
import NewsItem from "./NewsItem";

const NewsContainer = (props) => {
  let numberOfArticles = Object.keys(props.companyData.data.date).length;
  let mappingArray = [...Array(numberOfArticles).keys()];

  return (
    <Fragment>
      {props.companyData ? (
        <div className=" my-5 mx-3">
          <div className="d-flex justify-content-center">
            <h3>News Articles used for Prediction</h3>
          </div>

          {mappingArray.map((identifier, id) => {
            return (
              <NewsItem
                key={id}
                source={props.companyData.data.source[identifier]}
                headline={props.companyData.data.heading[identifier]}
                description={
                  props.companyData.data.article_content[identifier].substring(
                    0,
                    500
                  ) + "..."
                }
                time={props.companyData.data.date[identifier]}
                isPositive={props.companyData.data.preds[identifier]}
              />
            );
          })}
        </div>
      ) : (
        <div>Loading...</div>
      )}
    </Fragment>
  );
};

export default NewsContainer;
