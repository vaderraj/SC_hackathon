import React, { Fragment, useState, useEffect } from "react";
import StockChart from "../components/StockChart";
import NewsContainer from "../components/NewsContainer";
import "../styles.css";
import selectedCompanyData from "../data/relaince.json";
import axios from "axios";

import SearchDialog from "../components/SearchDialog";

const CompanyPage = (props) => {
  const newsContainerStyles = {
    overflowY: "scroll",
    height: "500px",
    width: "800px",
    border: "thin solid green",
  };

  const stockChartContainerStyle = {
    height: "500px",
    width: "800px",
    border: "thin solid green",
  };

  const [selectedCompany, setSelectedCompany] = useState(false);
  const [open, setOpen] = React.useState(true);
  const [companyData, setCompanyData] = useState();



  const load_data = ()=>{
    console.log("inside the load data", selectedCompany)
    
  }




  useEffect(() => {
    console.log("New", selectedCompany)
  
    console.log("inside if",selectedCompany)
    axios
      .get(`http://127.0.0.1:5000/${selectedCompany["id"]}`)
      .then((response) => {
        setCompanyData(response.data);
      })
      .catch((error) => {
        console.log(error);
      });
      
    
  }, [selectedCompany]);

  let numberOfArticles;
  let mappingArray;
  let numberOfPositives = 0;
  let buy = 2;

  if (companyData) {
    numberOfArticles = Object.keys(companyData.data.date).length;
    mappingArray = [...Array(numberOfArticles).keys()];

    mappingArray.forEach((identifier) => {
      if (companyData.data.preds[identifier] == 1) {
        numberOfPositives += 1;
      }
    });

    if (numberOfPositives > numberOfArticles / 2) {
      buy = 1;
    } else if (numberOfPositives < numberOfArticles / 2) {
      buy = 0;
    }
  }

  const buyText = <span style={{ color: "green" }}>Buy</span>;
  const sellText = <span style={{ color: "red" }}>Sell</span>;
  const neutralText = <span style={{ color: "blue" }}>Neutral</span>;

  return (
    <Fragment>
      <SearchDialog
        open={open}
        setOpen={setOpen}
        setCompany={setSelectedCompany}
        loadData={load_data}
      />
      
      {companyData ? (
        <div className="container-fluid d-flex flex-column justify-content-center align-items-center my-5">
          <div className="container-fluid d-flex flex-row justify-content-right align-items-center my-5">
            <div className="col-5" />
            <h3>
              Stock Sentiment of{" "}
              <span style={{ color: "orange" }}> {selectedCompany.name}</span>
            </h3>
            <div className="col-2" />

            <button
              className="btn-lg btn-primary ml-5"
              onClick={() => {
                setCompanyData();
                setOpen(true);
              }}
            >
              Change Company
            </button>
          </div>

          <div className="container-fluid d-flex flex-wrap flex-row justify-content-around align-items-center">
            <div
              className="card d-flex justify-content-center align-items-center m-5"
              style={stockChartContainerStyle}
            >
              <StockChart company={selectedCompany} />
            </div>

            <div
              className="card"
              style={newsContainerStyles}
              id="newsContainer"
            >
              <NewsContainer companyData={companyData} />
            </div>
          </div>
          <div className="mt-5">
            <h3>
              Suggestion:{" "}
              {buy == 1 ? buyText : buy == 0 ? sellText : neutralText}
              {buy == 1
                ? ` (${numberOfPositives}/ ${numberOfArticles})`
                : buy == 0
                ? ` (${
                    numberOfArticles - numberOfPositives
                  }/ ${numberOfArticles})`
                : null}
            </h3>
          </div>
        </div>
      ) : (
        <div>Loading</div>
      )}
    </Fragment>
  );
};

export default CompanyPage;
