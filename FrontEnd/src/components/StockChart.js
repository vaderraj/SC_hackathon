import React, { Fragment, useEffect, useState } from "react";
import axios from "axios";
import Plot from "react-plotly.js";

const StockChart = (props) => {
  const API_KEY = "P654EDCLDQTNV87I";
  const companyTitle = props.company.name;

  const [stockData, setStockData] = useState();

  useEffect(() => {
    const companySymbol = props.company.symbol;

    axios
      .get(
        `https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=${companySymbol}&outputsize=compact&apikey=${API_KEY}`
      )
      .then((response) => {
        let xValues = [];
        let yValues = [];
        console.log(response.data);
        let i = 0;

        for (let key in response.data["Time Series (Daily)"]) {
          if (i < 30) {
            xValues.push(key);
            yValues.push(response.data["Time Series (Daily)"][key]["1. open"]);
          }
          i += 1;
        }

        setStockData([xValues, yValues]);
      });
  }, []);

  return (
    <Fragment>
      {stockData ? (
        <Plot
          data={[
            {
              x: stockData[0],
              y: stockData[1],
              type: "scatter",
              mode: "lines+markers",
              marker: { color: "orange" },
            },
          ]}
          layout={{
            width: 775,
            height: 480,
            yaxis:{
              title: "Stock Price ( USD )"
            },
            xaxis:{
              title: "Date"
            },
            title: companyTitle + "'s Stock prices in the last 30 Days",
            titlefont: {
              size: 20,
            },
          }}
        />
      ) : (
        <div>Receiving Data...</div>
      )}
    </Fragment>
  );
};

export default StockChart;
