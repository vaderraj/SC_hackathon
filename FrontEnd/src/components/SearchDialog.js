import React, { Fragment, useState } from "react";
import TextField from "@material-ui/core/TextField";
import AutoComplete from "@material-ui/lab/Autocomplete";
import companySymbols from "../data/company_id_name.json";
import { useHistory } from "react-router-dom";
import { Dialog } from "@material-ui/core";

const SearchDialog = (props) => {
  const history = useHistory();

  return (
    <Fragment>
      <Dialog open={props.open} fullWidth="true">
        <div className="card d-flex flex-column justify-content-center align-items-center m-1">
          <h3 className="mt-3" style={{ color: "green" }}>
            Stock Sentiment Predictor
          </h3>

          <AutoComplete
            onChange={(event, newValue) => {
              console.log(newValue)
              props.setOpen(false);
              props.setCompany(newValue);
            }}
            id="company-search"
            options={companySymbols}
            style={{ height: 75 }}
            className="m-5 align-self-stretch"
            getOptionLabel={(option) => option.name}
            renderInput={(params) => (
              <TextField
                {...params}
                label="Select Company"
                variant="outlined"
              />
            )}
          />
        </div>
      </Dialog>
    </Fragment>
  );
};

export default SearchDialog;
