import requests
from datetime import datetime

import pandas as pd


class Datastream:
    """Class to call Datastream APIs."""

    def __init__(self, username, password):
        self.token_url = (
            "http://product.datastream.com/DSWSClient/V1/DSService.svc/rest/Token?"
        )
        self.data_url = "http://datastream.thomsonreuters.com/DswsClient/V1/DSService.svc/rest/Data?"
        self.token = self.get_token(username, password)

    def get_token(self, username, password):
        """Get token."""
        params = {"username": username, "password": password}
        res = requests.get(self.token_url, params=params).json()
        return res["TokenValue"]

    @staticmethod
    def from_json_to_df(response_json):
        """Converts json to Pandas dataframe."""
        # If dates is not available, the request is not constructed correctly
        if response_json["Dates"]:
            df = pd.DataFrame(
                index=[
                    datetime.utcfromtimestamp(float(d[6:-10]))
                    for d in response_json["Dates"]
                ]
            )
            df.index.name = "Date"

            # Loop through the values in the response
            for item in response_json["DataTypeValues"]:
                for e in item["SymbolValues"]:
                    df[(e["Symbol"], item["DataType"])] = e["Value"]

            # Use Pandas MultiIndex to get from tuples to two header rows
            df.columns = pd.MultiIndex.from_tuples(
                df.columns, names=["Instrument", "Field"]
            )
            return df

        print("Error - please check instruments and parameters (time series or static)")
        return None

    # pylint: disable=too-many-arguments
    def get_data(
        self, tickers, fields="", date="", start_date="", end_date="", freq=""
    ):
        """Get data in Pandas dataframe."""
        # Decide if the request is a time series or static request
        if not start_date:
            # snapshot requests, the value of 'date' needs to be put into 'start_date'
            datekind = "Snapshot"
            start_date = date
        else:
            datekind = "TimeSeries"

        # Put all the fields in a request and encode them for requests.get
        params = {
            "token": self.token,
            "instrument": tickers,
            "datatypes": fields,
            "datekind": datekind,
            "start": start_date,
            "end": end_date,
            "freq": freq,
        }

        # Retrieve data and use the json native decoder
        response = requests.get(self.data_url, params=params).json()

        # Convert the JSON response to a Pandas DataFrame
        return response  # self.from_json_to_df(response)
