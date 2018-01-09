import csv

import gdax
from datetime import datetime, timedelta
import time

public_client = gdax.PublicClient()


class HistoricRate:
    def __init__(self, product, historic_rate_received):
        self.id = product["id"]
        self.timestamp = datetime.fromtimestamp(historic_rate_received[0])
        self.lowestPrice = historic_rate_received[1]
        self.highestPrice = historic_rate_received[2]
        self.openingPrice = historic_rate_received[3]
        self.closingPrice = historic_rate_received[4]
        self.volumeTraded = historic_rate_received[5]

    def __str__(self):
        return "id : {}, timestamp : {}".format(
            self.id,
            self.timestamp)

    def __iter__(self):
        return iter([self.id,
                     self.timestamp,
                     self.lowestPrice,
                     self.highestPrice,
                     self.openingPrice,
                     self.closingPrice,
                     self.volumeTraded])


def get_batch_from_api(products, start_ts, end_ts):
    historic_rate_batch = []
    for product in products:
        api_response = public_client.get_product_historic_rates(product["id"], start_ts, end_ts)
        for historic_rate_received in api_response:
            try:
                historic_rate_batch.append(HistoricRate(product, historic_rate_received))
            except TypeError:
                print("bad data: %s" % api_response)

    return historic_rate_batch


def process_batch(products, start_ts, end_ts, writer):
    historic_rate_batch = get_batch_from_api(products, start_ts, end_ts)
    writer.writerows(historic_rate_batch)


if __name__ == '__main__':

    crypto_moneys = ["BTC", "ETH", "LTC"]  # BTC, ETH or LTC

    batchLengthHours = 6

    for crypto_money in crypto_moneys:

        products_EUR = list(filter(lambda x: "EUR" in x["id"] and crypto_money in x["id"], public_client.get_products()))

        beginning_date_str = "2017-01-01"
        end_date_str = "2017-12-09"

        batch_beginning_date = datetime.strptime(beginning_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d') + timedelta(days=1)

        with open("data_" + crypto_money + ".csv", 'a') as csv_file:
            wr = csv.writer(csv_file, delimiter=",")
            wr.writerow(
            ["id", "timestamp", "lowestPrice", "highestPrice", "openingPrice", "closingPrice", "volumeTraded"])
            # first batch
            batch_end_date = min(end_date, batch_beginning_date + timedelta(hours=batchLengthHours))
            process_batch(products_EUR, batch_beginning_date, batch_end_date, wr)

            # other batches
            while batch_end_date < end_date:
                time.sleep(1)  # limited to 3 call per sec

                batch_beginning_date += timedelta(hours=batchLengthHours)
                batch_end_date += timedelta(hours=batchLengthHours)
                print("fetching data between %s and %s" % (batch_beginning_date, batch_end_date))
                process_batch(products_EUR, batch_beginning_date, batch_end_date, wr)
