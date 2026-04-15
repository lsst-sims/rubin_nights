import unittest
from unittest.mock import Mock, patch

import pandas as pd
from astropy.time import Time

from rubin_nights.influx_query import InfluxQueryClient


class DummyResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        return None


class TestInfluxQueryClient(unittest.TestCase):

    def setUp(self) -> None:
        self.repertoire_payload = {
            "efd": {
                "database": "efd",
                "username": "user1",
                "password": "pass1",
                "url": "https://example.test/influx",
            }
        }
        self.segwarides_payload = {
            "username": "user2",
            "password": "pass2",
            "host": "seg.test",
            "path": "/influx/",
        }

    @patch("rubin_nights.influx_query.httpx.AsyncClient")
    @patch("rubin_nights.influx_query.httpx.Client")
    @patch("rubin_nights.influx_query.httpx.get")
    def test_init_uses_repertoire_for_usdf_with_auth(self, mock_get, mock_client, mock_async_client) -> None:
        mock_get.return_value = DummyResponse(self.repertoire_payload)

        client = InfluxQueryClient("usdf", db_name="efd", auth=("auth_user", "dummy_token"))

        self.assertEqual(client.site, "usdf")
        self.assertEqual(client.url, "https://example.test/influx")
        mock_get.assert_called_once()
        mock_client.assert_called_once()
        mock_async_client.assert_called_once()

    @patch("rubin_nights.influx_query.httpx.AsyncClient")
    @patch("rubin_nights.influx_query.httpx.Client")
    @patch("rubin_nights.influx_query.httpx.get")
    def test_init_falls_back_to_segwarides_on_repertoire_error(
        self, mock_get, mock_client, mock_async_client
    ) -> None:
        mock_get.side_effect = [
            Exception("boom"),
            DummyResponse(self.segwarides_payload),
        ]

        client = InfluxQueryClient("usdf", db_name="efd", auth=("auth_user", "dummy_token"))

        self.assertEqual(client.site, "usdf")
        self.assertEqual(client.url, "https://seg.test/influx")

    def test_build_influxdb_query(self) -> None:
        start = Time("2025-01-01T00:00:00", scale="utc")
        end = Time("2025-01-01T01:00:00", scale="utc")

        query = InfluxQueryClient.build_influxdb_query(
            "my_measurement",
            fields=["a", "b"],
            time_range=(start, end),
            filters=[("salIndex", "1")],
        )

        self.assertEqual(
            query,
            "SELECT a, b FROM \"my_measurement\" WHERE time >= '2025-01-01T00:00:00.000Z' "
            "AND time <= '2025-01-01T01:00:00.000Z' AND salIndex = 1",
        )

    def test_build_influxdb_top_n_query(self) -> None:
        cut = Time("2025-01-01T00:00:00", scale="utc")

        query = InfluxQueryClient.build_influxdb_top_n_query(
            "my_measurement",
            fields="a",
            num=5,
            time_cut=cut,
            filters=[("salIndex", "2")],
        )

        self.assertEqual(
            query,
            "SELECT a FROM \"my_measurement\" WHERE time <= '2025-01-01T00:00:00.000Z' "
            "AND salIndex = 2 GROUP BY * ORDER BY DESC LIMIT 5",
        )

    def test_to_dataframe_with_time_tags_and_name(self) -> None:
        client = object.__new__(InfluxQueryClient)
        response = {
            "results": [
                {
                    "series": [
                        {
                            "name": "topic_name",
                            "columns": ["time", "value"],
                            "values": [["2025-01-01T00:00:00Z", 42]],
                            "tags": {"site": "usdf"},
                        }
                    ]
                }
            ]
        }

        df = InfluxQueryClient._to_dataframe(client, response)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(list(df.columns), ["value", "site"])
        self.assertEqual(df.iloc[0]["value"], 42)
        self.assertEqual(df.iloc[0]["site"], "usdf")
        self.assertEqual(df.name, "topic_name")
        self.assertIsNotNone(df.index.tz)

    def test_to_dataframe_zero_results(self) -> None:
        client = object.__new__(InfluxQueryClient)
        response = {"results": [{}]}
        df = InfluxQueryClient._to_dataframe(client, response)
        self.assertTrue(df.empty)

    @patch("rubin_nights.influx_query.httpx.AsyncClient")
    @patch("rubin_nights.influx_query.httpx.Client")
    @patch.object(
        InfluxQueryClient, "_fetch_credentials_segwarides", return_value=("https://example.test", ("u", "p"))
    )
    def test_query_returns_dataframe(self, mock_creds, mock_client, mock_async_client) -> None:
        mock_response = Mock()
        mock_response.json.return_value = {
            "results": [
                {
                    "series": [
                        {
                            "name": "topic_name",
                            "columns": ["time", "value"],
                            "values": [["2025-01-01T00:00:00Z", 1]],
                        }
                    ]
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_client.return_value.get.return_value = mock_response

        client = InfluxQueryClient("summit", db_name="efd", results_as_dataframe=True)
        result = client.query("show measurements")

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.iloc[0]["value"], 1)

    @patch("rubin_nights.influx_query.httpx.AsyncClient")
    @patch("rubin_nights.influx_query.httpx.Client")
    @patch.object(
        InfluxQueryClient, "_fetch_credentials_segwarides", return_value=("https://example.test", ("u", "p"))
    )
    def test_query_returns_list_when_not_dataframe(self, mock_creds, mock_client, mock_async_client) -> None:
        mock_response = Mock()
        payload = {"results": [{"series": [{"columns": ["name"], "values": [["m1"]]}]}]}
        mock_response.json.return_value = payload
        mock_response.raise_for_status.return_value = None
        mock_client.return_value.get.return_value = mock_response

        client = InfluxQueryClient("summit", db_name="efd", results_as_dataframe=False)
        result = client.query("show measurements")

        self.assertEqual(result, payload)

    @patch("rubin_nights.influx_query.httpx.AsyncClient")
    @patch("rubin_nights.influx_query.httpx.Client")
    @patch.object(
        InfluxQueryClient, "_fetch_credentials_segwarides", return_value=("https://example.test", ("u", "p"))
    )
    def test_get_topics_caches_result(self, mock_creds, mock_client, mock_async_client) -> None:
        client = InfluxQueryClient("summit", db_name="efd")
        client.query = Mock(return_value=pd.DataFrame({"name": ["m1", "m2"]}))

        first = client.get_topics()
        second = client.get_topics()

        self.assertEqual(first, ["m1", "m2"])
        self.assertEqual(second, ["m1", "m2"])
        self.assertEqual(client.query.call_count, 1)


if __name__ == "__main__":
    unittest.main()
