import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
from astropy.time import Time

from rubin_nights.influx_query import AsyncInfluxQueryClient, InfluxQueryClient
from rubin_nights.reference_values import API_ENDPOINTS

FAKE_URL = "https://example.test/influx"
FAKE_CREDS = (FAKE_URL, ("u", "p"))
REPERTOIRE_AUTH = ("auth_user", "dummy_token")


class DummyResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        return None


def influx_response(values: list, columns: tuple[str, ...] = ("time", "value")) -> dict:
    """Minimal InfluxDB /query JSON response with one series."""
    return {"results": [{"series": [{"name": "topic_name", "columns": list(columns), "values": values}]}]}


class TestInfluxQueryBuilders(unittest.TestCase):

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
        self.assertEqual(df.attrs["name"], "topic_name")
        self.assertIsNotNone(df.index.tz)

    def test_to_dataframe_zero_results(self) -> None:
        client = object.__new__(InfluxQueryClient)
        response = {"results": [{}]}
        df = InfluxQueryClient._to_dataframe(client, response)
        self.assertTrue(df.empty)


@patch("rubin_nights.influx_query.httpx2.Client")
class TestInfluxQueryCredentials(unittest.TestCase):

    def setUp(self) -> None:
        # Repertoire returns a mapping of influx db aliases to credentials.
        self.repertoire_payload = {
            "usdf_efd": {"username": "user1", "password": "pass1", "url": FAKE_URL},
            "summit_efd": {"username": "user2", "password": "pass2", "url": "https://summit.test/influx"},
        }
        # Make sure an EFDAUTH set in the test environment does not leak in.
        env_patcher = patch.dict(os.environ)
        env_patcher.start()
        self.addCleanup(env_patcher.stop)
        os.environ.pop("EFDAUTH", None)
        # A credentials file in the same format.
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        self.auth_file = Path(tmpdir.name) / "efd_auth.json"
        self.auth_file.write_text(json.dumps(self.repertoire_payload))

    def test_init_uses_repertoire_for_usdf_with_auth(self, mock_client) -> None:
        # The same mocked httpx2.Client is used for the repertoire request
        # (as a context manager) and for the influx query client.
        mock_get = mock_client.return_value.__enter__.return_value.get
        mock_get.return_value = DummyResponse(self.repertoire_payload)

        client = InfluxQueryClient("usdf", db_name="efd", auth=REPERTOIRE_AUTH)

        self.assertEqual(client.site, "usdf")
        self.assertEqual(client.url, FAKE_URL)
        self.assertEqual(client.creds_from, "repertoire @ usdf")
        mock_get.assert_called_once_with(
            f"{API_ENDPOINTS['usdf']}/repertoire/discovery/influxdb", auth=REPERTOIRE_AUTH
        )
        # Influx query client set up with the url and credentials from repertoire.
        self.assertEqual(mock_client.call_args.kwargs["base_url"], FAKE_URL)
        self.assertEqual(mock_client.call_args.kwargs["auth"], ("user1", "pass1"))

    def test_repertoire_site_used_for_discovery(self, mock_client) -> None:
        mock_get = mock_client.return_value.__enter__.return_value.get
        mock_get.return_value = DummyResponse(self.repertoire_payload)

        client = InfluxQueryClient("summit", repertoire_site="usdf", auth=REPERTOIRE_AUTH)

        mock_get.assert_called_once_with(
            f"{API_ENDPOINTS['usdf']}/repertoire/discovery/influxdb", auth=REPERTOIRE_AUTH
        )
        self.assertEqual(client.url, "https://summit.test/influx")
        self.assertEqual(client.creds_from, "repertoire @ usdf")

    def test_repertoire_missing_alias_raises(self, mock_client) -> None:
        mock_get = mock_client.return_value.__enter__.return_value.get
        mock_get.return_value = DummyResponse({"summit_efd": self.repertoire_payload["summit_efd"]})

        with self.assertRaises(ValueError):
            InfluxQueryClient("usdf", auth=REPERTOIRE_AUTH)

    def test_repertoire_takes_precedence_over_file(self, mock_client) -> None:
        mock_get = mock_client.return_value.__enter__.return_value.get
        mock_get.return_value = DummyResponse(self.repertoire_payload)
        os.environ["EFDAUTH"] = str(self.auth_file)

        client = InfluxQueryClient("usdf", auth=REPERTOIRE_AUTH, influx_auth_file=self.auth_file)

        mock_get.assert_called_once()
        self.assertEqual(client.creds_from, "repertoire @ usdf")

    def test_influx_auth_file(self, mock_client) -> None:
        client = InfluxQueryClient("usdf", influx_auth_file=self.auth_file)

        self.assertEqual(client.url, FAKE_URL)
        self.assertEqual(client.creds_from, f"file @ {self.auth_file}")
        self.assertEqual(mock_client.call_args.kwargs["base_url"], FAKE_URL)
        self.assertEqual(mock_client.call_args.kwargs["auth"], ("user1", "pass1"))

    def test_influx_auth_file_takes_precedence_over_efdauth(self, mock_client) -> None:
        other_file = self.auth_file.with_name("other.json")
        other_file.write_text(
            json.dumps({"usdf_efd": {"username": "x", "password": "y", "url": "https://other"}})
        )
        os.environ["EFDAUTH"] = str(other_file)

        client = InfluxQueryClient("usdf", influx_auth_file=self.auth_file)

        self.assertEqual(client.url, FAKE_URL)
        self.assertEqual(client.creds_from, f"file @ {self.auth_file}")

    def test_efdauth_environment_variable(self, mock_client) -> None:
        os.environ["EFDAUTH"] = str(self.auth_file)

        client = InfluxQueryClient("summit")

        self.assertEqual(client.url, "https://summit.test/influx")
        self.assertEqual(client.creds_from, f"file @ {self.auth_file}")
        self.assertEqual(mock_client.call_args.kwargs["auth"], ("user2", "pass2"))

    def test_auth_file_missing_alias_raises(self, mock_client) -> None:
        with self.assertRaises(ValueError) as cm:
            InfluxQueryClient("base", influx_auth_file=self.auth_file)
        self.assertIn("base_efd", str(cm.exception))

    def test_no_credentials_raises(self, mock_client) -> None:
        with self.assertRaises(ValueError):
            InfluxQueryClient("usdf")
        mock_client.assert_not_called()

    def test_credentials_not_stored_on_client(self, mock_client) -> None:
        client = InfluxQueryClient("usdf", influx_auth_file=self.auth_file)

        self.assertFalse(hasattr(client, "influx_auth"))
        self.assertNotIn("pass1", repr(vars(client)))
        self.assertNotIn("pass1", repr(client))


class TestInfluxQueryClient(unittest.TestCase):

    @patch.object(InfluxQueryClient, "_fetch_credentials", return_value=FAKE_CREDS)
    @patch("rubin_nights.influx_query.httpx2.Client")
    def test_query_returns_dataframe(self, mock_client, mock_creds) -> None:
        mock_response = Mock()
        mock_response.json.return_value = influx_response([["2025-01-01T00:00:00Z", 1]])
        mock_response.raise_for_status.return_value = None
        mock_client.return_value.get.return_value = mock_response

        client = InfluxQueryClient("summit", db_name="efd", auth=REPERTOIRE_AUTH)
        result = client.query("show measurements")

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.iloc[0]["value"], 1)
        # show queries put the id tag comment first.
        sent = mock_client.return_value.get.call_args.kwargs["params"]
        self.assertEqual(sent["db"], "efd")
        self.assertTrue(sent["q"].startswith(" /* source:"))
        self.assertEqual(client.last_query, sent["q"])

    @patch.object(InfluxQueryClient, "_fetch_credentials", return_value=FAKE_CREDS)
    @patch("rubin_nights.influx_query.httpx2.Client")
    def test_query_failure_returns_empty(self, mock_client, mock_creds) -> None:
        mock_client.return_value.get.side_effect = Exception("boom")

        client = InfluxQueryClient("summit", db_name="efd", auth=REPERTOIRE_AUTH)
        result = client.query('SELECT * FROM "topic"')

        self.assertTrue(result.empty)

    @patch.object(InfluxQueryClient, "_fetch_credentials", return_value=FAKE_CREDS)
    @patch("rubin_nights.influx_query.httpx2.Client")
    def test_get_topics_caches_result(self, mock_client, mock_creds) -> None:
        client = InfluxQueryClient("summit", db_name="efd", auth=REPERTOIRE_AUTH)
        client.query = Mock(return_value=pd.DataFrame({"name": ["m1", "m2"]}))

        first = client.get_topics()
        second = client.get_topics()

        self.assertEqual(first, ["m1", "m2"])
        self.assertEqual(second, ["m1", "m2"])
        self.assertEqual(client.query.call_count, 1)

    @patch.object(InfluxQueryClient, "_fetch_credentials", return_value=FAKE_CREDS)
    @patch("rubin_nights.influx_query.httpx2.Client")
    def test_context_manager_closes_client(self, mock_client, mock_creds) -> None:
        with InfluxQueryClient("summit", auth=REPERTOIRE_AUTH) as client:
            self.assertIsInstance(client, InfluxQueryClient)
        mock_client.return_value.close.assert_called_once()


class TestAsyncInfluxQueryClient(unittest.IsolatedAsyncioTestCase):

    @patch.object(AsyncInfluxQueryClient, "_fetch_credentials", return_value=FAKE_CREDS)
    @patch("rubin_nights.influx_query.httpx2.AsyncClient")
    async def test_async_query_returns_dataframe(self, mock_async_client, mock_creds) -> None:
        mock_response = Mock()
        mock_response.json.return_value = influx_response([["2025-01-01T00:00:00Z", 7]])
        mock_response.raise_for_status.return_value = None
        mock_async_client.return_value.get = AsyncMock(return_value=mock_response)
        mock_async_client.return_value.aclose = AsyncMock()

        async with AsyncInfluxQueryClient("summit", auth=REPERTOIRE_AUTH) as client:
            self.assertEqual(mock_async_client.call_args.kwargs["base_url"], FAKE_URL)
            result = await client.query('SELECT * FROM "topic"')

        self.assertEqual(result.iloc[0]["value"], 7)
        mock_async_client.return_value.aclose.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
