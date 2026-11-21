import os
import unittest
from pathlib import Path
from unittest.mock import Mock

import rubin_nights.connections as connections
from rubin_nights.logging_query import NarrativeLogClient, NightReportClient

IN_CI = os.getenv("CI") == "true"


class TestConnections(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = Path(__file__).parent
        self.tokenfile = os.path.join(self.test_dir, "data", "dummy_token")
        # This matches content from dummy_token file
        self.expected_token = "NotARealToken"
        # This is just for an environment token
        self.expected_token2 = "NotARealTokenEither"

    def test_access_token(self) -> None:
        # Save current environment settings
        current_env_token = os.getenv("ACCESS_TOKEN")
        current_env_tokenfile = os.getenv("ACCESS_TOKEN_FILE")
        # Replace them with nothing
        if current_env_token is not None:
            del os.environ["ACCESS_TOKEN"]
        if current_env_tokenfile is not None:
            del os.environ["ACCESS_TOKEN_FILE"]

        # Use the test tokenfile
        token = connections.get_access_token(tokenfile=self.tokenfile)
        self.assertEqual(token, self.expected_token)
        # Use nothing
        connections.DEFAULT_TOKENFILE = "dummy_token_does_not_exist"
        token = connections.get_access_token()
        if os.getenv("EXTERNAL_INSTANCE_URL") is None:
            self.assertTrue(len(token) == 0)
        # This is expected to get a real token on the RSP
        else:
            self.assertTrue(len(token) > 0)
        # Use environment variable as token
        os.environ["ACCESS_TOKEN"] = self.expected_token2
        token = connections.get_access_token()
        self.assertEqual(token, self.expected_token2)
        del os.environ["ACCESS_TOKEN"]
        # Use environment variable for tokenfile
        os.environ["ACCESS_TOKEN_FILE"] = self.tokenfile
        token = connections.get_access_token()
        self.assertEqual(token, self.expected_token)
        del os.environ["ACCESS_TOKEN_FILE"]
        # Reset pre-environment variables.
        if current_env_token is not None:
            os.environ["ACCESS_TOKEN"] = current_env_token
        if current_env_tokenfile is not None:
            os.environ["ACCESS_TOKEN_FILE"] = current_env_tokenfile

    def test_usdf_lfa(self) -> None:
        http_uri = "https://s3.cp.lsst.org/"
        uri = (
            "rubinobs-lfa-cp/Scheduler:1/"
            "Scheduler:1/2025/07/21/Scheduler:1_Scheduler:1_2025-07-22T03:05:04.297.p"
        )
        result = (
            "s3://lfa@rubinobs-lfa-cp/Scheduler:1/"
            "Scheduler:1/2025/07/21/Scheduler:1_Scheduler:1_2025-07-22T03:05:04.297.p"
        )
        new_uri = connections.usdf_lfa(http_uri + uri, bucket="s3://lfa@")
        self.assertEqual(new_uri, result)

    def test_usdf_s3_env(self) -> None:
        # Check if LSST_DISABLE_BUCKET_VALIDATION and S3_ENDPOINT_URL get set
        env_vars = ["LSST_DISABLE_BUCKET_VALIDATION", "S3_ENDPOINT_URL"]
        # Save current setup
        current_vars = {}
        for var in env_vars:
            current_vars[var] = os.getenv(var)
        # Remove current setup from env.
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]
        # Call function - sets variables.
        connections.usdf_s3_env()
        # Check they were set.
        for var in env_vars:
            assert var in os.environ
        # Replace original values.
        for var in env_vars:
            if current_vars[var] is not None:
                os.environ[var] = current_vars[var]

    def test_close_clients(self) -> None:
        api_base = "https://example.test"
        mock_client = Mock()
        narrative_log = NarrativeLogClient(api_base, auth_token=self.expected_token)
        night_report = NightReportClient(api_base, auth_token=self.expected_token)
        # Mimic the structure returned by get_clients, including the
        # api_base string and a None client (e.g. "too" at usdf-dev).
        endpoints = {
            "api_base": api_base,
            "too": None,
            "efd": mock_client,
            "narrative_log": narrative_log,
            "night_report": night_report,
        }
        # Should skip the str and None entries without raising.
        connections.close_clients(endpoints)
        mock_client.close.assert_called_once()
        self.assertTrue(narrative_log.httpx_client.is_closed)
        self.assertTrue(night_report.httpx_client.is_closed)
        # Endpoints that are only strings or None are a no-op.
        connections.close_clients({"api_base": api_base, "too": None})
        connections.close_clients({})


if __name__ == "__main__":
    unittest.main()
